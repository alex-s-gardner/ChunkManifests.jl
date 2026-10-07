# Byte access for libhdf5: a virtual file driver that serves its reads from a
# _RangeSource, so scanning a remote object moves only the metadata it asks
# for. Nothing here knows what a manifest is; the HDF5 driver above calls
# `withrangefile` and walks the open file.
#
# libhdf5 has no public API for this. A driver is a `H5FD_class_t` of function
# pointers declared in H5FDdevelop.h, and its layout is not covered by the
# library's compatibility promise. Fields are therefore written at byte offsets
# emitted by a C program including the same headers as the libhdf5 being
# called, which keeps Julia's own padding rules out of it, and
# `_RANGE_VFD_LIBVERSIONS` gates the whole thing on the versions those offsets
# were read from. A mismatch is a segfault rather than an exception, which is
# why the gate refuses instead of trying.

const _RANGE_VFD_LIBVERSIONS = ((2, 2),)

const _H5FD_T_SIZE = 80
const _H5FD_CLASS_SIZE = 336
const _H5FD_T_MAXADDR = 40

const _CLS_VERSION = 0
const _CLS_VALUE = 4
const _CLS_NAME = 8
const _CLS_MAXADDR = 16
const _CLS_FC_DEGREE = 24
const _CLS_OPEN = 120
const _CLS_CLOSE = 128
const _CLS_CMP = 136
const _CLS_QUERY = 144
const _CLS_GET_EOA = 176
const _CLS_SET_EOA = 184
const _CLS_GET_EOF = 192
const _CLS_READ = 208
const _CLS_WRITE = 216
const _CLS_TRUNCATE = 264

# H5F_CLOSE_WEAK, which libhdf5's own file drivers declare. Left at
# H5F_CLOSE_DEFAULT, closing a file fails with "unknown file close degree".
const _H5F_CLOSE_WEAK = Int32(1)
# Driver identifiers at or above H5_VFD_RESERVED are for drivers outside the
# library.
const _RANGE_VFD_VALUE = Int32(513)
const _RANGE_VFD_NAME = "chunkmanifests_range"
const _RANGE_VFD_SCHEME = "chunkmanifests_range://"

# libhdf5 is handed an integer key, never a pointer to a Julia object, so
# nothing it holds can dangle or keep a Julia value alive.
const _RANGE_SOURCES = Dict{Int64, _RangeSource}()
# The source behind each file open through the driver, by the file's id, for
# the scan to hand prefetching hints to.
const _RANGE_FILES = Dict{HDF5.API.hid_t, _RangeSource}()
const _RANGE_LOCK = ReentrantLock()
const _RANGE_NEXTKEY = Ref{Int64}(0)
const _RANGE_DRIVER = Ref{Int64}(-1)
# Globals because libhdf5 holds these pointers for as long as the driver is
# registered.
const _RANGE_CLASS = zeros(UInt8, _H5FD_CLASS_SIZE)
const _RANGE_NAMEBUF = Vector{UInt8}(_RANGE_VFD_NAME * "\0")

_pokeptr(off, v::Ptr) = unsafe_store!(Ptr{Ptr{Nothing}}(pointer(_RANGE_CLASS) + off), v)
_poke32(off, v::Int32) = unsafe_store!(Ptr{Int32}(pointer(_RANGE_CLASS) + off), v)
_pokeu32(off, v::UInt32) = unsafe_store!(Ptr{UInt32}(pointer(_RANGE_CLASS) + off), v)
_pokeu64(off, v::UInt64) = unsafe_store!(Ptr{UInt64}(pointer(_RANGE_CLASS) + off), v)

# The key libhdf5 was given, stored just past the public header it requires a
# driver's file handle to begin with.
_rangekey(file::Ptr{Nothing}) = unsafe_load(Ptr{Int64}(file + _H5FD_T_SIZE))
_rangelookup(file::Ptr{Nothing}) = @lock _RANGE_LOCK _RANGE_SOURCES[_rangekey(file)]

# libhdf5 is not thread-safe, and HDF5.jl serializes every call into it with
# `HDF5.API.liblock`, which its finalizers also take before closing anything.
# A call made here directly, rather than through HDF5.jl, takes that lock too:
# otherwise a finalizer on another thread can enter libhdf5 alongside it.
#
# Every callback below is called from C and must return a value rather than
# throw: an exception crossing that boundary takes the process down. Each
# converts a failure into the error code libhdf5 expects and lets libhdf5
# raise it, which is what puts the reason in its error stack.
function _range_open(name::Cstring, ::Cuint, ::Int64, maxaddr::UInt64)::Ptr{Nothing}
    try
        key = tryparse(Int64, last(split(unsafe_string(name), "://")))
        key === nothing && return C_NULL
        @lock _RANGE_LOCK haskey(_RANGE_SOURCES, key) || return C_NULL
        handle = Ptr{UInt8}(Libc.malloc(_H5FD_T_SIZE + sizeof(Int64)))
        handle == C_NULL && return C_NULL
        ccall(
            :memset, Ptr{Nothing}, (Ptr{UInt8}, Cint, Csize_t),
            handle, 0, _H5FD_T_SIZE + sizeof(Int64)
        )
        unsafe_store!(Ptr{UInt64}(handle + _H5FD_T_MAXADDR), maxaddr)
        unsafe_store!(Ptr{Int64}(handle + _H5FD_T_SIZE), key)
        return Ptr{Nothing}(handle)
    catch
        return C_NULL
    end
end

function _range_close(file::Ptr{Nothing})::Cint
    try
        Libc.free(file)
        return Cint(0)
    catch
        return Cint(-1)
    end
end

function _range_cmp(a::Ptr{Nothing}, b::Ptr{Nothing})::Cint
    try
        return Cint(cmp(_rangekey(a), _rangekey(b)))
    catch
        return Cint(0)
    end
end

function _range_query(::Ptr{Nothing}, flags::Ptr{Culong})::Cint
    flags == C_NULL || unsafe_store!(flags, Culong(0))
    return Cint(0)
end

function _range_get_eoa(file::Ptr{Nothing}, ::Cint)::UInt64
    try
        return _rangelookup(file).eoa
    catch
        return typemax(UInt64)
    end
end

function _range_set_eoa(file::Ptr{Nothing}, ::Cint, addr::UInt64)::Cint
    try
        _rangelookup(file).eoa = addr
        return Cint(0)
    catch
        return Cint(-1)
    end
end

function _range_get_eof(file::Ptr{Nothing}, ::Cint)::UInt64
    try
        return _rangelookup(file).size
    catch
        return typemax(UInt64)
    end
end

function _range_read(
        file::Ptr{Nothing}, ::Cint, ::Int64, addr::UInt64, size::Csize_t, buffer::Ptr{UInt8}
    )::Cint
    try
        size == 0 && return Cint(0)
        _rangefill!(_rangelookup(file), buffer, addr, UInt64(size))
        return Cint(0)
    catch
        return Cint(-1)
    end
end

# A read-only driver still has to define both: `H5FDregister` refuses a class
# missing either `read` or `write`. Both refuse rather than quietly succeeding.
_range_write(::Ptr{Nothing}, ::Cint, ::Int64, ::UInt64, ::Csize_t, ::Ptr{UInt8})::Cint =
    Cint(-1)
_range_truncate(::Ptr{Nothing}, ::Int64, ::Bool)::Cint = Cint(-1)

function _rangevfdsupported()
    v = HDF5.API.h5_get_libversion()
    return (Int(v.major), Int(v.minor)) in _RANGE_VFD_LIBVERSIONS
end

# Registered on first use rather than at load time: registering touches
# libhdf5, and a driver nothing scans with should cost nothing.
function _rangedriver()
    @lock _RANGE_LOCK begin
        _RANGE_DRIVER[] >= 0 && return _RANGE_DRIVER[]
        _pokeu32(_CLS_VERSION, UInt32(1))
        _poke32(_CLS_VALUE, _RANGE_VFD_VALUE)
        _pokeptr(_CLS_NAME, Ptr{Nothing}(pointer(_RANGE_NAMEBUF)))
        _pokeu64(_CLS_MAXADDR, typemax(UInt64) - UInt64(1))
        _poke32(_CLS_FC_DEGREE, _H5F_CLOSE_WEAK)
        _pokeptr(
            _CLS_OPEN,
            @cfunction(_range_open, Ptr{Nothing}, (Cstring, Cuint, Int64, UInt64))
        )
        _pokeptr(_CLS_CLOSE, @cfunction(_range_close, Cint, (Ptr{Nothing},)))
        _pokeptr(_CLS_CMP, @cfunction(_range_cmp, Cint, (Ptr{Nothing}, Ptr{Nothing})))
        _pokeptr(_CLS_QUERY, @cfunction(_range_query, Cint, (Ptr{Nothing}, Ptr{Culong})))
        _pokeptr(_CLS_GET_EOA, @cfunction(_range_get_eoa, UInt64, (Ptr{Nothing}, Cint)))
        _pokeptr(
            _CLS_SET_EOA, @cfunction(_range_set_eoa, Cint, (Ptr{Nothing}, Cint, UInt64))
        )
        _pokeptr(_CLS_GET_EOF, @cfunction(_range_get_eof, UInt64, (Ptr{Nothing}, Cint)))
        _pokeptr(
            _CLS_READ,
            @cfunction(
                _range_read, Cint,
                (Ptr{Nothing}, Cint, Int64, UInt64, Csize_t, Ptr{UInt8})
            )
        )
        _pokeptr(
            _CLS_WRITE,
            @cfunction(
                _range_write, Cint,
                (Ptr{Nothing}, Cint, Int64, UInt64, Csize_t, Ptr{UInt8})
            )
        )
        _pokeptr(
            _CLS_TRUNCATE, @cfunction(_range_truncate, Cint, (Ptr{Nothing}, Int64, Bool))
        )
        id = @lock HDF5.API.liblock ccall(
            (:H5FDregister, HDF5.API.libhdf5), Int64, (Ptr{Nothing},),
            pointer(_RANGE_CLASS)
        )
        id < 0 && error("H5FDregister rejected the range driver class")
        _RANGE_DRIVER[] = id
        return id
    end
end

"""
    withrangefile(f, access::RangeAccess, source::_RangeSource)

Opens `source` through the range driver and hands the open `HDF5.File` to
`f`, keeping the source registered for exactly as long as libhdf5 holds it.
Metadata is prefetched while it is open (see src/access/h5prefetch.jl), and
prefetches may still be in flight when this returns; `_drainprefetches!`
waits for them.
"""
function withrangefile(f::Function, access::RangeAccess, source::_RangeSource)
    uri = source.uri
    _rangevfdsupported() || throw(
        ArgumentError(
            "RangeAccess cannot scan $(repr(uri)): it drives libhdf5 through a virtual " *
                "file driver whose struct layout is not stable public API, and this " *
                "libhdf5 $(HDF5.API.h5_get_libversion()) is not one of the versions that " *
                "layout has been verified against. Scan with DownloadAccess(), which " *
                "fetches the object once and works anywhere",
        )
    )
    driver = _rangedriver()
    source.h5sizes === nothing && (source.h5sizes = _h5sizes(source))
    key = @lock _RANGE_LOCK begin
        _RANGE_NEXTKEY[] += 1
        k = _RANGE_NEXTKEY[]
        _RANGE_SOURCES[k] = source
        k
    end
    fapl = HDF5.API.h5p_create(HDF5.API.H5P_FILE_ACCESS)
    return try
        status = @lock HDF5.API.liblock ccall(
            (:H5Pset_driver, HDF5.API.libhdf5), Cint, (Int64, Int64, Ptr{Nothing}),
            fapl, driver, C_NULL
        )
        status < 0 && error("H5Pset_driver rejected the range driver")
        if access.pagebuffer > 0
            # Metadata written in aggregated pages then arrives in a few large
            # aligned reads instead of many small scattered ones.
            @lock HDF5.API.liblock ccall(
                (:H5Pset_page_buffer_size, HDF5.API.libhdf5), Cint,
                (Int64, Csize_t, Cuint, Cuint),
                fapl, Csize_t(access.pagebuffer), Cuint(0), Cuint(0)
            )
        end
        # `h5open` is bypassed deliberately: it decides whether a name is
        # openable by matching known schemes and otherwise stat-ing it, which a
        # name only this driver understands fails, and it closes the property
        # list it is handed.
        fid = HDF5.API.h5f_open(
            _RANGE_VFD_SCHEME * string(key), HDF5.API.H5F_ACC_RDONLY, fapl
        )
        file = HDF5.File(fid, uri)
        @lock _RANGE_LOCK _RANGE_FILES[fid] = source
        try
            f(file)
        finally
            @lock _RANGE_LOCK delete!(_RANGE_FILES, fid)
            close(file)
        end
    finally
        HDF5.API.h5p_close(fapl)
        @lock _RANGE_LOCK delete!(_RANGE_SOURCES, key)
    end
end
