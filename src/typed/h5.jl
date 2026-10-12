# Typed HDF5 reads through libhdf5's C API (HDF5_jll): the datasets and attributes a caller names, each read
# as the type the caller names, and a chunked dataset opened as a [`TypedArray`](@ref) whose chunks are
# fetched through the file's transport and decoded here rather than by libhdf5.
#
# A remote file is opened through a virtual file driver that serves libhdf5's reads from range requests,
# so only the metadata a read touches moves; a chunked dataset's chunk index is walked the same way, and its
# samples are read from the object directly, concurrently, a chunk at a time.
#
# libhdf5 is not thread-safe: every call into it here holds `_H5_LOCK`. Every `ccall` names
# `HDF5_jll.libhdf5`, which the JLL sets in its `__init__`, so a relocated executable calls the library it
# carries.

import HDF5_jll

const hid_t = Int64
const herr_t = Cint
const hsize_t = UInt64
const haddr_t = UInt64

const H5P_DEFAULT = hid_t(0)
const H5S_ALL = hid_t(0)
const H5F_ACC_RDONLY = Cuint(0)
const H5S_SELECT_SET = Cint(0)
const H5T_INTEGER = Cint(0)
const H5T_FLOAT = Cint(1)
const H5T_STRING = Cint(3)
const H5T_COMPOUND = Cint(6)
const H5D_CONTIGUOUS = Cint(1)
const H5D_CHUNKED = Cint(2)
const HADDR_UNDEF = typemax(haddr_t)
const H5O_TYPE_DATASET = Cint(1)

const _H5_LOCK = ReentrantLock()
const _H5_READY = Ref(false)

# libhdf5 started once per process, with its automatic error printing off: a failed call is reported as
# an exception naming what failed, not as a stack dump on stderr.
function _h5init()
    _H5_READY[] && return nothing
    ccall((:H5open, HDF5_jll.libhdf5), herr_t, ()) < 0 && throw(ErrorException("libhdf5 did not initialize"))
    ccall((:H5Eset_auto2, HDF5_jll.libhdf5), herr_t, (hid_t, Ptr{Cvoid}, Ptr{Cvoid}), hid_t(0), C_NULL, C_NULL)
    _H5_READY[] = true
    return nothing
end

_h5check(id::hid_t, what::String) = id < 0 ? throw(ErrorException("HDF5: " * what * " failed")) : id
_h5check(status::herr_t, what::String) = status < 0 ? throw(ErrorException("HDF5: " * what * " failed")) : status

# A native type id, which libhdf5 exports as a global initialized when it starts. Read through the handle
# the JLL opened: `cglobal` with a library named at run time fails in a trimmed executable.
_h5native(sym::Symbol) = unsafe_load(Ptr{hid_t}(Libdl.dlsym(HDF5_jll.libhdf5_handle, sym)))

_h5memtype(::Type{Float64}) = _h5native(:H5T_NATIVE_DOUBLE_g)
_h5memtype(::Type{Float32}) = _h5native(:H5T_NATIVE_FLOAT_g)
_h5memtype(::Type{Int64}) = _h5native(:H5T_NATIVE_LLONG_g)
_h5memtype(::Type{Int32}) = _h5native(:H5T_NATIVE_INT_g)
_h5memtype(::Type{UInt8}) = _h5native(:H5T_NATIVE_UCHAR_g)

# ---------------------------------------------------------------------------------------------------
# Remote files: a read-only virtual file driver over range requests
# ---------------------------------------------------------------------------------------------------

# The bytes of a remote object libhdf5 reads, cached in aligned blocks, with a run of adjacent missing
# blocks fetched in one request. Both ends are fetched when the object is first opened: HDF5 keeps its
# superblock at the head and much of what a writer emits on closing at the tail. The bytes are kept for the
# session and shared by every later open of the same object, so reopening a file to read more of it
# fetches only what no earlier open read.
struct _H5Bytes
    size::UInt64
    blocksize::UInt64
    head::Vector{UInt8}
    tail::Vector{UInt8}
    tailstart::UInt64
    blocks::Dict{UInt64, Vector{UInt8}}
end

# Keyed by the object and the headers it is read with, so a read with other credentials asks the server
# again rather than being answered from what a credentialed read fetched.
const _H5BYTES = Dict{Tuple{String, Vector{Pair{String, String}}}, _H5Bytes}()

_h5bytes_key(t::LeafTransport, uri::String) =
    (uri, t isa HTTPTransport ? t.headers : t isa S3Transport ? t.headers : Pair{String, String}[])

mutable struct _H5Remote
    const transport::LeafTransport
    const uri::String
    const bytes::_H5Bytes
    eoa::UInt64
    requests::Int
end

function _h5remote(transport::LeafTransport, uri::String; head::Int = 4 * 1024 * 1024, tail::Int = 1024 * 1024,
                   blocksize::Int = 256 * 1024)
    key = _h5bytes_key(transport, uri)
    cached = get(_H5BYTES, key, nothing)
    cached === nothing || return _H5Remote(transport, uri, cached, UInt64(0), 0)
    total = objectsize(transport, uri)
    nh = min(head, total)
    h = nh > 0 ? fetchrange(transport, uri, ByteRange(0, nh)) : UInt8[]
    nt = min(tail, total - nh)
    t = nt > 0 ? fetchrange(transport, uri, ByteRange(total - nt, nt)) : UInt8[]
    bytes = _H5Bytes(UInt64(total), UInt64(blocksize), h, t, UInt64(total - nt), Dict{UInt64, Vector{UInt8}}())
    _H5BYTES[key] = bytes
    return _H5Remote(transport, uri, bytes, UInt64(0), 3)
end

function _h5fill!(r::_H5Remote, buffer::Ptr{UInt8}, addr::UInt64, size::UInt64)
    c = r.bytes
    if addr + size <= length(c.head)
        h = c.head
        GC.@preserve h unsafe_copyto!(buffer, pointer(h) + addr, size)
        return nothing
    end
    if addr >= c.tailstart && addr + size <= c.size
        t = c.tail
        GC.@preserve t unsafe_copyto!(buffer, pointer(t) + (addr - c.tailstart), size)
        return nothing
    end
    bs = c.blocksize
    b0, b1 = addr ÷ bs, (addr + size - 1) ÷ bs
    run = typemax(UInt64)
    for b in b0:(b1 + 1)
        missing_here = b <= b1 && !haskey(c.blocks, b)
        if missing_here && run == typemax(UInt64)
            run = b
        elseif !missing_here && run != typemax(UInt64)
            _h5fetchblocks!(r, run, b - 1)
            run = typemax(UInt64)
        end
    end
    written = UInt64(0)
    for b in b0:b1
        block = c.blocks[b]
        start = b * bs
        from = max(addr, start)
        to = min(addr + size, start + UInt64(length(block)))
        to <= from && continue
        GC.@preserve block unsafe_copyto!(buffer + (from - addr), pointer(block) + (from - start), to - from)
        written += to - from
    end
    written == size || throw(ErrorException("a short read assembling blocks of " * r.uri))
    return nothing
end

function _h5fetchblocks!(r::_H5Remote, from::UInt64, to::UInt64)
    c = r.bytes
    bs = c.blocksize
    start = from * bs
    stop = min((to + 1) * bs, c.size)
    bytes = fetchrange(r.transport, r.uri, ByteRange(start, stop - start))
    r.requests += 1
    for b in from:to
        lo = (b - from) * bs
        lo >= length(bytes) && break
        hi = min(lo + bs, UInt64(length(bytes)))
        c.blocks[b] = bytes[(lo + 1):hi]
    end
    return nothing
end

# The driver class: a `H5FD_class_t` of function pointers, written at the byte offsets libhdf5 2.2's
# H5FDdevelop.h gives, which are not covered by its compatibility promise — so the driver is refused on
# any other version.
const _H5VFD_LIBVERSIONS = ((2, 2),)
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
const _H5F_CLOSE_WEAK = Int32(1)
const _H5VFD_VALUE = Int32(514)
const _H5VFD_NAME = "chunkmanifests_typed"
const _H5VFD_SCHEME = "chunkmanifests_typed://"

# libhdf5 is handed an integer key, never a pointer to a Julia object.
const _H5REMOTES = Dict{Int64, _H5Remote}()
const _H5NEXTKEY = Ref{Int64}(0)
const _H5DRIVER = Ref{hid_t}(-1)
const _H5CLASS = zeros(UInt8, _H5FD_CLASS_SIZE)
const _H5NAMEBUF = Vector{UInt8}(_H5VFD_NAME * "\0")

_poke!(off::Int, v::Ptr{Cvoid}) = unsafe_store!(Ptr{Ptr{Cvoid}}(pointer(_H5CLASS) + off), v)
_poke!(off::Int, v::Int32) = unsafe_store!(Ptr{Int32}(pointer(_H5CLASS) + off), v)
_poke!(off::Int, v::UInt32) = unsafe_store!(Ptr{UInt32}(pointer(_H5CLASS) + off), v)
_poke!(off::Int, v::UInt64) = unsafe_store!(Ptr{UInt64}(pointer(_H5CLASS) + off), v)

_h5key(file::Ptr{Cvoid}) = unsafe_load(Ptr{Int64}(file + _H5FD_T_SIZE))
_h5lookup(file::Ptr{Cvoid}) = _H5REMOTES[_h5key(file)]

# Each callback is called from C and returns a code rather than throwing: an exception crossing that
# boundary would take the process down. libhdf5 raises the failure, which names it in its own error stack.
function _h5vfd_open(name::Cstring, ::Cuint, ::Int64, maxaddr::UInt64)::Ptr{Cvoid}
    try
        s = unsafe_string(name)
        startswith(s, _H5VFD_SCHEME) || return C_NULL
        key = tryparse(Int64, s[(ncodeunits(_H5VFD_SCHEME) + 1):end])
        (key === nothing || !haskey(_H5REMOTES, key)) && return C_NULL
        handle = Ptr{UInt8}(Libc.malloc(_H5FD_T_SIZE + sizeof(Int64)))
        handle == C_NULL && return C_NULL
        ccall(:memset, Ptr{Cvoid}, (Ptr{UInt8}, Cint, Csize_t), handle, 0, _H5FD_T_SIZE + sizeof(Int64))
        unsafe_store!(Ptr{UInt64}(handle + _H5FD_T_MAXADDR), maxaddr)
        unsafe_store!(Ptr{Int64}(handle + _H5FD_T_SIZE), key)
        return Ptr{Cvoid}(handle)
    catch
        return C_NULL
    end
end

_h5vfd_close(file::Ptr{Cvoid})::Cint = (Libc.free(file); Cint(0))
_h5vfd_cmp(a::Ptr{Cvoid}, b::Ptr{Cvoid})::Cint = Cint(cmp(_h5key(a), _h5key(b)))
function _h5vfd_query(::Ptr{Cvoid}, flags::Ptr{Culong})::Cint
    flags == C_NULL || unsafe_store!(flags, Culong(0))
    return Cint(0)
end
function _h5vfd_get_eoa(file::Ptr{Cvoid}, ::Cint)::UInt64
    try
        return _h5lookup(file).eoa
    catch
        return typemax(UInt64)
    end
end
function _h5vfd_set_eoa(file::Ptr{Cvoid}, ::Cint, addr::UInt64)::Cint
    try
        _h5lookup(file).eoa = addr
        return Cint(0)
    catch
        return Cint(-1)
    end
end
function _h5vfd_get_eof(file::Ptr{Cvoid}, ::Cint)::UInt64
    try
        return _h5lookup(file).bytes.size
    catch
        return typemax(UInt64)
    end
end
function _h5vfd_read(file::Ptr{Cvoid}, ::Cint, ::Int64, addr::UInt64, size::Csize_t, buffer::Ptr{UInt8})::Cint
    try
        size == 0 && return Cint(0)
        _h5fill!(_h5lookup(file), buffer, addr, UInt64(size))
        return Cint(0)
    catch
        return Cint(-1)
    end
end
# A read-only driver must still define both: `H5FDregister` refuses a class missing either.
_h5vfd_write(::Ptr{Cvoid}, ::Cint, ::Int64, ::UInt64, ::Csize_t, ::Ptr{UInt8})::Cint = Cint(-1)
_h5vfd_truncate(::Ptr{Cvoid}, ::Int64, Bool)::Cint = Cint(-1)

function _h5libversion()
    maj, min, rel = Ref{Cuint}(0), Ref{Cuint}(0), Ref{Cuint}(0)
    ccall((:H5get_libversion, HDF5_jll.libhdf5), herr_t, (Ref{Cuint}, Ref{Cuint}, Ref{Cuint}), maj, min, rel)
    return (Int(maj[]), Int(min[]), Int(rel[]))
end

# Registered on first use: registering touches libhdf5, and a process that opens only local files should
# not pay for it. Called holding `_H5_LOCK`.
function _h5driver()
    _H5DRIVER[] >= 0 && return _H5DRIVER[]
    v = _h5libversion()
    (v[1], v[2]) in _H5VFD_LIBVERSIONS || throw(ArgumentError(
        "reading a remote HDF5 file needs libhdf5 " * join([string(a, ".", b) for (a, b) in _H5VFD_LIBVERSIONS], " or ") *
        ", whose file driver layout this reader was written against; the loaded libhdf5 is " *
        string(v[1], ".", v[2], ".", v[3])))
    _poke!(_CLS_VERSION, UInt32(1))
    _poke!(_CLS_VALUE, _H5VFD_VALUE)
    _poke!(_CLS_NAME, Ptr{Cvoid}(pointer(_H5NAMEBUF)))
    _poke!(_CLS_MAXADDR, typemax(UInt64) - UInt64(1))
    _poke!(_CLS_FC_DEGREE, _H5F_CLOSE_WEAK)
    _poke!(_CLS_OPEN, @cfunction(_h5vfd_open, Ptr{Cvoid}, (Cstring, Cuint, Int64, UInt64)))
    _poke!(_CLS_CLOSE, @cfunction(_h5vfd_close, Cint, (Ptr{Cvoid},)))
    _poke!(_CLS_CMP, @cfunction(_h5vfd_cmp, Cint, (Ptr{Cvoid}, Ptr{Cvoid})))
    _poke!(_CLS_QUERY, @cfunction(_h5vfd_query, Cint, (Ptr{Cvoid}, Ptr{Culong})))
    _poke!(_CLS_GET_EOA, @cfunction(_h5vfd_get_eoa, UInt64, (Ptr{Cvoid}, Cint)))
    _poke!(_CLS_SET_EOA, @cfunction(_h5vfd_set_eoa, Cint, (Ptr{Cvoid}, Cint, UInt64)))
    _poke!(_CLS_GET_EOF, @cfunction(_h5vfd_get_eof, UInt64, (Ptr{Cvoid}, Cint)))
    _poke!(_CLS_READ, @cfunction(_h5vfd_read, Cint, (Ptr{Cvoid}, Cint, Int64, UInt64, Csize_t, Ptr{UInt8})))
    _poke!(_CLS_WRITE, @cfunction(_h5vfd_write, Cint, (Ptr{Cvoid}, Cint, Int64, UInt64, Csize_t, Ptr{UInt8})))
    _poke!(_CLS_TRUNCATE, @cfunction(_h5vfd_truncate, Cint, (Ptr{Cvoid}, Int64, Bool)))
    id = ccall((:H5FDregister, HDF5_jll.libhdf5), hid_t, (Ptr{UInt8},), _H5CLASS)
    id < 0 && throw(ErrorException("libhdf5 rejected the remote file driver"))
    _H5DRIVER[] = id
    return id
end

# ---------------------------------------------------------------------------------------------------
# Files
# ---------------------------------------------------------------------------------------------------

"""
    H5File

An HDF5 file open for typed reads: its libhdf5 id, where it is, the transport its chunks are read through,
and, for a remote file, the key its byte source is registered under (`0` for a local file).
"""
struct H5File
    fid::hid_t
    uri::String
    transport::LeafTransport
    key::Int64
end

"""
    h5open_typed(uri; transport) -> H5File

Open the HDF5 file at `uri` — a local path, an `https://` URL or an `s3://` URI — for typed reads. A remote
file is read in place: only the metadata a read touches is fetched. Close it with `close`.

libhdf5 holds per-OS-thread state for the call in progress, and a remote read waits on the network inside
that call, so the calling task is pinned to its thread while it reads.
"""
function h5open_typed(uri::AbstractString; transport::LeafTransport)
    u = String(uri)
    current_task().sticky = true
    lock(_H5_LOCK)
    try
        _h5init()
        if transport isa LocalTransport
            fid = ccall((:H5Fopen, HDF5_jll.libhdf5), hid_t, (Cstring, Cuint, hid_t), u, H5F_ACC_RDONLY, H5P_DEFAULT)
            return H5File(_h5check(fid, "opening " * u), u, transport, 0)
        end
        driver = _h5driver()
        _H5NEXTKEY[] += 1
        key = _H5NEXTKEY[]
        _H5REMOTES[key] = _h5remote(transport, u)
        fapl = _h5check(ccall((:H5Pcreate, HDF5_jll.libhdf5), hid_t, (hid_t,), _h5native(:H5P_CLS_FILE_ACCESS_ID_g)),
                        "creating a file access property list")
        try
            _h5check(ccall((:H5Pset_driver, HDF5_jll.libhdf5), herr_t, (hid_t, hid_t, Ptr{Cvoid}), fapl, driver, C_NULL),
                     "selecting the remote file driver")
            fid = ccall((:H5Fopen, HDF5_jll.libhdf5), hid_t, (Cstring, Cuint, hid_t), _H5VFD_SCHEME * string(key),
                        H5F_ACC_RDONLY, fapl)
            fid < 0 && (delete!(_H5REMOTES, key); throw(ErrorException("HDF5: opening " * u * " failed")))
            return H5File(fid, u, transport, key)
        finally
            ccall((:H5Pclose, HDF5_jll.libhdf5), herr_t, (hid_t,), fapl)
        end
    finally
        unlock(_H5_LOCK)
    end
end

function Base.close(f::H5File)
    lock(_H5_LOCK)
    try
        ccall((:H5Fclose, HDF5_jll.libhdf5), herr_t, (hid_t,), f.fid)
        f.key == 0 || delete!(_H5REMOTES, f.key)
    finally
        unlock(_H5_LOCK)
    end
    return nothing
end

# `f(dataset_id)` for the dataset at `path`, closed afterwards. Called holding `_H5_LOCK`.
function _withdataset(f::F, file::H5File, path::String) where {F}
    d = ccall((:H5Dopen2, HDF5_jll.libhdf5), hid_t, (hid_t, Cstring, hid_t), file.fid, path, H5P_DEFAULT)
    d < 0 && throw(ArgumentError(file.uri * " has no dataset " * path))
    try
        return f(d)
    finally
        ccall((:H5Dclose, HDF5_jll.libhdf5), herr_t, (hid_t,), d)
    end
end

function _locked(f::F) where {F}
    lock(_H5_LOCK)
    try
        return f()
    finally
        unlock(_H5_LOCK)
    end
end

"""
    h5_exists(file, path) -> Bool

Whether `path` names a link in `file`, every group along it included.
"""
function h5_exists(file::H5File, path::AbstractString)
    return _locked() do
        p = ""
        for part in eachsplit(String(path), '/'; keepempty = false)
            p = p * "/" * part
            ccall((:H5Lexists, HDF5_jll.libhdf5), Cint, (hid_t, Cstring, hid_t), file.fid, p, H5P_DEFAULT) > 0 ||
                return false
        end
        return true
    end
end

"""
    h5_isgroup(file, path) -> Bool

Whether `path` names a group in `file`.
"""
function h5_isgroup(file::H5File, path::AbstractString)
    return _locked() do
        g = ccall((:H5Gopen2, HDF5_jll.libhdf5), hid_t, (hid_t, Cstring, hid_t), file.fid, String(path), H5P_DEFAULT)
        g < 0 && return false
        ccall((:H5Gclose, HDF5_jll.libhdf5), herr_t, (hid_t,), g)
        return true
    end
end

# A dataset's dimensions, slowest first (HDF5's order).
function _h5dims(d::hid_t)
    s = _h5check(ccall((:H5Dget_space, HDF5_jll.libhdf5), hid_t, (hid_t,), d), "reading a dataspace")
    try
        n = ccall((:H5Sget_simple_extent_ndims, HDF5_jll.libhdf5), Cint, (hid_t,), s)
        dims = zeros(hsize_t, max(n, 0))
        n > 0 && ccall((:H5Sget_simple_extent_dims, HDF5_jll.libhdf5), Cint, (hid_t, Ptr{hsize_t}, Ptr{hsize_t}),
                       s, dims, C_NULL)
        return Int[Int(x) for x in dims]
    finally
        ccall((:H5Sclose, HDF5_jll.libhdf5), herr_t, (hid_t,), s)
    end
end

"""
    h5_dims(file, path) -> Vector{Int}

The dimensions of the dataset at `path`, slowest first; empty for a scalar.
"""
h5_dims(file::H5File, path::AbstractString) = _locked(() -> _withdataset(_h5dims, file, String(path)))

"""
    h5_read(file, path, T) -> Vector{T}

Every element of the numeric dataset at `path`, converted by libhdf5 to `T` (`Float64`, `Float32`, `Int64`,
`Int32` or `UInt8`), in HDF5's order — the last dimension fastest.
"""
function h5_read(file::H5File, path::AbstractString, ::Type{T}) where {T}
    return _locked() do
        _withdataset(file, String(path)) do d
            n = prod(_h5dims(d); init = 1)
            out = Vector{T}(undef, n)
            _h5check(ccall((:H5Dread, HDF5_jll.libhdf5), herr_t, (hid_t, hid_t, hid_t, hid_t, hid_t, Ptr{T}),
                           d, _h5memtype(T), H5S_ALL, H5S_ALL, H5P_DEFAULT, out), "reading " * String(path))
            out
        end
    end
end

"""
    h5_read_elements(file, path, T, indices) -> Vector{T}

The elements of the one-dimensional dataset at `path` at the zero-based `indices`, as `T`, without reading
the rest.
"""
function h5_read_elements(file::H5File, path::AbstractString, ::Type{T}, indices::Vector{Int}) where {T}
    return _locked() do
        _withdataset(file, String(path)) do d
            space = _h5check(ccall((:H5Dget_space, HDF5_jll.libhdf5), hid_t, (hid_t,), d), "reading a dataspace")
            mem = _h5check(ccall((:H5Screate_simple, HDF5_jll.libhdf5), hid_t, (Cint, Ptr{hsize_t}, Ptr{hsize_t}),
                                 1, hsize_t[length(indices)], C_NULL), "creating a dataspace")
            try
                coords = hsize_t[i for i in indices]
                _h5check(ccall((:H5Sselect_elements, HDF5_jll.libhdf5), herr_t, (hid_t, Cint, Csize_t, Ptr{hsize_t}),
                               space, H5S_SELECT_SET, length(indices), coords), "selecting elements")
                out = Vector{T}(undef, length(indices))
                _h5check(ccall((:H5Dread, HDF5_jll.libhdf5), herr_t, (hid_t, hid_t, hid_t, hid_t, hid_t, Ptr{T}),
                               d, _h5memtype(T), mem, space, H5P_DEFAULT, out), "reading " * String(path))
                out
            finally
                ccall((:H5Sclose, HDF5_jll.libhdf5), herr_t, (hid_t,), mem)
                ccall((:H5Sclose, HDF5_jll.libhdf5), herr_t, (hid_t,), space)
            end
        end
    end
end

# The strings a string-typed object holds, read with its own type: `n` fixed-length strings with their
# padding stripped, or variable-length ones.
function _h5strings(read!::F, t::hid_t, n::Int) where {F}
    if ccall((:H5Tis_variable_str, HDF5_jll.libhdf5), Cint, (hid_t,), t) > 0
        ptrs = Vector{Ptr{UInt8}}(undef, n)
        read!(t, pointer(ptrs))
        out = String[p == C_NULL ? "" : unsafe_string(p) for p in ptrs]
        for p in ptrs
            p == C_NULL || Libc.free(p)
        end
        return out
    end
    sz = Int(ccall((:H5Tget_size, HDF5_jll.libhdf5), Csize_t, (hid_t,), t))
    buf = zeros(UInt8, sz * n)
    GC.@preserve buf read!(t, Ptr{Ptr{UInt8}}(pointer(buf)))
    out = String[]
    for i in 0:(n - 1)
        s = buf[(i * sz + 1):((i + 1) * sz)]
        k = findfirst(iszero, s)
        push!(out, rstrip(String(s[1:(k === nothing ? sz : k - 1)])))
    end
    return out
end

"""
    h5_read_strings(file, path) -> Vector{String}

The strings the string dataset at `path` holds — one for a scalar — padding stripped.
"""
function h5_read_strings(file::H5File, path::AbstractString)
    return _locked() do
        _withdataset(file, String(path)) do d
            t = _h5check(ccall((:H5Dget_type, HDF5_jll.libhdf5), hid_t, (hid_t,), d), "reading a datatype")
            try
                ccall((:H5Tget_class, HDF5_jll.libhdf5), Cint, (hid_t,), t) == H5T_STRING ||
                    throw(ArgumentError(String(path) * " in " * file.uri * " is not a string dataset"))
                n = prod(_h5dims(d); init = 1)
                _h5strings(t, n) do mt, buf
                    _h5check(ccall((:H5Dread, HDF5_jll.libhdf5), herr_t,
                                   (hid_t, hid_t, hid_t, hid_t, hid_t, Ptr{Ptr{UInt8}}),
                                   d, mt, H5S_ALL, H5S_ALL, H5P_DEFAULT, buf), "reading " * String(path))
                end
            finally
                ccall((:H5Tclose, HDF5_jll.libhdf5), herr_t, (hid_t,), t)
            end
        end
    end
end

"""
    h5_attr_string(file, path, name) -> Union{String,Nothing}

The string attribute `name` of the object at `path`, or `nothing` where it has none.
"""
function h5_attr_string(file::H5File, path::AbstractString, name::AbstractString)
    return _locked() do
        p, n = String(path), String(name)
        ccall((:H5Aexists_by_name, HDF5_jll.libhdf5), Cint, (hid_t, Cstring, Cstring, hid_t), file.fid, p, n,
              H5P_DEFAULT) > 0 || return nothing
        a = _h5check(ccall((:H5Aopen_by_name, HDF5_jll.libhdf5), hid_t, (hid_t, Cstring, Cstring, hid_t, hid_t),
                           file.fid, p, n, H5P_DEFAULT, H5P_DEFAULT), "opening attribute " * n)
        try
            t = _h5check(ccall((:H5Aget_type, HDF5_jll.libhdf5), hid_t, (hid_t,), a), "reading a datatype")
            try
                ccall((:H5Tget_class, HDF5_jll.libhdf5), Cint, (hid_t,), t) == H5T_STRING || return nothing
                s = _h5strings(t, 1) do mt, buf
                    _h5check(ccall((:H5Aread, HDF5_jll.libhdf5), herr_t, (hid_t, hid_t, Ptr{Ptr{UInt8}}), a, mt, buf),
                             "reading attribute " * n)
                end
                return s[1]
            finally
                ccall((:H5Tclose, HDF5_jll.libhdf5), herr_t, (hid_t,), t)
            end
        finally
            ccall((:H5Aclose, HDF5_jll.libhdf5), herr_t, (hid_t,), a)
        end
    end
end

# ---------------------------------------------------------------------------------------------------
# Chunked datasets as typed arrays
# ---------------------------------------------------------------------------------------------------

# Where each chunk of a two-dimensional dataset is, gathered by `H5Dchunk_iter`.
mutable struct _H5ChunkTable
    offset::Matrix{UInt64}
    nbytes::Matrix{UInt64}
    chunk::NTuple{2, Int}
    masked::Bool
end

function _h5chunk_visit(offset::Ptr{hsize_t}, mask::Cuint, addr::haddr_t, size::hsize_t, op::Ptr{Cvoid})::Cint
    t = unsafe_pointer_to_objref(op)::_H5ChunkTable
    mask == 0 || (t.masked = true; return Cint(1))
    # HDF5's element offset is slowest first; the table is indexed (x, y), fastest first.
    y, x = Int(unsafe_load(offset, 1)) ÷ t.chunk[1], Int(unsafe_load(offset, 2)) ÷ t.chunk[2]
    t.offset[x + 1, y + 1] = addr
    t.nbytes[x + 1, y + 1] = size
    return Cint(0)
end

# The dataset's filter pipeline as one `ChunkCodec`: none, deflate, or shuffle then deflate — the
# pipelines the NISAR products use.
function _h5codec(dcpl::hid_t, elsize::Int, context::String)
    n = ccall((:H5Pget_nfilters, HDF5_jll.libhdf5), Cint, (hid_t,), dcpl)
    shuffle = 0
    zlib = false
    for i in 0:(n - 1)
        flags, nelmts, config = Ref{Cuint}(0), Ref{Csize_t}(8), Ref{Cuint}(0)
        cd = zeros(Cuint, 8)
        id = ccall((:H5Pget_filter2, HDF5_jll.libhdf5), Cint,
                   (hid_t, Cuint, Ptr{Cuint}, Ptr{Csize_t}, Ptr{Cuint}, Csize_t, Ptr{UInt8}, Ptr{Cuint}),
                   dcpl, i, flags, nelmts, cd, 0, C_NULL, config)
        if id == 2 && !zlib
            shuffle = elsize
        elseif id == 1
            zlib = true
        else
            throw(ArgumentError(context * " uses HDF5 filter " * string(id) * " in an order this reader does not " *
                                "decode; it reads shuffle and deflate"))
        end
    end
    return ChunkCodec(zlib ? CODEC_ZLIB : CODEC_NONE; shuffle)
end

"""
    h5_open_array(file, path, T) -> TypedArray{T}

The two-dimensional dataset at `path` as a [`TypedArray`](@ref) of `T`, indexed `(x, y)` — HDF5's fastest
dimension first — whose chunks are read from the file through its transport. `T` is `ComplexF32` for a
`{r, i}` compound of two 32-bit floats, or the dataset's own numeric type. A chunk the file never wrote
reads as the dataset's fill value. A contiguous dataset is read as chunks of one row each.
"""
function h5_open_array(file::H5File, path::AbstractString, ::Type{T}) where {T}
    p = String(path)
    context = p * " in " * file.uri
    return _locked() do
        _withdataset(file, p) do d
            dims = _h5dims(d)
            length(dims) == 2 || throw(ArgumentError(context * " is not two-dimensional"))
            dcpl = _h5check(ccall((:H5Dget_create_plist, HDF5_jll.libhdf5), hid_t, (hid_t,), d), "reading a property list")
            try
                layout = ccall((:H5Pget_layout, HDF5_jll.libhdf5), Cint, (hid_t,), dcpl)
                _h5checktype(d, T, context)
                fill = _h5fillvalue(dcpl, T)
                layout == H5D_CONTIGUOUS && return _h5contiguous(file, d, T, dims, fill)
                layout == H5D_CHUNKED || throw(ArgumentError(context * " is neither chunked nor contiguous"))
                cdims = zeros(hsize_t, 2)
                ccall((:H5Pget_chunk, HDF5_jll.libhdf5), Cint, (hid_t, Cint, Ptr{hsize_t}), dcpl, 2, cdims)
                chunk = (Int(cdims[1]), Int(cdims[2]))
                codec = _h5codec(dcpl, sizeof(T), context)
                ny, nx = cld(dims[1], chunk[1]), cld(dims[2], chunk[2])
                table = _H5ChunkTable(zeros(UInt64, nx, ny), zeros(UInt64, nx, ny), chunk, false)
                cb = @cfunction(_h5chunk_visit, Cint, (Ptr{hsize_t}, Cuint, haddr_t, hsize_t, Ptr{Cvoid}))
                GC.@preserve table _h5check(ccall((:H5Dchunk_iter, HDF5_jll.libhdf5), herr_t,
                                                  (hid_t, hid_t, Ptr{Cvoid}, Ptr{Cvoid}), d, H5P_DEFAULT, cb,
                                                  pointer_from_objref(table)), "walking the chunks of " * context)
                table.masked && throw(ArgumentError(context * " has a chunk HDF5 skipped filters for"))
                store = TypedChunkStore(file.transport, file.uri, table.offset, table.nbytes)
                _typedarray(T, store, (dims[2], dims[1]), (chunk[2], chunk[1]), codec, fill)
            finally
                ccall((:H5Pclose, HDF5_jll.libhdf5), herr_t, (hid_t,), dcpl)
            end
        end
    end
end

# A contiguous dataset's rows as chunks: row `y` starts `y` row lengths into the dataset's one block, which
# a dataset never written has not allocated.
function _h5contiguous(file::H5File, d::hid_t, ::Type{T}, dims::Vector{Int}, fill) where {T}
    addr = ccall((:H5Dget_offset, HDF5_jll.libhdf5), haddr_t, (hid_t,), d)
    ny, nx = dims[1], dims[2]
    rowbytes = UInt64(nx * sizeof(T))
    offset = zeros(UInt64, 1, ny)
    nbytes = zeros(UInt64, 1, ny)
    if addr != HADDR_UNDEF
        for y in 1:ny
            offset[1, y] = addr + UInt64(y - 1) * rowbytes
            nbytes[1, y] = rowbytes
        end
    end
    store = TypedChunkStore(file.transport, file.uri, offset, nbytes)
    return _typedarray(T, store, (nx, ny), (nx, 1), ChunkCodec(CODEC_NONE), fill)
end

# The memory type a dataset is read as for `T`.
function _h5memtype_for(::Type{ComplexF32})
    native_float = _h5native(:H5T_NATIVE_FLOAT_g)
    ct = _h5check(ccall((:H5Tcreate, HDF5_jll.libhdf5), hid_t, (Cint, Csize_t), H5T_COMPOUND, 8), "creating a type")
    ccall((:H5Tinsert, HDF5_jll.libhdf5), herr_t, (hid_t, Cstring, Csize_t, hid_t), ct, "r", 0, native_float)
    ccall((:H5Tinsert, HDF5_jll.libhdf5), herr_t, (hid_t, Cstring, Csize_t, hid_t), ct, "i", 4, native_float)
    return ct, true
end
_h5memtype_for(::Type{T}) where {T} = (_h5memtype(T), false)

# The file's samples must be `T`'s own bytes, since the chunks are decoded without libhdf5's conversion: a
# little-endian `{r, i}` pair of 32-bit floats for `ComplexF32`, the matching little-endian number otherwise.
function _h5checktype(d::hid_t, ::Type{T}, context::String) where {T}
    t = _h5check(ccall((:H5Dget_type, HDF5_jll.libhdf5), hid_t, (hid_t,), d), "reading a datatype")
    try
        mt, owned = _h5memtype_for(T)
        try
            same = ccall((:H5Tequal, HDF5_jll.libhdf5), Cint, (hid_t, hid_t), t, mt) > 0
            same || throw(ArgumentError(context * " does not hold samples stored as " * _h5typename(T)))
        finally
            owned && ccall((:H5Tclose, HDF5_jll.libhdf5), herr_t, (hid_t,), mt)
        end
    finally
        ccall((:H5Tclose, HDF5_jll.libhdf5), herr_t, (hid_t,), t)
    end
    return nothing
end

_h5typename(::Type{ComplexF32}) = "ComplexF32"
_h5typename(::Type{Float32}) = "Float32"
_h5typename(::Type{Float64}) = "Float64"
_h5typename(::Type{UInt8}) = "UInt8"

# The dataset's fill value as `T`, or `nothing` where it defines none.
function _h5fillvalue(dcpl::hid_t, ::Type{T}) where {T}
    status = Ref{Cint}(0)
    ccall((:H5Pfill_value_defined, HDF5_jll.libhdf5), herr_t, (hid_t, Ref{Cint}), dcpl, status)
    # H5D_FILL_VALUE_UNDEFINED is 0; the default (1) is zeros, and user-defined (2) is read.
    status[] == 0 && return nothing
    mt, owned = _h5memtype_for(T)
    try
        v = Ref{T}(zero(T))
        ccall((:H5Pget_fill_value, HDF5_jll.libhdf5), herr_t, (hid_t, hid_t, Ref{T}), dcpl, mt, v)
        return v[]
    finally
        owned && ccall((:H5Tclose, HDF5_jll.libhdf5), herr_t, (hid_t,), mt)
    end
end
