module ChunkManifestsOpenJpegExt

# Decodes a JPEG2000Tile chunk with libopenjp2. The codestream it is handed
# holds one tile (see `_j2k_tilecodestream`), so the whole image libopenjp2
# returns is that tile.

using ChunkManifests
import OpenJpeg_jll

const libopenjp2 = OpenJpeg_jll.libopenjp2

const _OPJ_CODEC_J2K = Cint(0)

# `opj_image_comp_t` and `opj_image_t` from openjpeg.h (2.4 and 2.5 alike).
struct _OpjImageComp
    dx::UInt32
    dy::UInt32
    w::UInt32
    h::UInt32
    x0::UInt32
    y0::UInt32
    prec::UInt32
    bpp::UInt32
    sgnd::UInt32
    resno_decoded::UInt32
    factor::UInt32
    data::Ptr{Int32}
    alpha::UInt16
end

struct _OpjImage
    x0::UInt32
    y0::UInt32
    x1::UInt32
    y1::UInt32
    numcomps::UInt32
    color_space::Cint
    comps::Ptr{_OpjImageComp}
    icc_profile_buf::Ptr{UInt8}
    icc_profile_len::UInt32
end

# `opj_dparameters_t` is only ever filled by `opj_set_default_decoder_parameters`
# and read by libopenjp2, so it is held as opaque bytes. Its size is about 8 KiB
# (two 4096-byte path buffers and a few integers); this is ample.
const _DPARAMETERS_BYTES = 64 * 1024

# The codestream libopenjp2 reads through the stream callbacks, and the messages
# its error handler reports. One per decode, so concurrent decodes share nothing.
mutable struct _Decode
    bytes::Vector{UInt8}
    pos::Int
    errors::Vector{String}
end

function _opj_read(buffer::Ptr{UInt8}, n::Csize_t, user::Ptr{Cvoid})::Csize_t
    d = unsafe_pointer_to_objref(user)::_Decode
    available = length(d.bytes) - d.pos
    available <= 0 && return typemax(Csize_t)  # (OPJ_SIZE_T)-1: end of stream
    k = min(Int(n), available)
    GC.@preserve d unsafe_copyto!(buffer, pointer(d.bytes, d.pos + 1), k)
    d.pos += k
    return Csize_t(k)
end

function _opj_skip(n::Int64, user::Ptr{Cvoid})::Int64
    d = unsafe_pointer_to_objref(user)::_Decode
    target = clamp(d.pos + n, 0, length(d.bytes))
    skipped = target - d.pos
    d.pos = target
    return skipped
end

function _opj_seek(n::Int64, user::Ptr{Cvoid})::Cint
    d = unsafe_pointer_to_objref(user)::_Decode
    0 <= n <= length(d.bytes) || return Cint(0)
    d.pos = n
    return Cint(1)
end

function _opj_error(message::Cstring, user::Ptr{Cvoid})::Cvoid
    d = unsafe_pointer_to_objref(user)::_Decode
    push!(d.errors, rstrip(unsafe_string(message)))
    return nothing
end

# C function pointers are valid only in the process that made them.
const _CALLBACKS = Ref{NTuple{4, Ptr{Cvoid}}}()

function _fail(d::_Decode, step::AbstractString)
    detail = isempty(d.errors) ? "" : ": " * join(d.errors, "; ")
    throw(ErrorException("libopenjp2 failed to $step a JPEG 2000 tile$detail"))
end

function _decode!(dest::AbstractArray{T}, codestream::Vector{UInt8}, bounds, chunkshape) where {T}
    readcb, skipcb, seekcb, errorcb = _CALLBACKS[]
    d = _Decode(codestream, 0, String[])
    params = zeros(UInt8, _DPARAMETERS_BYTES)
    image = Ref{Ptr{_OpjImage}}(C_NULL)
    codec = ccall((:opj_create_decompress, libopenjp2), Ptr{Cvoid}, (Cint,), _OPJ_CODEC_J2K)
    codec == C_NULL && throw(ErrorException("libopenjp2 could not create a decoder"))
    stream = C_NULL
    GC.@preserve d params begin
        try
            ccall((:opj_set_error_handler, libopenjp2), Cint, (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}),
                codec, errorcb, pointer_from_objref(d))
            ccall((:opj_set_default_decoder_parameters, libopenjp2), Cvoid, (Ptr{UInt8},), params)
            ccall((:opj_setup_decoder, libopenjp2), Cint, (Ptr{Cvoid}, Ptr{UInt8}), codec, params) == 1 ||
                _fail(d, "set up the decoder for")
            stream = ccall((:opj_stream_create, libopenjp2), Ptr{Cvoid}, (Csize_t, Cint),
                Csize_t(max(length(codestream), 1)), Cint(1))
            stream == C_NULL && throw(ErrorException("libopenjp2 could not create a stream"))
            ccall((:opj_stream_set_read_function, libopenjp2), Cvoid, (Ptr{Cvoid}, Ptr{Cvoid}), stream, readcb)
            ccall((:opj_stream_set_skip_function, libopenjp2), Cvoid, (Ptr{Cvoid}, Ptr{Cvoid}), stream, skipcb)
            ccall((:opj_stream_set_seek_function, libopenjp2), Cvoid, (Ptr{Cvoid}, Ptr{Cvoid}), stream, seekcb)
            ccall((:opj_stream_set_user_data, libopenjp2), Cvoid, (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}),
                stream, pointer_from_objref(d), C_NULL)
            ccall((:opj_stream_set_user_data_length, libopenjp2), Cvoid, (Ptr{Cvoid}, UInt64),
                stream, UInt64(length(codestream)))
            ccall((:opj_read_header, libopenjp2), Cint, (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Ptr{_OpjImage}}),
                stream, codec, image) == 1 || _fail(d, "read the header of")
            ccall((:opj_decode, libopenjp2), Cint, (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{_OpjImage}),
                codec, stream, image[]) == 1 || _fail(d, "decode")
            ccall((:opj_end_decompress, libopenjp2), Cint, (Ptr{Cvoid}, Ptr{Cvoid}), codec, stream) == 1 ||
                _fail(d, "finish decoding")
            _place!(reshape(dest, chunkshape), unsafe_load(image[]), bounds)
        finally
            image[] == C_NULL || ccall((:opj_image_destroy, libopenjp2), Cvoid, (Ptr{_OpjImage},), image[])
            stream == C_NULL || ccall((:opj_stream_destroy, libopenjp2), Cvoid, (Ptr{Cvoid},), stream)
            ccall((:opj_destroy_codec, libopenjp2), Cvoid, (Ptr{Cvoid},), codec)
        end
    end
    return dest
end

# The tile's samples, row-major from the tile's corner, into the first rows and
# columns of the chunk; a tile clipped by the image edge is narrower or shorter
# than the chunk, whose remainder Zarr never reads.
function _place!(chunk::AbstractMatrix{T}, img::_OpjImage, bounds) where {T}
    img.numcomps == 1 || throw(ErrorException("JPEG 2000 tile decoded to $(img.numcomps) components, expected 1"))
    comp = unsafe_load(img.comps)
    w, h = Int(comp.w), Int(comp.h)
    (w, h) == (bounds.x1 - bounds.x0, bounds.y1 - bounds.y0) || throw(
        ErrorException("JPEG 2000 tile decoded to $w×$h, expected $(bounds.x1 - bounds.x0)×$(bounds.y1 - bounds.y0)")
    )
    (comp.sgnd == 1) == (T <: Signed) && comp.prec <= 8 * sizeof(T) || throw(
        ErrorException(
            "JPEG 2000 tile holds $(comp.prec)-bit $(comp.sgnd == 1 ? "signed" : "unsigned") samples, " *
                "which do not fit the array's $T"
        )
    )
    samples = unsafe_wrap(Array, comp.data, (w, h))
    chunk[(w + 1):end, :] .= zero(T)
    chunk[1:w, (h + 1):end] .= zero(T)
    chunk[1:w, 1:h] .= T.(samples)
    return chunk
end

function __init__()
    _CALLBACKS[] = (
        @cfunction(_opj_read, Csize_t, (Ptr{UInt8}, Csize_t, Ptr{Cvoid})),
        @cfunction(_opj_skip, Int64, (Int64, Ptr{Cvoid})),
        @cfunction(_opj_seek, Cint, (Int64, Ptr{Cvoid})),
        @cfunction(_opj_error, Cvoid, (Cstring, Ptr{Cvoid})),
    )
    ChunkManifests._J2K_DECODE[] = _decode!
    return nothing
end

end # module ChunkManifestsOpenJpegExt
