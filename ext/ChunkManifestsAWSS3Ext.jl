module ChunkManifestsAWSS3Ext

using ChunkManifests
import AWSS3
import AWS

# Wraps an AWS config together with extra HTTP headers (currently just the
# requester-pays header) so both can travel in S3Transport's single `aws`
# field without widening the struct declared in the main package.
struct _S3Config
    aws::AWS.AbstractAWSConfig
    headers::Dict{String, String}
end

_awsconfig(aws) = aws
_awsconfig(aws::_S3Config) = aws.aws
_headers(aws) = Dict{String, String}()
_headers(aws::_S3Config) = aws.headers

"""
    S3Transport(bucket; aws=nothing, requesterpays=false)

Construct an [`S3Transport`](@ref) for `bucket`. `aws` defaults to
`AWS.current_aws_config()`. Set `requesterpays=true` to send
`x-amz-request-payer: requester` with every request, as NASA/ESA archive
buckets commonly require.
"""
function ChunkManifests.S3Transport(
        bucket::AbstractString; aws = nothing, requesterpays::Bool = false
    )
    config = aws === nothing ? AWS.current_aws_config() : aws
    wrapped = requesterpays ? _S3Config(config, Dict("x-amz-request-payer" => "requester")) : config
    return ChunkManifests.S3Transport(String(bucket), wrapped)
end

# A bare key is read from `t.bucket`. A full `s3://bucket/key` uri is read
# from the bucket it names, which may differ from `t.bucket` — a manifest's
# PathTable can legitimately span buckets. A malformed `s3://` uri (no key
# component) throws rather than guessing.
function _s3_bucket_key(t::ChunkManifests.S3Transport, uri::AbstractString)
    if startswith(uri, "s3://")
        rest = chop(uri; head = 5, tail = 0)
        parts = split(rest, '/'; limit = 2)
        length(parts) == 2 && !isempty(parts[1]) || throw(
            ArgumentError(
                "malformed s3:// uri, expected s3://bucket/key, got $(repr(uri))"
            )
        )
        return String(parts[1]), String(parts[2])
    else
        return t.bucket, uri
    end
end

# AWSS3.s3_get's byte_range is 1-based inclusive on both ends; ByteRange is
# 0-based half-open, so the last byte offset+nbytes-1 becomes offset+nbytes
# once shifted onto that convention.
_awss3_byterange(r::ChunkManifests.ByteRange) = (r.offset + 1):(r.offset + r.nbytes)

# See `concurrency` and `maxblock` for why these differ from the defaults.
ChunkManifests.concurrency(::ChunkManifests.S3Transport) = 32
ChunkManifests.maxblock(::ChunkManifests.S3Transport) = 16 * 1024 * 1024

"""
    fetchrange(t::S3Transport, uri, r::ByteRange) -> Vector{UInt8}

Read `r` from the S3 object named by `uri` (a bare key resolved against
`t.bucket`, or a full `s3://bucket/key` uri naming its own bucket). Throws if
the object is missing, access is denied, or the response is shorter than
`r.nbytes`; never substitutes empty or truncated bytes for a failed read.
"""
function ChunkManifests.fetchrange(
        t::ChunkManifests.S3Transport, uri::AbstractString, r::ChunkManifests.ByteRange
    )
    r.nbytes == 0 && return UInt8[]

    bucket, key = _s3_bucket_key(t, uri)
    config = _awsconfig(t.aws)
    headers = _headers(t.aws)
    byte_range = _awss3_byterange(r)

    body = try
        AWSS3.s3_get(config, bucket, key; raw = true, byte_range = byte_range, headers = headers)
    catch e
        error(
            "failed to read bytes [$(r.offset), $(r.offset + r.nbytes)) " *
                "from s3://$bucket/$key: $e",
        )
    end

    length(body) == r.nbytes || error(
        "short read from s3://$bucket/$key: requested $(r.nbytes) bytes at " *
            "offset $(r.offset), got $(length(body)) bytes",
    )

    return body
end

"""
    objectsize(t::S3Transport, uri) -> UInt64

Size of the object at `uri` from its `Content-Length`, via a HEAD request that
transfers no object data.
"""
function ChunkManifests.objectsize(t::ChunkManifests.S3Transport, uri::AbstractString)
    bucket, key = _s3_bucket_key(t, uri)
    headers = try
        AWSS3.s3_get_meta(_awsconfig(t.aws), bucket, key)
    catch e
        error("failed to size s3://$bucket/$key: $e")
    end
    len = get(headers, "Content-Length", get(headers, "content-length", nothing))
    len === nothing && error(
        "sizing s3://$bucket/$key: response carried no Content-Length header",
    )
    return parse(UInt64, string(len))
end

# A ranged GET returning the body, the offset of its first byte and the
# object's size from `Content-Range`. S3 clips a range running past the end of
# the object, so this also serves a head or tail longer than the object.
function _s3clipped(t::ChunkManifests.S3Transport, uri::AbstractString, rangeheader::String)
    bucket, key = _s3_bucket_key(t, uri)
    headers = merge(_headers(t.aws), Dict("Range" => rangeheader))
    r = try
        AWSS3.S3.get_object(
            bucket, key, Dict{String, Any}("headers" => headers);
            aws_config = _awsconfig(t.aws),
        )
    catch e
        error("failed to read $rangeheader from s3://$bucket/$key: $e")
    end
    contentrange = ChunkManifests.HTTP.header(r.response, "Content-Range", "")
    start, total = ChunkManifests._contentrange(contentrange, rangeheader, uri)
    return Vector{UInt8}(r.body), start, total
end

# Both ends without a request spent sizing the object: the head request and a
# suffix request for the tail run concurrently and each states the object's
# size. On an object shorter than both together the two overlap, and the tail
# returned is only what the head does not hold.
function ChunkManifests._fetchends(
        t::ChunkManifests.S3Transport, uri::AbstractString, head::Integer, tail::Integer
    )
    (head > 0 || tail > 0) || return invoke(
        ChunkManifests._fetchends,
        Tuple{ChunkManifests.AbstractTransport, AbstractString, Integer, Integer},
        t, uri, head, tail,
    )
    tailtask = tail > 0 ? Threads.@spawn(_s3clipped(t, uri, "bytes=-$tail")) : nothing
    headpart = head > 0 ? _s3clipped(t, uri, "bytes=0-$(head - 1)") : nothing
    tailpart = tailtask === nothing ? nothing : fetch(tailtask)
    total = something(headpart, tailpart)[3]
    (headpart === nothing || tailpart === nothing || tailpart[3] == total) || error(
        "$(repr(uri)) reported sizes $total and $(tailpart[3]) in two responses; it " *
            "changed while being read",
    )
    h = min(UInt64(head), total)
    tl = min(UInt64(tail), total - h)
    span(part, from, n) = n == 0 ? UInt8[] : ChunkManifests._bodyspan(part[1], part[2], from, n, uri)
    return span(headpart, UInt64(0), h), span(tailpart, total - tl, tl), total
end

end # module ChunkManifestsAWSS3Ext
