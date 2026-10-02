# HTTPTransport: byte ranges from an HTTP(S) server, read with ranged GET
# requests over a shared, connection-pooling HTTP.Client. ByteRange offsets
# are zero-based and half-open; HTTP's Range header is inclusive on both
# ends, so every request below converts [offset, offset+nbytes) to
# "bytes=offset-(offset+nbytes-1)".

function _httpuri(uri::AbstractString)
    (startswith(uri, "http://") || startswith(uri, "https://")) || throw(ArgumentError(
        "not an HTTP(S) URI: $(repr(uri))"
    ))
    return uri
end

"""
    HTTPTransport(; retries=3, headers=Pair{String,String}[], kwargs...)

Reads byte ranges from an HTTP(S) server with ranged GET requests. Every
[`fetchrange`](@ref) call made through one `HTTPTransport` shares the same
underlying `HTTP.Client`, so TCP/TLS connections to a given host are reused
across calls instead of being re-established per request.

`retries` bounds how many times a transiently failing request (a 5xx
response, a request timeout, or a dropped connection) is retried, with
exponential backoff, before giving up; a 4xx response is never retried.
`headers` are attached to every request issued through this transport (for
example an `Authorization` header for a private archive). Remaining keyword
arguments are forwarded to `HTTP.Client` (`connect_timeout`,
`request_timeout`, and so on).
"""
struct HTTPTransport <: AbstractTransport
    client::HTTP.Client
    retries::Int
end

function HTTPTransport(; retries::Integer=3, headers=Pair{String,String}[], kwargs...)
    retries >= 0 || throw(ArgumentError("retries must be nonnegative, got $retries"))
    client = HTTP.Client(; default_headers=headers, kwargs...)
    return HTTPTransport(client, Int(retries))
end

"""
    fetchrange(t::HTTPTransport, uri, r::ByteRange) -> Vector{UInt8}

Fetch `r` from `uri` with a single ranged GET.

A `206` response is the well-formed case: the server honored `Range` and the
body is exactly `r`. A `200` response means the server ignored `Range` and
sent the whole resource instead — a real failure mode with misconfigured
servers and CDNs. Rather than returning that whole body as if it were the
requested range, it is sliced locally to `r`; if it is too short to even
contain `r`, that is a short read like any other and throws. Any other
status throws, naming `uri`, `r` and the status. The final body length is
always checked against `r.nbytes` before returning.
"""
function fetchrange(t::HTTPTransport, uri::AbstractString, r::ByteRange)
    _httpuri(uri)
    r.nbytes == 0 && return UInt8[]

    lastbyte = r.offset + r.nbytes - 1
    rangeheader = "bytes=$(r.offset)-$(lastbyte)"

    resp = try
        HTTP.get(
            uri, ["Range" => rangeheader];
            client=t.client, retries=t.retries, status_exception=false,
        )
    catch err
        throw(ErrorException(
            "HTTP request failed fetching range $r ($rangeheader) from $(repr(uri)): $err"
        ))
    end

    body = if resp.status == 206
        resp.body
    elseif resp.status == 200
        stop = r.offset + r.nbytes
        length(resp.body) >= stop || throw(ErrorException(
            "HTTP server at $(repr(uri)) ignored Range header $rangeheader and " *
            "returned only $(length(resp.body)) bytes with status 200, fewer " *
            "than the $stop bytes needed to satisfy range $r",
        ))
        resp.body[(r.offset + 1):stop]
    else
        throw(ErrorException(
            "HTTP $(resp.status) fetching range $r ($rangeheader) from $(repr(uri))"
        ))
    end

    length(body) == r.nbytes || throw(ErrorException(
        "short read from $(repr(uri)): requested $(r.nbytes) bytes for range $r, " *
        "got $(length(body)) bytes (HTTP status $(resp.status))",
    ))
    return body
end
