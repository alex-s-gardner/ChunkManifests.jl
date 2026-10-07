# HTTPTransport: byte ranges from an HTTP(S) server, read with ranged GET
# requests over a shared, connection-pooling HTTP.Client. ByteRange offsets
# are zero-based and half-open; HTTP's Range header is inclusive on both
# ends, so every request below converts [offset, offset+nbytes) to
# "bytes=offset-(offset+nbytes-1)".

function _httpuri(uri::AbstractString)
    (startswith(uri, "http://") || startswith(uri, "https://")) || throw(
        ArgumentError(
            "not an HTTP(S) URI: $(repr(uri))"
        )
    )
    return uri
end

const _HTTP_IDLE_PER_HOST = 64

"""
    HTTPTransport(; retries=3, headers=Pair{String,String}[], connect_timeout=10,
                  read_idle_timeout=60, kwargs...)

Reads byte ranges from an HTTP(S) server with ranged GET requests. Every
[`fetchrange`](@ref) call made through one `HTTPTransport` shares the same
underlying `HTTP.Client`, so TCP/TLS connections to a given host are reused
across calls instead of being re-established per request. Up to
$(_HTTP_IDLE_PER_HOST) idle connections per host are kept for reuse, since a
scan or a read issues many requests at once and a new TLS connection costs
several round trips.

`retries` bounds how many times a transiently failing request (a 5xx
response, a request timeout, or a dropped connection) is retried, with
exponential backoff, before giving up; a 4xx response is never retried.
`headers` are attached to every request issued through this transport (for
example an `Authorization` header for a private archive).

`connect_timeout` bounds, in seconds, establishing a connection including its
TLS handshake, and `read_idle_timeout` how long a response may stall; a
handshake a server never answers would otherwise wait forever. `0` turns
either off. Remaining keyword arguments are forwarded to `HTTP.Client`
(`request_timeout`, `transport`, and so on).
"""
struct HTTPTransport <: AbstractTransport
    client::HTTP.Client
    retries::Int
end

function HTTPTransport(;
        retries::Integer = 3, headers = Pair{String, String}[],
        connect_timeout::Real = 10, read_idle_timeout::Real = 60,
        transport = HTTP.Transport(;
            proxy = HTTP.ProxyFromEnvironment(), max_idle_per_host = _HTTP_IDLE_PER_HOST,
        ),
        kwargs...,
    )
    retries >= 0 || throw(ArgumentError("retries must be nonnegative, got $retries"))
    client = HTTP.Client(;
        default_headers = headers, connect_timeout, read_idle_timeout, transport, kwargs...,
    )
    return HTTPTransport(client, Int(retries))
end

# See `concurrency` and `maxblock` for why these differ from the defaults.
concurrency(::HTTPTransport) = 32
maxblock(::HTTPTransport) = 16 * 1024 * 1024

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
            client = t.client, retries = t.retries, status_exception = false,
        )
    catch err
        throw(
            ErrorException(
                "HTTP request failed fetching range $r ($rangeheader) from $(repr(uri)): $err"
            )
        )
    end

    body = if resp.status == 206
        resp.body
    elseif resp.status == 200
        stop = r.offset + r.nbytes
        length(resp.body) >= stop || throw(
            ErrorException(
                "HTTP server at $(repr(uri)) ignored Range header $rangeheader and " *
                    "returned only $(length(resp.body)) bytes with status 200, fewer " *
                    "than the $stop bytes needed to satisfy range $r",
            )
        )
        resp.body[(r.offset + 1):stop]
    else
        throw(
            ErrorException(
                "HTTP $(resp.status) fetching range $r ($rangeheader) from $(repr(uri))"
            )
        )
    end

    length(body) == r.nbytes || throw(
        ErrorException(
            "short read from $(repr(uri)): requested $(r.nbytes) bytes for range $r, " *
                "got $(length(body)) bytes (HTTP status $(resp.status))",
        )
    )
    return body
end

"""
    objectsize(t::HTTPTransport, uri) -> UInt64

Total size of the object at `uri`, from the `Content-Range` of a one-byte
ranged request.

A `HEAD` would be the obvious route, but some archive hosts answer it with
`Content-Length: 0` after a redirect, which would silently report an empty
object. Asking for `bytes=0-0` costs one byte and makes the server state the
total it is serving.
"""
function objectsize(t::HTTPTransport, uri::AbstractString)
    url = _httpuri(uri)
    resp = try
        HTTP.get(
            url, ["Range" => "bytes=0-0"];
            client = t.client, retries = t.retries, status_exception = false,
        )
    catch err
        error("HTTP request failed sizing $(repr(uri)): $err")
    end

    resp.status == 206 || error(
        "HTTP $(resp.status) sizing $(repr(uri)): expected 206 with a Content-Range " *
            "header; a server that ignores Range cannot report a total size this way",
    )

    contentrange = HTTP.header(resp, "Content-Range", "")
    m = match(r"^bytes\s+\d+-\d+/(\d+)$", contentrange)
    m === nothing && error(
        "sizing $(repr(uri)): could not read a total from Content-Range " *
            repr(contentrange),
    )
    return parse(UInt64, m[1])
end

# A ranged GET whose range may run past the end of the object, which a server
# answers by clipping it. Returns the body, the offset of its first byte, and
# the object's size: a `206` states both in `Content-Range`, a `200` is a
# server ignoring `Range` and sending the whole object, and a `416` is a range
# holding no byte of an empty object, with the size in `Content-Range`.
#
# The response is streamed so that `onsize(total)` runs as soon as the headers
# arrive, before the body has, which is how a second request can be started
# on what the first reveals without waiting for its bytes.
function _httpclipped(
        t::HTTPTransport, uri::AbstractString, rangeheader::AbstractString;
        onsize = Returns(nothing),
    )
    body = UInt8[]
    start = total = UInt64(0)
    status = 0
    try
        HTTP.open(
            "GET", uri, ["Range" => rangeheader];
            client = t.client, retries = t.retries, status_exception = false,
        ) do stream
            resp = HTTP.startread(stream)
            status = resp.status
            contentrange = HTTP.header(resp, "Content-Range", "")
            if status == 206
                start, total = _contentrange(contentrange, rangeheader, uri)
                onsize(total)
            elseif status == 416
                m = match(r"^bytes\s+\*/(\d+)$", contentrange)
                m === nothing && error(
                    "HTTP 416 for $rangeheader from $(repr(uri)) carried no size in " *
                        "Content-Range: $(repr(contentrange))",
                )
                total = parse(UInt64, m[1])
                onsize(total)
            end
            body = read(stream)
        end
    catch err
        throw(
            ErrorException("HTTP request failed fetching $rangeheader from $(repr(uri)): $err")
        )
    end
    status in (200, 206, 416) ||
        error("HTTP $status fetching $rangeheader from $(repr(uri))")
    status == 200 && return body, UInt64(0), UInt64(length(body))
    status == 416 && return UInt8[], UInt64(0), total
    return body, start, total
end

"""
    _fetchends(t::HTTPTransport, uri, head, tail) -> (headbytes, tailbytes, size)

Both ends of `uri` without a request spent sizing it. The request for the
head states the object's size in its response headers, and the request for
the tail goes out as soon as they arrive, so the two overlap and the tail
asks only for bytes the head does not hold.
"""
function _fetchends(t::HTTPTransport, uri::AbstractString, head::Integer, tail::Integer)
    _httpuri(uri)
    if head == 0
        tail == 0 && return UInt8[], UInt8[], objectsize(t, uri)
        body, start, total = _httpclipped(t, uri, "bytes=-$tail")
        tl = min(UInt64(tail), total)
        return UInt8[], _bodyspan(body, start, total - tl, tl, uri), total
    end

    tailtask = Ref{Union{Nothing, Task}}(nothing)
    function starttail(total)
        tl = min(UInt64(tail), total - min(UInt64(head), total))
        tl == 0 && return nothing
        tailtask[] = Threads.@spawn _httpclipped(t, uri, "bytes=$(total - tl)-$(total - 1)")
        return nothing
    end
    body, start, total = _httpclipped(t, uri, "bytes=0-$(head - 1)"; onsize = starttail)

    h = min(UInt64(head), total)
    headbytes = h == 0 ? UInt8[] : _bodyspan(body, start, UInt64(0), h, uri)
    tl = min(UInt64(tail), total - h)
    tailbytes = if tl == 0
        UInt8[]
    elseif tailtask[] === nothing
        # The server sent the whole object in answer to the head request.
        _bodyspan(body, start, total - tl, tl, uri)
    else
        tbody, tstart, ttotal = fetch(tailtask[])
        ttotal == total || error(
            "$(repr(uri)) reported sizes $total and $ttotal in two responses; it " *
                "changed while being read",
        )
        _bodyspan(tbody, tstart, total - tl, tl, uri)
    end
    return headbytes, tailbytes, total
end
