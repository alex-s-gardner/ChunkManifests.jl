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

"""
    HTTPTransport(; retries=3, headers=Pair{String,String}[], connect_timeout=10,
                  read_idle_timeout=60)

Reads byte ranges from an HTTP(S) server with ranged GET requests, made with
libcurl through Downloads.jl. Every request made through one `HTTPTransport`
shares one libcurl multi handle, so TCP/TLS connections to a host are reused
across calls instead of being re-established per request, and requests from
several tasks run concurrently on it. libcurl rather than HTTP.jl because of
throughput: on one link, a 127 MB object came at 16 MB/s through libcurl with
1, 8 or 32 requests in flight, and at 3 MB/s through HTTP.jl however many.

`retries` bounds how many times a transiently failing request (a 5xx
response, a timeout, or a dropped connection) is retried, with exponential
backoff, before giving up; a 4xx response is never retried. `headers` are
attached to every request issued through this transport (for example an
`Authorization` header for a private archive). Redirects are followed, and
libcurl does not send a caller's `Authorization` header on to another host.

`connect_timeout` bounds, in seconds, establishing a connection including its
TLS handshake, and `read_idle_timeout` how long a response may stall; a
handshake a server never answers would otherwise wait forever. `0` turns
either off. Proxies are taken from the environment, as libcurl takes them.
"""
struct HTTPTransport <: AbstractTransport
    downloader::Downloads.Downloader
    headers::Vector{Pair{String, String}}
    retries::Int
end

function HTTPTransport(;
        retries::Integer = 3, headers = Pair{String, String}[],
        connect_timeout::Real = 10, read_idle_timeout::Real = 60,
    )
    retries >= 0 || throw(ArgumentError("retries must be nonnegative, got $retries"))
    connect_timeout >= 0 || throw(ArgumentError("connect_timeout must be nonnegative, got $connect_timeout"))
    read_idle_timeout >= 0 || throw(ArgumentError("read_idle_timeout must be nonnegative, got $read_idle_timeout"))
    downloader = Downloads.Downloader()
    connectms = round(Int, 1000 * connect_timeout)
    idle = ceil(Int, read_idle_timeout)
    downloader.easy_hook = (easy, _) -> begin
        connectms > 0 && Downloads.Curl.setopt(easy, LibCURL.CURLOPT_CONNECTTIMEOUT_MS, connectms)
        # Fewer than one byte a second for `idle` seconds is a stalled response.
        if idle > 0
            Downloads.Curl.setopt(easy, LibCURL.CURLOPT_LOW_SPEED_LIMIT, 1)
            Downloads.Curl.setopt(easy, LibCURL.CURLOPT_LOW_SPEED_TIME, idle)
        end
    end
    return HTTPTransport(downloader, [String(first(h)) => String(last(h)) for h in headers], Int(retries))
end

# A ranged GET, retried while it fails transiently. Returns the final status,
# the response's `Content-Range` (empty when it has none), and the body.
#
# `progress(total, now)` is Downloads.jl's: `total` is the response's
# `Content-Length` as soon as its headers arrive, before the body has.
function _httpget(t::HTTPTransport, uri::AbstractString, rangeheader::AbstractString;
                  progress = nothing)
    headers = [t.headers; "Range" => String(rangeheader)]
    attempt = 0
    while true
        body = IOBuffer()
        resp = try
            Downloads.request(uri; output = body, headers, downloader = t.downloader, throw = false, progress)
        catch err
            throw(ErrorException("HTTP request failed fetching $rangeheader from $(repr(uri)): $err"))
        end
        transient = resp isa Downloads.RequestError || resp.status >= 500
        if transient && attempt < t.retries
            sleep(min(0.25 * 2.0^attempt, 8.0))
            attempt += 1
            continue
        end
        resp isa Downloads.RequestError && throw(
            ErrorException("HTTP request failed fetching $rangeheader from $(repr(uri)): $(resp.message)")
        )
        contentrange = ""
        for (k, v) in resp.headers
            lowercase(k) == "content-range" && (contentrange = v)
        end
        return resp.status, contentrange, take!(body)
    end
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
    status, _, received = _httpget(t, uri, rangeheader)

    body = if status == 206
        received
    elseif status == 200
        stop = r.offset + r.nbytes
        length(received) >= stop || throw(
            ErrorException(
                "HTTP server at $(repr(uri)) ignored Range header $rangeheader and " *
                    "returned only $(length(received)) bytes with status 200, fewer " *
                    "than the $stop bytes needed to satisfy range $r",
            )
        )
        received[(r.offset + 1):stop]
    else
        throw(
            ErrorException(
                "HTTP $status fetching range $r ($rangeheader) from $(repr(uri))"
            )
        )
    end

    length(body) == r.nbytes || throw(
        ErrorException(
            "short read from $(repr(uri)): requested $(r.nbytes) bytes for range $r, " *
                "got $(length(body)) bytes (HTTP status $status)",
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
    status, contentrange, _ = _httpget(t, url, "bytes=0-0")
    status == 206 || error(
        "HTTP $status sizing $(repr(uri)): expected 206 with a Content-Range " *
            "header; a server that ignores Range cannot report a total size this way",
    )
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
function _httpclipped(t::HTTPTransport, uri::AbstractString, rangeheader::AbstractString;
                      progress = nothing)
    status, contentrange, body = _httpget(t, uri, rangeheader; progress)
    status in (200, 206, 416) ||
        error("HTTP $status fetching $rangeheader from $(repr(uri))")
    status == 200 && return body, UInt64(0), UInt64(length(body))
    if status == 416
        m = match(r"^bytes\s+\*/(\d+)$", contentrange)
        m === nothing && error(
            "HTTP 416 for $rangeheader from $(repr(uri)) carried no size in " *
                "Content-Range: $(repr(contentrange))",
        )
        return UInt8[], UInt64(0), parse(UInt64, m[1])
    end
    start, total = _contentrange(contentrange, rangeheader, uri)
    return body, start, total
end

"""
    _fetchends(t::HTTPTransport, uri, head, tail) -> (headbytes, tailbytes, size)

Both ends of `uri` without a request spent sizing it. The tail is a suffix
range, requested as soon as the head's headers show a full head — an object
no longer than the head is answered by the head request alone — so the two
overlap, and each response states the object's size. On an object shorter
than both together the tail repeats bytes the head holds, at most `tail` of
them.
"""
function _fetchends(t::HTTPTransport, uri::AbstractString, head::Integer, tail::Integer)
    _httpuri(uri)
    head == 0 && tail == 0 && return UInt8[], UInt8[], objectsize(t, uri)
    if head == 0
        body, start, total = _httpclipped(t, uri, "bytes=-$tail")
        tl = min(UInt64(tail), total)
        return UInt8[], _bodyspan(body, start, total - tl, tl, uri), total
    end

    tailtask = Ref{Union{Nothing, Task}}(nothing)
    lock = ReentrantLock()
    starttail(length, _) = length == head && tail > 0 && @lock lock begin
        tailtask[] === nothing && (tailtask[] = Threads.@spawn _httpclipped(t, uri, "bytes=-$tail"))
    end
    body, start, total = _httpclipped(t, uri, "bytes=0-$(head - 1)"; progress = starttail)

    h = min(UInt64(head), total)
    headbytes = h == 0 ? UInt8[] : _bodyspan(body, start, UInt64(0), h, uri)
    tl = min(UInt64(tail), total - h)
    tl == 0 && return headbytes, UInt8[], total
    task = @lock lock tailtask[]
    # The head response held the whole object, as a server ignoring `Range` sends it.
    task === nothing && return headbytes, _bodyspan(body, start, total - tl, tl, uri), total
    tbody, tstart, ttotal = fetch(task)
    ttotal == total || error(
        "$(repr(uri)) reported sizes $total and $ttotal in two responses; it " *
            "changed while being read",
    )
    return headbytes, _bodyspan(tbody, tstart, total - tl, tl, uri), total
end
