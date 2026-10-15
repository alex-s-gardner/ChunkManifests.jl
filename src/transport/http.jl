# HTTPTransport: byte ranges from an HTTP(S) server, read with ranged GET
# requests on libcurl's multi interface (`curl.jl`). ByteRange offsets are
# zero-based and half-open; HTTP's Range header is inclusive on both ends, so
# every request below converts [offset, offset+nbytes) to
# "bytes=offset-(offset+nbytes-1)".

function _httpuri(uri::AbstractString)
    (startswith(uri, "http://") || startswith(uri, "https://")) || throw(
        ArgumentError(
            "not an HTTP(S) URI: $(repr(uri))"
        )
    )
    return uri
end

# Where each URL a transport has read leads, when it follows redirects once rather than per request.
struct _Redirects
    final::Dict{String, String}
    lock::ReentrantLock
end
_Redirects() = _Redirects(Dict{String, String}(), ReentrantLock())

"""
    HTTPTransport(; retries=3, headers=Pair{String,String}[], connect_timeout=10,
                  read_idle_timeout=60, resolve=false)

Reads byte ranges from an HTTP(S) server with ranged GET requests, made with
libcurl. The ranges of one [`fetchranges`](@ref) call run concurrently on one
libcurl multi handle — up to [`concurrency`](@ref) at a time — and multi handles
are pooled, so TCP/TLS connections to a host are reused across calls instead of
being re-established per request. libcurl rather than HTTP.jl because of
throughput: on one link, a 127 MB object came at 16 MB/s through libcurl with
1, 8 or 32 requests in flight, and at 3 MB/s through HTTP.jl however many.

`retries` bounds how many times a transiently failing request (a 5xx
response, a timeout, or a dropped connection) is retried, with exponential
backoff, before giving up; a 4xx response is never retried. `headers` are
attached to every request issued through this transport (for example an
`Authorization` header for a private archive). Redirects are followed, and
libcurl does not send a caller's `Authorization` header on to another host.
Credentials for a redirect chain such as Earthdata Login's are read from
`~/.netrc`, with cookies kept for the chain.

`connect_timeout` bounds, in seconds, establishing a connection including its
TLS handshake, and `read_idle_timeout` how long a response may stall; a
handshake a server never answers would otherwise wait forever. `0` turns
either off. Proxies are taken from the environment, as libcurl takes them.

`resolve = true` follows each URL's redirects once, on its first read, and
reads every range after from the URL they end at — for a download behind a
login chain that ends at a pre-signed URL, such as an Earthdata archive's,
where following the chain costs seconds per request. `headers` go only to the
URL as given: the resolved one carries its own authorization, and S3 refuses
a pre-signed request that also has an `Authorization` header. A pre-signed URL
expires, and S3 then answers `403`; a range read that gets one follows the
original URL again, once, and retries.
"""
struct HTTPTransport <: AbstractTransport
    client::_CurlClient
    headers::Vector{Pair{String, String}}
    retries::Int
    redirects::Union{Nothing, _Redirects}
end

function HTTPTransport(;
        retries::Integer = 3, headers = Pair{String, String}[],
        connect_timeout::Real = 10, read_idle_timeout::Real = 60, resolve::Bool = false,
    )
    retries >= 0 || throw(ArgumentError("retries must be nonnegative, got $retries"))
    connect_timeout >= 0 || throw(ArgumentError("connect_timeout must be nonnegative, got $connect_timeout"))
    read_idle_timeout >= 0 || throw(ArgumentError("read_idle_timeout must be nonnegative, got $read_idle_timeout"))
    client = _CurlClient(; connect_timeout, read_idle_timeout)
    return HTTPTransport(client, [String(first(h)) => String(last(h)) for h in headers], Int(retries),
                         resolve ? _Redirects() : nothing)
end

_headerlines(t::HTTPTransport) = [string(k, ": ", v) for (k, v) in t.headers]

# The URL to read `uri` from, and the header lines to send it: `uri` itself and the transport's headers,
# unless the transport resolves redirects. Then it is where `uri` leads — followed on first use, or again
# when `stale` — sent no headers.
function _target(t::HTTPTransport, uri::String; stale::Bool = false)
    r = t.redirects
    r === nothing && return uri, _headerlines(t)
    if !stale
        u = @lock r.lock get(r.final, uri, "")
        isempty(u) || return u, String[]
    end
    request = _CurlRequest(uri, [_headerlines(t); "Range: bytes=0-0"])
    attempt = 0
    while true
        resp = only(_curl_get(t.client, [request]; concurrency = 1))
        if _transient(resp) && attempt < t.retries
            _backoff(attempt)
            attempt += 1
            continue
        end
        isempty(resp.error) || throw(ErrorException("HTTP GET " * uri * " failed: " * resp.error))
        # A login page answers 200 and ignores `Range`; the object itself answers 206.
        resp.status == 206 || throw(ErrorException(
            "following " * uri * " ended at HTTP " * string(resp.status) * " from " * resp.url *
            " rather than at the object; a download behind a login needs its credentials"))
        @lock r.lock r.final[uri] = resp.url
        return resp.url, String[]
    end
end

_rangeheader(r::ByteRange) = string("bytes=", r.offset, "-", r.offset + r.nbytes - 1)

_transient(resp::_CurlResponse) = !isempty(resp.error) || resp.status >= 500

_backoff(attempt::Int) = sleep(min(0.25 * 2.0^attempt, 8.0))

# A ranged GET, retried while it fails transiently. Returns the final status,
# the response's `Content-Range` (empty when it has none), and the body.
function _curlget(client::_CurlClient, lines::Vector{String}, retries::Int, uri::AbstractString,
                  rangeheader::AbstractString; onheaders::Union{Nothing, _HeadersDone} = nothing)
    request = _CurlRequest(uri, [lines; string("Range: ", rangeheader)]; onheaders)
    attempt = 0
    while true
        resp = only(_curl_get(client, [request]; concurrency = 1))
        if _transient(resp) && attempt < retries
            _backoff(attempt)
            attempt += 1
            continue
        end
        isempty(resp.error) || throw(
            ErrorException("HTTP request failed fetching $rangeheader from $(repr(uri)): $(resp.error)")
        )
        return resp.status, resp.contentrange, resp.body
    end
end

# A whole-object GET, with no `Range`, retried while it fails transiently; a status other than 200 throws.
function _getall(client::_CurlClient, lines::Vector{String}, retries::Int, uri::String)
    request = _CurlRequest(uri, lines)
    attempt = 0
    while true
        resp = only(_curl_get(client, [request]; concurrency = 1))
        if _transient(resp) && attempt < retries
            _backoff(attempt)
            attempt += 1
            continue
        end
        isempty(resp.error) || throw(ErrorException("HTTP request failed fetching " * uri * ": " * resp.error))
        resp.status == 200 || throw(ErrorException(
            "HTTP " * string(resp.status) * " fetching " * uri * _bodyexcerpt(resp.body)))
        return resp.body
    end
end

"""
    fetchall(t, uri) -> Vector{UInt8}

The whole object at `uri`, in one request without `Range` — for a small object from a server that may not
honor ranges, such as an API answering with a generated document.
"""
function fetchall(t::HTTPTransport, uri::AbstractString)
    url, lines = _target(t, String(_httpuri(uri)))
    return _getall(t.client, lines, t.retries, url)
end

# `requests` sent together, and those that get no response (a dropped connection, a failed TLS handshake) sent
# again, up to `retries` times with the backoff a single request gets. A response of any status is final.
function _curl_get_retrying(client::_CurlClient, requests::Vector{_CurlRequest}, retries::Int; concurrency::Int)
    responses = _curl_get(client, requests; concurrency)
    attempt = 0
    while attempt < retries
        failed = findall(r -> !isempty(r.error), responses)
        isempty(failed) && break
        _backoff(attempt)
        attempt += 1
        responses[failed] = _curl_get(client, requests[failed]; concurrency)
    end
    return responses
end

"""
    http_request(url; method = "GET", headers = String[], retries = 3) -> (status, body)

One HTTP request with no body, following redirects, as `"Name: value"` header lines; returns the final
response's status and body. A `GET` that gets no response is sent again up to `retries` times, with
exponential backoff; any other method is sent once, since a request that reached the server may have acted
before the connection failed. A request that still gets no response throws. For the odd call outside reading
objects, such as a credential exchange, so that it needs no other HTTP client.
"""
function http_request(url::AbstractString; method::AbstractString = "GET", headers::Vector{String} = String[],
                      retries::Integer = 3)
    u = String(_httpuri(url))
    lines = method == "GET" ? headers : String[headers; "Content-Length: 0"]
    resp = only(_curl_get_retrying(HTTPTransport().client, [_CurlRequest(u, lines; method)],
                                   method == "GET" ? Int(retries) : 0; concurrency = 1))
    isempty(resp.error) || throw(ErrorException("HTTP " * String(method) * " " * u * " failed: " * resp.error))
    return resp.status, resp.body
end

"""
    http_resolve(urls; headers = String[], retries = 3) -> Vector{Tuple{Int,String}}

Where each of `urls` leads: one one-byte ranged GET per URL, run concurrently, each following its
redirects, and for each the final response's status and the URL it came from. For a URL that redirects
to a pre-signed one, such as an Earthdata Login-guarded download, so the object can then be read from
there by range without the redirect chain on every request. A request that gets no response is sent again
up to `retries` times, with exponential backoff; one that still gets none throws.
"""
function http_resolve(urls::AbstractVector{<:AbstractString}; headers::Vector{String} = String[],
                      retries::Integer = 3)
    requests = [_CurlRequest(String(_httpuri(u)), [headers; "Range: bytes=0-0"]) for u in urls]
    responses = _curl_get_retrying(_CurlClient(), requests, Int(retries); concurrency = 8)
    for (u, resp) in zip(urls, responses)
        isempty(resp.error) || throw(ErrorException("HTTP GET " * String(u) * " failed: " * resp.error))
    end
    return Tuple{Int, String}[(resp.status, resp.url) for resp in responses]
end

# See `concurrency` and `maxblock` for why these differ from the defaults.
concurrency(::HTTPTransport) = 32
maxblock(::HTTPTransport) = 16 * 1024 * 1024

# Up to 300 characters of an error response's body, which for S3 and most archives says why the
# request was refused; empty when the body is empty or not text.
function _bodyexcerpt(body::Vector{UInt8})
    isempty(body) && return ""
    text = String(copy(body[1:min(end, 300)]))
    isvalid(text) || return ""
    return string(": ", replace(strip(text), r"\s+" => " "))
end

# The bytes of `r` out of a ranged response. A `206` response is the
# well-formed case: the server honored `Range` and the body is exactly `r`. A
# `200` response means the server ignored `Range` and sent the whole resource
# instead — a real failure mode with misconfigured servers and CDNs — so it is
# sliced locally to `r`, or, if too short to contain `r`, reported as a short
# read. Any other status throws.
function _rangebody(status::Int, received::Vector{UInt8}, r::ByteRange, uri::AbstractString)
    rangeheader = _rangeheader(r)
    body = if status == 206
        received
    elseif status == 200
        stop = r.offset + r.nbytes
        length(received) >= stop || throw(
            ErrorException(
                "HTTP server at $(repr(uri)) ignored Range header $rangeheader and " *
                    "returned only $(length(received)) bytes with status 200, fewer " *
                    "than the $stop bytes needed to satisfy range $(_rangeheader(r))",
            )
        )
        received[(r.offset + 1):stop]
    else
        throw(
            ErrorException(
                "HTTP $status fetching range $rangeheader from $(repr(uri))$(_bodyexcerpt(received))"
            )
        )
    end
    length(body) == r.nbytes || throw(
        ErrorException(
            "short read from $(repr(uri)): requested $(r.nbytes) bytes for range $rangeheader, " *
                "got $(length(body)) bytes (HTTP status $status)",
        )
    )
    return body
end

# Every range of `blocks` from `uri` in one batch of concurrent requests, each
# retried while it fails transiently. With `expired`, a `403` sets it and ends
# the batch, for the caller to resolve `uri` again; its result is then incomplete.
function _fetchblocks(client::_CurlClient, lines::Vector{String}, retries::Int, uri::String,
                      blocks::AbstractVector{ByteRange}, concurrency::Int;
                      expired::Union{Nothing, Base.RefValue{Bool}} = nothing)
    out = Vector{Vector{UInt8}}(undef, length(blocks))
    pending = Int[]
    for k in eachindex(blocks)
        blocks[k].nbytes == 0 ? (out[k] = UInt8[]) : push!(pending, k)
    end
    attempt = 0
    while !isempty(pending)
        requests = [_CurlRequest(uri, [lines; string("Range: ", _rangeheader(blocks[k]))]) for k in pending]
        responses = _curl_get(client, requests; concurrency)
        again = Int[]
        for (k, resp) in zip(pending, responses)
            if _transient(resp) && attempt < retries
                push!(again, k)
                continue
            end
            isempty(resp.error) || throw(
                ErrorException(
                    "HTTP request failed fetching $(_rangeheader(blocks[k])) from $(repr(uri)): $(resp.error)"
                )
            )
            if resp.status == 403 && expired !== nothing
                expired[] = true
                return out
            end
            out[k] = _rangebody(resp.status, resp.body, blocks[k], uri)
        end
        isempty(again) || (_backoff(attempt); attempt += 1)
        pending = again
    end
    return out
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
    return only(_blocks(t, String(uri), [r], 1))
end

# `blocks` of `uri` through `t`; for a transport resolving redirects, from where `uri` leads, followed
# again once if the resolved URL has expired.
function _blocks(t::HTTPTransport, uri::String, blocks::AbstractVector{ByteRange}, concurrency::Int)
    url, lines = _target(t, uri)
    t.redirects === nothing && return _fetchblocks(t.client, lines, t.retries, url, blocks, concurrency)
    expired = Ref(false)
    out = _fetchblocks(t.client, lines, t.retries, url, blocks, concurrency; expired)
    expired[] || return out
    url, lines = _target(t, uri; stale = true)
    return _fetchblocks(t.client, lines, t.retries, url, blocks, concurrency)
end

"""
    fetchranges(t::HTTPTransport, uri, ranges) -> Vector{Vector{UInt8}}

As the generic [`fetchranges`](@ref): `ranges` are coalesced with `t`'s
[`maxgap`](@ref) and [`maxblock`](@ref), and the merged blocks are fetched with
at most [`concurrency`](@ref) requests in flight. The requests run on one
libcurl multi handle, so they are concurrent without occupying a thread each.
"""
function fetchranges(t::HTTPTransport, uri::AbstractString, ranges::AbstractVector{ByteRange})
    _httpuri(uri)
    merged, mapping = coalesce_ranges(ranges; maxgap = maxgap(t), maxblock = maxblock(t))
    blocks = _blocks(t, String(uri), merged, concurrency(t))
    return _assemble(ranges, mapping, blocks)
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
    url, lines = _target(t, String(_httpuri(uri)))
    status, contentrange, _ = _curlget(t.client, lines, t.retries, url, "bytes=0-0")
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
function _clipped(client::_CurlClient, lines::Vector{String}, retries::Int, uri::AbstractString,
                  rangeheader::AbstractString; onheaders::Union{Nothing, _HeadersDone} = nothing)
    status, contentrange, body = _curlget(client, lines, retries, uri, rangeheader; onheaders)
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

# The head request's headers are in: if they announce a full head, the object
# is longer than the head, so the tail is requested now rather than after the
# head's body has arrived.
function _headers_done!(h::_HeadersDone, contentlength::Int)
    contentlength == h.head && h.tail > 0 || return nothing
    @lock h.lock begin
        h.task === nothing &&
            (h.task = Threads.@spawn _clipped(h.client, h.headers, h.retries, h.uri, "bytes=-$(h.tail)"))
    end
    return nothing
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
function _fetchends(t::HTTPTransport, given::AbstractString, head::Integer, tail::Integer)
    uri, lines = _target(t, String(_httpuri(given)))
    head == 0 && tail == 0 && return UInt8[], UInt8[], objectsize(t, given)
    if head == 0
        body, start, total = _clipped(t.client, lines, t.retries, uri, "bytes=-$tail")
        tl = min(UInt64(tail), total)
        return UInt8[], _bodyspan(body, start, total - tl, tl, uri), total
    end

    starter = _HeadersDone(t.client, lines, t.retries, String(uri), Int(head), Int(tail), ReentrantLock(), nothing)
    body, start, total = _clipped(t.client, lines, t.retries, uri, "bytes=0-$(head - 1)"; onheaders = starter)

    h = min(UInt64(head), total)
    headbytes = h == 0 ? UInt8[] : _bodyspan(body, start, UInt64(0), h, uri)
    tl = min(UInt64(tail), total - h)
    tl == 0 && return headbytes, UInt8[], total
    task = @lock starter.lock starter.task
    # The head response held the whole object, as a server ignoring `Range` sends it.
    task === nothing && return headbytes, _bodyspan(body, start, total - tl, tl, uri), total
    tbody, tstart, ttotal = fetch(task)::Tuple{Vector{UInt8}, UInt64, UInt64}
    ttotal == total || error(
        "$(repr(uri)) reported sizes $total and $ttotal in two responses; it " *
            "changed while being read",
    )
    return headbytes, _bodyspan(tbody, tstart, total - tl, tl, uri), total
end
