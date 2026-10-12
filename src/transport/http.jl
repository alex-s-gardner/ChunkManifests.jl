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

"""
    HTTPTransport(; retries=3, headers=Pair{String,String}[], connect_timeout=10,
                  read_idle_timeout=60)

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
"""
struct HTTPTransport <: AbstractTransport
    client::_CurlClient
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
    client = _CurlClient(; connect_timeout, read_idle_timeout)
    return HTTPTransport(client, [String(first(h)) => String(last(h)) for h in headers], Int(retries))
end

_headerlines(t::HTTPTransport) = [string(k, ": ", v) for (k, v) in t.headers]

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

_httpget(t::HTTPTransport, uri::AbstractString, rangeheader::AbstractString) =
    _curlget(t.client, _headerlines(t), t.retries, uri, rangeheader)

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
fetchall(t::HTTPTransport, uri::AbstractString) =
    _getall(t.client, _headerlines(t), t.retries, String(_httpuri(uri)))

"""
    http_request(url; method = "GET", headers = String[]) -> (status, body)

One HTTP request with no body, following redirects, as `"Name: value"` header lines; returns the final
response's status and body. A request that gets no response throws. For the odd call outside reading
objects, such as a credential exchange, so that it needs no other HTTP client.
"""
function http_request(url::AbstractString; method::AbstractString = "GET", headers::Vector{String} = String[])
    u = String(_httpuri(url))
    lines = method == "GET" ? headers : String[headers; "Content-Length: 0"]
    resp = only(_curl_get(HTTPTransport().client, [_CurlRequest(u, lines; method)]; concurrency = 1))
    isempty(resp.error) || throw(ErrorException("HTTP " * String(method) * " " * u * " failed: " * resp.error))
    return resp.status, resp.body
end

"""
    http_resolve(urls; headers = String[]) -> Vector{Tuple{Int,String}}

Where each of `urls` leads: one one-byte ranged GET per URL, run concurrently, each following its
redirects, and for each the final response's status and the URL it came from. For a URL that redirects
to a pre-signed one, such as an Earthdata Login-guarded download, so the object can then be read from
there by range without the redirect chain on every request. A request that gets no response throws.
"""
function http_resolve(urls::AbstractVector{<:AbstractString}; headers::Vector{String} = String[])
    requests = [_CurlRequest(String(_httpuri(u)), [headers; "Range: bytes=0-0"]) for u in urls]
    responses = _curl_get(_CurlClient(), requests; concurrency = 8)
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
# retried while it fails transiently.
function _fetchblocks(client::_CurlClient, lines::Vector{String}, retries::Int, uri::String,
                      blocks::AbstractVector{ByteRange}, concurrency::Int)
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
    return only(_fetchblocks(t.client, _headerlines(t), t.retries, String(uri), [r], 1))
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
    blocks = _fetchblocks(t.client, _headerlines(t), t.retries, String(uri), merged, concurrency(t))
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
function _fetchends(t::HTTPTransport, uri::AbstractString, head::Integer, tail::Integer)
    _httpuri(uri)
    lines = _headerlines(t)
    head == 0 && tail == 0 && return UInt8[], UInt8[], objectsize(t, uri)
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
