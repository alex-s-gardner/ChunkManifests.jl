# A ranged-GET client on libcurl's multi interface: one call runs a batch of requests concurrently on
# the calling thread. HTTPTransport and S3Transport fetch through it.
#
# Every libcurl call names `LibCURL_jll.libcurl` itself rather than a module-level copy: the JLL sets
# that global when it loads, so a `const` taken while this package is precompiled would keep the path
# of the machine that precompiled it. And every call passes typed arguments — LibCURL.jl's generic
# `curl_easy_setopt` takes `Any...` varargs — so the client compiles without dynamic dispatch.

const _CURLE_OK = Cint(0)
const _CURLMSG_DONE = Cint(1)
const _CURL_NETRC_OPTIONAL = Clong(1)
const _CURLSSLOPT_REVOKE_BEST_EFFORT = Clong(1 << 3)
const _CURL_ERROR_SIZE = 256
const _CURL_WAIT_MS = Cint(10)

_curl_setopt(h::Ptr{Cvoid}, opt::Integer, v::Clong) =
    ccall((:curl_easy_setopt, LibCURL_jll.libcurl), Cint, (Ptr{Cvoid}, UInt32, Clong...), h, opt, v)
_curl_setopt(h::Ptr{Cvoid}, opt::Integer, v::String) =
    ccall((:curl_easy_setopt, LibCURL_jll.libcurl), Cint, (Ptr{Cvoid}, UInt32, Cstring...), h, opt, v)
_curl_setopt(h::Ptr{Cvoid}, opt::Integer, v::Ptr) =
    ccall((:curl_easy_setopt, LibCURL_jll.libcurl), Cint, (Ptr{Cvoid}, UInt32, Ptr{Cvoid}...), h, opt, v)

"""
    _CurlClient(; connect_timeout, read_idle_timeout, useragent, noproxy = "")

The options every request shares; `noproxy` lists hosts reached without the environment's proxy.
Requests run on libcurl multi handles taken from a pool, so the connections a batch opens are reused
by the next batch that takes the same handle.
"""
struct _CurlClient
    connect_ms::Int
    idle_s::Int
    useragent::String
    noproxy::String
    lock::ReentrantLock
    multis::Vector{Ptr{Cvoid}}
end

function _CurlClient(; connect_timeout::Real = 10, read_idle_timeout::Real = 60,
                     useragent::AbstractString = "ChunkManifests.jl", noproxy::AbstractString = "")
    return _CurlClient(round(Int, 1000 * connect_timeout), ceil(Int, read_idle_timeout),
                       String(useragent), String(noproxy), ReentrantLock(), Ptr{Cvoid}[])
end

"""
    _HeadersDone

Called, through [`_headers_done!`](@ref), once the headers of a successful (`2xx`) response have
all arrived, with the `Content-Length` they state (`-1` when absent). A concrete type rather than a
closure, so the call libcurl's header callback makes is resolved when the package is compiled.
[`_fetchends`](@ref) uses it to start the tail request as soon as the head's headers show an object
longer than the head; it holds what that request needs.
"""
mutable struct _HeadersDone
    client::_CurlClient
    headers::Vector{String}
    retries::Int
    uri::String
    head::Int
    tail::Int
    lock::ReentrantLock
    task::Union{Nothing, Task}
end

"""
    _CurlRequest(url, headers; method = "GET", onheaders = nothing)

One bodiless request of `url` with `headers`, each a complete `"Name: value"` line (a ranged GET
carries its `Range` among them). `onheaders` is called once a successful response's headers have
arrived; see [`_HeadersDone`](@ref).
"""
struct _CurlRequest
    url::String
    headers::Vector{String}
    method::String
    onheaders::Union{Nothing, _HeadersDone}
end
_CurlRequest(url::AbstractString, headers::Vector{String}; method::AbstractString = "GET", onheaders = nothing) =
    _CurlRequest(String(url), headers, String(method), onheaders)

"""
    _CurlResponse

`status` is the final response's HTTP status, `contentrange` its `Content-Range` header (empty when
absent) and `body` its body. `error` is empty for a completed exchange, whatever its status, and
otherwise libcurl's description of why no response arrived. `url` is the URL the final response came
from, at the end of any redirects.
"""
struct _CurlResponse
    status::Int
    contentrange::String
    body::Vector{UInt8}
    error::String
    url::String
end

# The state of one transfer, reached from libcurl's callbacks through `pointer_from_objref`.
mutable struct _CurlTransfer
    request::_CurlRequest
    easy::Ptr{Cvoid}
    slist::Ptr{Cvoid}
    errbuf::Vector{UInt8}
    body::Vector{UInt8}
    status::Int
    contentrange::String
    contentlength::Int
    url::String
end

function _curl_write(ptr::Ptr{UInt8}, size::Csize_t, n::Csize_t, user::Ptr{Cvoid})::Csize_t
    x = unsafe_pointer_to_objref(user)::_CurlTransfer
    len = size * n
    append!(x.body, unsafe_wrap(Array, ptr, len))
    return len
end

# One header line per call, including the status line of every response a redirect chain passes
# through and the empty line that ends each header block.
function _curl_header(ptr::Ptr{UInt8}, size::Csize_t, n::Csize_t, user::Ptr{Cvoid})::Csize_t
    x = unsafe_pointer_to_objref(user)::_CurlTransfer
    len = size * n
    line = rstrip(unsafe_string(ptr, len))
    if startswith(line, "HTTP/")
        parts = split(line)
        x.status = length(parts) >= 2 ? something(tryparse(Int, parts[2]), 0) : 0
        x.contentrange = ""
        x.contentlength = -1
    elseif isempty(line)
        cb = x.request.onheaders
        cb === nothing || 200 <= x.status < 300 && _headers_done!(cb, x.contentlength)
    else
        i = findfirst(':', line)
        if i !== nothing
            name = lowercase(strip(line[1:prevind(line, i)]))
            value = String(strip(line[nextind(line, i):end]))
            if name == "content-range"
                x.contentrange = value
            elseif name == "content-length"
                x.contentlength = something(tryparse(Int, value), -1)
            end
        end
    end
    return len
end

# Wait up to `_CURL_WAIT_MS` for activity on `multi`. GC-safe where Julia supports it (1.12 and later), so
# other threads keep collecting while this one waits.
@static if VERSION >= v"1.12"
    _multi_poll(multi::Ptr{Cvoid}) = @ccall gc_safe = true LibCURL_jll.libcurl.curl_multi_poll(
        multi::Ptr{Cvoid}, C_NULL::Ptr{Cvoid}, 0::Cuint, _CURL_WAIT_MS::Cint, C_NULL::Ptr{Cint})::Cint
else
    _multi_poll(multi::Ptr{Cvoid}) = @ccall LibCURL_jll.libcurl.curl_multi_poll(
        multi::Ptr{Cvoid}, C_NULL::Ptr{Cvoid}, 0::Cuint, _CURL_WAIT_MS::Cint, C_NULL::Ptr{Cint})::Cint
end

function _take_multi!(c::_CurlClient)
    @lock c.lock isempty(c.multis) || return pop!(c.multis)
    m = ccall((:curl_multi_init, LibCURL_jll.libcurl), Ptr{Cvoid}, ())
    m == C_NULL && error("curl_multi_init failed")
    return m
end
_give_multi!(c::_CurlClient, m::Ptr{Cvoid}) = (@lock c.lock push!(c.multis, m); nothing)

function _easy!(c::_CurlClient, x::_CurlTransfer)
    h = ccall((:curl_easy_init, LibCURL_jll.libcurl), Ptr{Cvoid}, ())
    h == C_NULL && error("curl_easy_init failed")
    x.easy = h
    r = x.request
    _curl_setopt(h, LibCURL.CURLOPT_URL, r.url)
    r.method == "GET" || _curl_setopt(h, LibCURL.CURLOPT_CUSTOMREQUEST, r.method)
    isempty(c.noproxy) || _curl_setopt(h, LibCURL.CURLOPT_NOPROXY, c.noproxy)
    _curl_setopt(h, LibCURL.CURLOPT_NOSIGNAL, Clong(1))
    _curl_setopt(h, LibCURL.CURLOPT_FOLLOWLOCATION, Clong(1))
    _curl_setopt(h, LibCURL.CURLOPT_MAXREDIRS, Clong(50))
    _curl_setopt(h, LibCURL.CURLOPT_USERAGENT, c.useragent)
    # `~/.netrc` and an in-memory cookie engine, as Downloads.jl sets them: Earthdata Login
    # authenticates a redirect chain this way.
    _curl_setopt(h, LibCURL.CURLOPT_NETRC, _CURL_NETRC_OPTIONAL)
    _curl_setopt(h, LibCURL.CURLOPT_COOKIEFILE, "")
    _curl_setopt(h, LibCURL.CURLOPT_SSL_OPTIONS, _CURLSSLOPT_REVOKE_BEST_EFFORT)
    ca = NetworkOptions.ca_roots_path()
    ca === nothing || _curl_setopt(h, LibCURL.CURLOPT_CAINFO, ca)
    c.connect_ms > 0 && _curl_setopt(h, LibCURL.CURLOPT_CONNECTTIMEOUT_MS, Clong(c.connect_ms))
    # Fewer than one byte a second for `idle_s` seconds is a stalled response.
    if c.idle_s > 0
        _curl_setopt(h, LibCURL.CURLOPT_LOW_SPEED_LIMIT, Clong(1))
        _curl_setopt(h, LibCURL.CURLOPT_LOW_SPEED_TIME, Clong(c.idle_s))
    end
    slist = C_NULL
    for line in r.headers
        slist = ccall((:curl_slist_append, LibCURL_jll.libcurl), Ptr{Cvoid}, (Ptr{Cvoid}, Cstring), slist, line)
    end
    x.slist = slist
    slist == C_NULL || _curl_setopt(h, LibCURL.CURLOPT_HTTPHEADER, slist)
    _curl_setopt(h, LibCURL.CURLOPT_ERRORBUFFER, pointer(x.errbuf))
    _curl_setopt(h, LibCURL.CURLOPT_WRITEFUNCTION,
                 @cfunction(_curl_write, Csize_t, (Ptr{UInt8}, Csize_t, Csize_t, Ptr{Cvoid})))
    _curl_setopt(h, LibCURL.CURLOPT_WRITEDATA, pointer_from_objref(x))
    _curl_setopt(h, LibCURL.CURLOPT_HEADERFUNCTION,
                 @cfunction(_curl_header, Csize_t, (Ptr{UInt8}, Csize_t, Csize_t, Ptr{Cvoid})))
    _curl_setopt(h, LibCURL.CURLOPT_HEADERDATA, pointer_from_objref(x))
    return h
end

# The URL a finished transfer's final response came from.
function _effective_url!(x::_CurlTransfer)
    p = Ref{Ptr{UInt8}}(C_NULL)
    ccall((:curl_easy_getinfo, LibCURL_jll.libcurl), Cint, (Ptr{Cvoid}, UInt32, Ptr{Ptr{UInt8}}...), x.easy,
          LibCURL.CURLINFO_EFFECTIVE_URL, p) == _CURLE_OK && p[] != C_NULL && (x.url = unsafe_string(p[]))
    return nothing
end

function _release!(x::_CurlTransfer)
    x.easy == C_NULL || ccall((:curl_easy_cleanup, LibCURL_jll.libcurl), Cvoid, (Ptr{Cvoid},), x.easy)
    x.slist == C_NULL || ccall((:curl_slist_free_all, LibCURL_jll.libcurl), Cvoid, (Ptr{Cvoid},), x.slist)
    x.easy = C_NULL
    x.slist = C_NULL
    return nothing
end

# `CURLMsg` from curl.h: the message kind, the easy handle, and a union whose `CURLcode` member is the
# low 32 bits of a pointer-sized word.
struct _CurlMsg
    msg::Cint
    easy::Ptr{Cvoid}
    data::UInt
end

"""
    _curl_get(c::_CurlClient, requests; concurrency) -> Vector{_CurlResponse}

Perform `requests`, at most `concurrency` at a time, and return one response per request in the
same order. The calling task waits until all have finished; while it waits it yields every few
milliseconds, so other tasks on its thread run, and from Julia 1.12 libcurl's waits are GC-safe, so
other threads keep collecting.
"""
function _curl_get(c::_CurlClient, requests::Vector{_CurlRequest}; concurrency::Int = 32)
    concurrency >= 1 || throw(ArgumentError("concurrency must be at least 1, got $concurrency"))
    n = length(requests)
    transfers = [_CurlTransfer(r, C_NULL, C_NULL, zeros(UInt8, _CURL_ERROR_SIZE), UInt8[], 0, "", -1, r.url)
                 for r in requests]
    errors = fill("", n)
    isempty(transfers) && return _CurlResponse[]
    multi = _take_multi!(c)
    running = Ref{Cint}(0)
    queued = Ref{Cint}(0)
    active = Dict{Ptr{Cvoid}, Int}()
    next = 1
    done = 0
    try
        GC.@preserve transfers begin
            while done < n
                while next <= n && length(active) < concurrency
                    h = _easy!(c, transfers[next])
                    ccall((:curl_multi_add_handle, LibCURL_jll.libcurl), Cint, (Ptr{Cvoid}, Ptr{Cvoid}), multi, h)
                    active[h] = next
                    next += 1
                end
                ccall((:curl_multi_perform, LibCURL_jll.libcurl), Cint, (Ptr{Cvoid}, Ptr{Cint}), multi, running)
                while true
                    p = ccall((:curl_multi_info_read, LibCURL_jll.libcurl), Ptr{_CurlMsg}, (Ptr{Cvoid}, Ptr{Cint}),
                              multi, queued)
                    p == C_NULL && break
                    m = unsafe_load(p)
                    m.msg == _CURLMSG_DONE || continue
                    k = active[m.easy]
                    delete!(active, m.easy)
                    code = Cint(m.data % UInt32)
                    if code != _CURLE_OK
                        x = transfers[k]
                        z = findfirst(iszero, x.errbuf)
                        detail = String(x.errbuf[1:(z === nothing ? end : z - 1)])
                        what = unsafe_string(ccall((:curl_easy_strerror, LibCURL_jll.libcurl), Cstring, (Cint,), code))
                        errors[k] = isempty(detail) ? what : string(what, ": ", detail)
                    end
                    _effective_url!(transfers[k])
                    ccall((:curl_multi_remove_handle, LibCURL_jll.libcurl), Cint, (Ptr{Cvoid}, Ptr{Cvoid}), multi, m.easy)
                    _release!(transfers[k])
                    done += 1
                end
                done < n || break
                # Wait in short slices and yield between them: a long wait would hold this thread,
                # starving every other task scheduled on it — including, in a single-threaded
                # process, a server the requests are addressed to. `curl_multi_poll` returns early
                # on activity, so the slice bounds only how long a waiting thread stays unavailable.
                _multi_poll(multi)
                yield()
            end
        end
    finally
        for (h, k) in active
            ccall((:curl_multi_remove_handle, LibCURL_jll.libcurl), Cint, (Ptr{Cvoid}, Ptr{Cvoid}), multi, h)
            _release!(transfers[k])
        end
        _give_multi!(c, multi)
    end
    return [_CurlResponse(x.status, x.contentrange, x.body, errors[k], x.url) for (k, x) in pairs(transfers)]
end
