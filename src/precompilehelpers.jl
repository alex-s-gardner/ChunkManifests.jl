# Helpers for the precompile workloads of this package's extensions, each of which scans its own
# format: a local server answering ranged GETs, and a remote scan through it. The server is local
# and touches no network.

# Answers ranged GETs for one object, as an object store does.
function _precompile_server(bytes::Vector{UInt8})
    return HTTP.serve!("127.0.0.1", 0) do req
        rangeheader = HTTP.header(req, "Range", "")
        n = length(bytes)
        m = match(r"^bytes=(\d+)-(\d+)$", rangeheader)
        a, b = if m !== nothing
            parse(Int, m[1]), min(parse(Int, m[2]), n - 1)
        else
            suffix = match(r"^bytes=-(\d+)$", rangeheader)
            suffix === nothing && return HTTP.Response(200, bytes)
            max(0, n - parse(Int, suffix[1])), n - 1
        end
        a > b && return HTTP.Response(416, ["Content-Range" => "bytes */$n"])
        return HTTP.Response(206, ["Content-Range" => "bytes $a-$b/$n"], bytes[(a + 1):(b + 1)])
    end
end

# Scans `bytes` as an object named `name` on a local HTTP server, through the
# range driver and transports a remote scan uses, and calls `readwith` on the
# group it returns.
function _precompile_remote(readwith, bytes::Vector{UInt8}, name::AbstractString; kwargs...)
    http = HTTPTransport()
    server = _precompile_server(bytes)
    try
        url = "http://127.0.0.1:$(HTTP.port(server))/$name"
        transport = TransportContainers(["http://127.0.0.1" => http])
        readwith(scan(url; transport, kwargs...))
    finally
        close(server)
    end
    return nothing
end
