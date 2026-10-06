# Choosing and applying a SourceAccess, and fetching a remote object to a
# local cache when a driver can only open a local file.
#
# The invariant every mechanism here preserves: a manifest records the URI the
# caller named, never a cache path. A manifest whose entries pointed into a
# temporary directory would read correctly once and be useless to anyone else.

# A URI with a scheme cannot be handed to a local file open.
_isremote(uri::AbstractString) = occursin(r"^[a-zA-Z][a-zA-Z0-9+.\-]*://", uri)

"""
    resolve_access(access::SourceAccess, driver, uri) -> SourceAccess

The concrete mechanism `access` stands for when scanning `uri` with `driver`.

Every mechanism except [`AutoAccess`](@ref) resolves to itself, so a caller
who names one gets it or gets an error — never a quieter substitute that
transfers more than they asked for.
"""
resolve_access(access::SourceAccess, driver, uri::AbstractString) = access

# The transport a scan read through, which the manifest it produces then reads
# its chunks with. A manifest built through a transport that is authenticated,
# or bound to particular prefixes, is of little use if reading it falls back to
# a default one. Mechanisms that do their own I/O carry none, and leave the
# manifest its default.
_scantransport(::SourceAccess) = TransportContainers()
_scantransport(access::DownloadAccess) = access.transport
_scantransport(access::RangeAccess) = access.transport

function resolve_access(::AutoAccess, driver, uri::AbstractString)
    _isremote(uri) || return LocalAccess()
    return _remoteaccess(driver, uri)
end

# Drivers that can read a remote object in place override this. The default is
# to fetch it, which works everywhere at the cost of transferring the whole
# object.
_remoteaccess(driver, uri::AbstractString) = DownloadAccess()

"""
    withsourcepath(f, access::SourceAccess, uri) -> f(localpath)

Call `f` with a local path holding `uri`'s bytes, for the mechanisms that work
by giving a driver a local file.

A mechanism that reads an object in place has no local path to offer and does
not implement this; the driver handles it directly instead.
"""
function withsourcepath(f, access::SourceAccess, uri::AbstractString)
    throw(
        ArgumentError(
            "$(nameof(typeof(access))) does not resolve $(repr(uri)) to a local path; " *
                "a driver must open it through that mechanism itself",
        )
    )
end

function withsourcepath(f, ::LocalAccess, uri::AbstractString)
    isfile(uri) || throw(ArgumentError("no such file: $(repr(uri))"))
    return f(uri)
end

function withsourcepath(f, access::DownloadAccess, uri::AbstractString)
    _isremote(uri) || return withsourcepath(f, LocalAccess(), uri)

    dir = access.cachedir === nothing ? mktempdir() : access.cachedir
    isdir(dir) || mkpath(dir)
    local_path = joinpath(dir, _cachename(uri))

    cleanup = access.cachedir === nothing && !access.keep
    try
        isfile(local_path) || _download(access.transport, uri, local_path)
        return f(local_path)
    finally
        cleanup && rm(dir; recursive = true, force = true)
    end
end

# A cache file name that is stable for a URI and safe on every filesystem, so
# a repeated scan of the same object reuses the copy already fetched.
function _cachename(uri::AbstractString)
    base = last(split(uri, '/'))
    stem = isempty(base) ? "object" : base
    return string(string(hash(uri); base = 16), "-", stem)
end

# Fetched in blocks rather than as one range: a source worth scanning remotely
# is routinely larger than memory.
const _DOWNLOAD_BLOCK = 64 * 1024 * 1024

function _download(
        transport::AbstractTransport, uri::AbstractString, dest::AbstractString;
        blocksize::Integer = _DOWNLOAD_BLOCK,
    )
    blocksize > 0 || throw(ArgumentError("blocksize must be positive, got $blocksize"))
    total = objectsize(transport, uri)
    partial = dest * ".part"
    open(partial, "w") do io
        offset = UInt64(0)
        while offset < total
            nbytes = min(UInt64(blocksize), total - offset)
            write(io, fetchrange(transport, uri, ByteRange(offset, nbytes)))
            offset += nbytes
        end
    end
    # Renamed only once complete, so an interrupted fetch cannot leave a
    # truncated file that a later scan would mistake for a cached copy.
    mv(partial, dest; force = true)
    return dest
end
