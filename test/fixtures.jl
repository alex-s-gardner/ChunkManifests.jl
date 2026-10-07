# Real-data fixtures and helpers shared across the test files.
#
# Synthetic files cover the shapes and codecs this package handles, but only a
# real granule covers what a real writer actually produces: dimension scales
# laid out as NetCDF4 writes them, a fixed-length-string grid mapping with no
# allocated storage, filter pipelines in the order libhdf5 recorded them, and
# arrays large enough that a chunk-grid error cannot hide.
#
# Both fixtures live in the repository, so every run opens them. The
# environment variable each one accepts is for pointing a run at a different
# copy, not for supplying a missing one.
#
# A fixture that resolves nowhere removes coverage rather than failing, so
# `CHUNKMANIFESTS_REQUIRE_FIXTURES` names the ones whose absence is a failure —
# which is what CI sets, so a green run cannot mean the real files were never
# opened.

const _FIXTURE_SOURCES = (
    itslive = (
        env = "CHUNKMANIFESTS_ITSLIVE",
        inrepo = "data/antarctic_grounded_ice.nc",
        what = "NetCDF4 mask: 22896x18392 UInt8 shuffle+deflate, x/y coordinate variables, fixed-length-string grid mapping",
    ),
    geotiff = (
        env = "CHUNKMANIFESTS_GEOTIFF",
        inrepo = "data/junk.tif",
        what = "a GeoTIFF written by GDAL",
    ),
)

# "" rather than `nothing` for a fixture that resolved nowhere: every caller
# gates on `isfile(path)`, which answers false for both, so one spelling of
# "missing" avoids two ways to ask the same question.
function _resolvefixture(spec)
    p = get(ENV, spec.env, "")
    isempty(p) || return isfile(p) ? p : ""
    inrepo = joinpath(@__DIR__, spec.inrepo)
    return isfile(inrepo) ? inrepo : ""
end

const ITSLIVE_PATH = _resolvefixture(_FIXTURE_SOURCES.itslive)
const GEOTIFF_JUNK_PATH = _resolvefixture(_FIXTURE_SOURCES.geotiff)

const _FIXTURE_PATHS = (itslive = ITSLIVE_PATH, geotiff = GEOTIFF_JUNK_PATH)

# Which fixtures a missing copy is a failure for. `1`, `true` or `all` requires
# every one; otherwise a comma-separated list of names. Named per fixture so
# that one too large to commit could be demanded only where it is present.
function _requiredfixtures()
    raw = lowercase(strip(get(ENV, "CHUNKMANIFESTS_REQUIRE_FIXTURES", "")))
    isempty(raw) && return Symbol[]
    raw in ("1", "true", "yes", "all") && return collect(keys(_FIXTURE_SOURCES))
    names = Symbol[]
    for part in split(raw, ',')
        name = Symbol(strip(part))
        haskey(_FIXTURE_SOURCES, name) || error(
            "CHUNKMANIFESTS_REQUIRE_FIXTURES names $(repr(string(name))), which is not a " *
                "fixture; known fixtures are $(collect(keys(_FIXTURE_SOURCES)))"
        )
        push!(names, name)
    end
    return names
end

const FIXTURES_REQUIRED = _requiredfixtures()

# Reports which fixtures resolved, so a run's real-data coverage is visible in
# the log rather than inferred from which testsets are absent.
function report_fixtures()
    missing_ = Symbol[]
    for (name, spec) in pairs(_FIXTURE_SOURCES)
        path = _FIXTURE_PATHS[name]
        if isempty(path)
            push!(missing_, name)
            @warn "real-data fixture not found; the cases needing it will not run" name spec.what spec.env
        else
            @info "real-data fixture found" name path
        end
    end
    return missing_
end

# Counts fetches through a LocalTransport, so a test can assert how much I/O a
# read performed rather than only that it was correct.
#
# `coalesce` decides what the count means, and the two readings answer different
# questions. Left true, `fetchranges` keeps its coalescing default and the count
# is the number of real requests — what test/readahead.jl asks, where merging
# byte-adjacent chunks into one read is the behavior under test. Set false, each
# range is fetched on its own and the count is the number of chunks a read
# selected — what test/rasters.jl asks, where coalescing would merge adjacent
# chunks and hide the selectivity being measured.
struct FetchCountingTransport <: AbstractTransport
    inner::LocalTransport
    count::Threads.Atomic{Int}
    coalesce::Bool
end

function FetchCountingTransport(; coalesce::Bool = true)
    return FetchCountingTransport(LocalTransport(), Threads.Atomic{Int}(0), coalesce)
end

function ChunkManifests.fetchrange(
        t::FetchCountingTransport, uri::AbstractString, r::ByteRange
    )
    Threads.atomic_add!(t.count, 1)
    return ChunkManifests.fetchrange(t.inner, uri, r)
end

function ChunkManifests.fetchranges(
        t::FetchCountingTransport, uri::AbstractString, rs::AbstractVector{ByteRange}
    )
    t.coalesce && return invoke(
        ChunkManifests.fetchranges,
        Tuple{AbstractTransport, Any, AbstractVector{ByteRange}}, t, uri, rs,
    )
    return [ChunkManifests.fetchrange(t, uri, r) for r in rs]
end

# Delegated, not counted: a size lookup is not a range fetch, and the cases
# that assert on the count are measuring reads. RangeAccess needs this to
# learn how large the object it is seeking within is.
ChunkManifests.objectsize(t::FetchCountingTransport, uri::AbstractString) =
    ChunkManifests.objectsize(t.inner, uri)

# Records the offset and bytes of every range fetched through it, in the order
# the fetches completed, so a test can see what a reader asked for and what was
# fetched ahead of it.
struct RecordingTransport{T <: AbstractTransport} <: AbstractTransport
    inner::T
    log::Vector{Tuple{UInt64, Vector{UInt8}}}
    lock::ReentrantLock
end

function ChunkManifests.fetchrange(t::RecordingTransport, uri::AbstractString, r::ByteRange)
    bytes = ChunkManifests.fetchrange(t.inner, uri, r)
    @lock t.lock push!(t.log, (r.offset, bytes))
    return bytes
end

ChunkManifests.objectsize(t::RecordingTransport, uri::AbstractString) =
    ChunkManifests.objectsize(t.inner, uri)

# A one-block AffineChunkMap pointing at `uri`, and a ManifestArray over one,
# for cases that need a chunk grid of a given shape without a file behind it.
function dummy_chunkmap(shape, chunkshape, uri)
    table = PathTable()
    push_uri!(table, uri)
    N = length(shape)
    return AffineChunkMap(
        table, cld.(shape, chunkshape), UInt64(0), ntuple(_ -> UInt64(1), N), UInt32(0)
    )
end

function dummy_manifestarray(::Type{T}, shape, chunkshape, uri; kwargs...) where {T}
    return ManifestArray{T}(dummy_chunkmap(shape, chunkshape, uri), shape, chunkshape; kwargs...)
end

dummy_manifestarray(shape, chunkshape, uri; kwargs...) =
    dummy_manifestarray(Float64, shape, chunkshape, uri; kwargs...)

# Writers for hand-built TIFF fixtures, shared by the GeoTIFF and Rasters tests.
include("tiffwriter.jl")
