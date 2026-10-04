# Real-data fixtures.
#
# Synthetic files cover the shapes and codecs this package handles, but only a
# real granule covers what a real writer actually produces: dimension scales
# laid out as NetCDF4 writes them, a fixed-length-string grid mapping with no
# allocated storage, filter pipelines in the order libhdf5 recorded them, and
# arrays large enough that a chunk-grid error cannot hide.
#
# Each fixture resolves from an environment variable, then a copy inside this
# repository, then the location it sits in on the maintainer's machine. A
# fixture that resolves nowhere removes coverage rather than failing, so
# `CHUNKMANIFESTS_REQUIRE_FIXTURES=1` turns that into a test failure listing
# what is missing — which is what CI sets, so a green run cannot mean the real
# files were never opened.

const _FIXTURE_SOURCES = (
    atl06 = (
        env="CHUNKMANIFESTS_ATL06",
        local_="data/ATL06_20220404104324_01881512_006_02.h5",
        fallback="/Users/gardnera/Documents/GitHub/H5ToTable.jl/data/ATL06_20220404104324_01881512_006_02.h5",
        what="ICESat-2 ATL06 granule: Float32 and Int8 datasets, deflate and shuffle+deflate pipelines, dimension scales",
    ),
    itslive = (
        env="CHUNKMANIFESTS_ITSLIVE",
        local_="data/antarctic_grounded_ice.nc",
        fallback="/Users/gardnera/Documents/GitHub/ItsLiveMasks.jl/data/antarctic_grounded_ice.nc",
        what="NetCDF4 mask: 22896x18392 UInt8 shuffle+deflate, x/y coordinate variables, fixed-length-string grid mapping",
    ),
    geotiff = (
        env="CHUNKMANIFESTS_GEOTIFF",
        local_="data/junk.tif",
        fallback="/Users/gardnera/Documents/GitHub/GRACE.jl/junk.tif",
        what="a GeoTIFF written by GDAL",
    ),
)

function _resolvefixture(spec)
    p = get(ENV, spec.env, "")
    isempty(p) || return isfile(p) ? p : nothing
    inrepo = joinpath(@__DIR__, spec.local_)
    isfile(inrepo) && return inrepo
    isfile(spec.fallback) && return spec.fallback
    return nothing
end

const FIXTURES = NamedTuple{keys(_FIXTURE_SOURCES)}(
    map(_resolvefixture, values(_FIXTURE_SOURCES))
)

# Which fixtures a missing copy is a failure for. `1`, `true` or `all` requires
# every one; otherwise a comma-separated list of names. Per-fixture because the
# two small files are committed and so are always there, while the ATL06
# granule is 37 MB and has to be fetched, so only the job that fetches it can
# demand it.
function _requiredfixtures()
    raw = lowercase(strip(get(ENV, "CHUNKMANIFESTS_REQUIRE_FIXTURES", "")))
    isempty(raw) && return Symbol[]
    raw in ("1", "true", "yes", "all") && return collect(keys(_FIXTURE_SOURCES))
    names = Symbol[]
    for part in split(raw, ',')
        name = Symbol(strip(part))
        isempty(string(name)) && continue
        haskey(_FIXTURE_SOURCES, name) || error(
            "CHUNKMANIFESTS_REQUIRE_FIXTURES names $(repr(string(name))), which is not a " *
            "fixture; known fixtures are $(collect(keys(_FIXTURE_SOURCES)))"
        )
        push!(names, name)
    end
    return names
end

const FIXTURES_REQUIRED = _requiredfixtures()

const ATL06_PATH = something(FIXTURES.atl06, "")
const ITSLIVE_PATH = something(FIXTURES.itslive, "")
const GEOTIFF_JUNK_PATH = something(FIXTURES.geotiff, "")

# Reports which fixtures resolved, so a run's real-data coverage is visible in
# the log rather than inferred from which testsets are absent.
function report_fixtures()
    missing_ = Symbol[]
    for (name, spec) in pairs(_FIXTURE_SOURCES)
        path = FIXTURES[name]
        if path === nothing
            push!(missing_, name)
            @warn "real-data fixture not found; the cases needing it will not run" name spec.what spec.env
        else
            @info "real-data fixture found" name path
        end
    end
    return missing_
end
