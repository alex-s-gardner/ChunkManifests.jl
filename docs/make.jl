using Documenter
using ChunkManifests
using Zarr

# Doctests run against the fixture committed under `test/data`, so the output in
# the manual is the output of a real scan of a real NetCDF4 file.
DocMeta.setdocmeta!(
    ChunkManifests,
    :DocTestSetup,
    quote
        using ChunkManifests, Zarr
        NETCDF = joinpath(pkgdir(ChunkManifests), "test", "data", "antarctic_grounded_ice.nc")
    end;
    recursive = true,
)

makedocs(;
    modules = [ChunkManifests],
    authors = "Alex S. Gardner, JPL/NASA",
    sitename = "ChunkManifests.jl",
    format = Documenter.HTML(;
        canonical = "https://alex-s-gardner.github.io/ChunkManifests.jl",
        edit_link = "main",
        assets = String[],
        # The reference splices every docstring onto one page, which puts it
        # over the size Documenter warns about. One page is what makes it
        # searchable in the browser, so the warning is the thing to drop.
        size_threshold_ignore = ["api.md"],
    ),
    pages = [
        "Home" => "index.md",
        "Scanning a source" => "scanning.md",
        "Saving and loading" => "manifests.md",
        "Several files at once" => "combining.md",
        "Downstream packages" => "integration.md",
        "Remote sources" => "remote.md",
        "Fetching chunk bytes" => "transports.md",
        "Limitations" => "limitations.md",
        "How it works" => "concepts.md",
        "API reference" => "api.md",
    ],
    # Every exported name must appear in the reference, so an export added
    # without a docstring fails the build rather than going unnoticed.
    checkdocs = :exports,
    doctest = true,
)

deploydocs(;
    repo = "github.com/alex-s-gardner/ChunkManifests.jl",
    devbranch = "main",
    push_preview = true,
)
