# Benchmark driver for the paths a caller hits repeatedly.
#
# The store's read path dominates any real workload: it runs once per chunk per
# read, for the life of the manifest, where scanning runs once per file ever.
# Scanning is still measured because a granule with thousands of chunks pays it
# at a size where an accidental quadratic would hurt.
#
# Run: julia --project=. benchmark/benchmarks.jl
# Compare: julia --project=. benchmark/benchmarks.jl baseline.jls
# The second form writes nothing and instead prints the delta against a
# serialized earlier run, so a change's effect is read off directly.

using ChunkManifests
using BenchmarkTools
using HDF5
import Serialization
import Zarr

const NCHUNKS_1D = 2000
const CHUNKLEN = 64

# One HDF5 file whose dataset has many chunks, which is where a per-chunk cost
# shows up. Uncompressed so the measurement is this package's work rather than
# zlib's.
function manychunk_file(dir)
    path = joinpath(dir, "many.h5")
    n = NCHUNKS_1D * CHUNKLEN
    h5open(path, "w") do f
        d = create_dataset(
            f, "v", datatype(Float64), dataspace((n,)); chunk=(CHUNKLEN,)
        )
        write(d, collect(Float64, 1:n))
    end
    return path
end

# A 3-D asymmetric grid, so a chunk-index error cannot hide in symmetry.
function grid_file(dir)
    path = joinpath(dir, "grid.h5")
    data = reshape(collect(Float64, 1:(40 * 50 * 60)), 40, 50, 60)
    h5open(path, "w") do f
        d = create_dataset(f, "g", datatype(Float64), dataspace(size(data)); chunk=(8, 10, 12))
        write(d, data)
    end
    return path
end

# Twelve slices to concatenate, each a NetCDF4-shaped variable on (x, time).
function series_files(dir)
    paths = String[]
    for k in 1:12
        path = joinpath(dir, "slice$k.h5")
        h5open(path, "w") do f
            x = create_dataset(f, "x", datatype(Int32), dataspace((16,)); chunk=(8,))
            write(x, Int32.(1:16))
            t = create_dataset(f, "time", datatype(Int32), dataspace((12,)); chunk=(6,))
            write(t, Int32.((12 * (k - 1) + 1):(12 * k)))
            d = create_dataset(
                f, "h", datatype(Float64), dataspace((16, 12)); chunk=(8, 6)
            )
            write(d, fill(Float64(k), 16, 12))
            HDF5.API.h5ds_set_scale(x, "x")
            HDF5.API.h5ds_set_scale(t, "time")
            HDF5.API.h5ds_attach_scale(d, t, 0)
            HDF5.API.h5ds_attach_scale(d, x, 1)
        end
        push!(paths, path)
    end
    return paths
end

const DIR = mktempdir()
const MANY = manychunk_file(DIR)
const GRID = grid_file(DIR)
const SLICES = series_files(DIR)

# Results the benchmark must keep producing, so a faster run that changed an
# answer is caught here rather than in review.
function checksums()
    many = scan(HDF5Driver(), MANY; siblings=false)
    grid = scan(HDF5Driver(), GRID; siblings=false)
    zm = Zarr.zopen(many)["v"]
    zg = Zarr.zopen(grid)["g"]
    combined = ChunkManifests.combine(ManifestSeries(SLICES, :time))
    zc = Zarr.zopen(combined)["h"]
    return (
        many_sum = sum(zm[:]),
        many_window = sum(zm[1000:3000]),
        grid_sum = sum(zg[:, :, :]),
        grid_window = sum(zg[3:20, 5:30, 7:40]),
        grid_shape = size(zg),
        combined_sum = sum(zc[:, :]),
        combined_shape = size(zc),
        combined_keys = sort(collect(keys(arraysof(combined)))),
        many_nfiles = length(pathtable(many)),
    )
end

function suite()
    s = BenchmarkGroup()

    s["scan"]["many chunks ($(NCHUNKS_1D))"] =
        @benchmarkable scan(HDF5Driver(), $MANY; siblings=false)
    s["scan"]["3-D grid"] = @benchmarkable scan(HDF5Driver(), $GRID; siblings=false)

    many = scan(HDF5Driver(), MANY; siblings=false)
    grid = scan(HDF5Driver(), GRID; siblings=false)
    # Readahead off: it caches across the samples of one benchmark and would
    # measure the cache rather than the read path.
    many0 = ChunkManifest(many; readahead=ReadaheadCache(; maxbytes=0))
    grid0 = ChunkManifest(grid; readahead=ReadaheadCache(; maxbytes=0))

    s["store"]["zopen"] = @benchmarkable Zarr.zopen($many0)
    s["store"]["metadata key"] = @benchmarkable $many0["v/.zarray"]
    s["store"]["one chunk key"] = @benchmarkable $many0["v/0"]
    s["store"]["subkeys (all chunks)"] = @benchmarkable Zarr.subkeys($many0, "v")
    s["store"]["storagesize"] = @benchmarkable Zarr.storagesize($many0, "v")

    zm = Zarr.zopen(many0)["v"]
    zg = Zarr.zopen(grid0)["g"]
    s["read"]["1-D full ($(NCHUNKS_1D) chunks)"] = @benchmarkable $zm[:]
    s["read"]["1-D window"] = @benchmarkable $zm[1000:3000]
    s["read"]["3-D full"] = @benchmarkable $zg[:, :, :]
    s["read"]["3-D window"] = @benchmarkable $zg[3:20, 5:30, 7:40]
    s["read"]["1-D reduction"] = @benchmarkable sum($zm)

    s["combine"]["12 slices"] =
        @benchmarkable ChunkManifests.combine(ManifestSeries($SLICES, :time))
    series = ManifestSeries(SLICES, :time)
    s["combine"]["12 slices, prescanned"] =
        @benchmarkable ChunkManifests.combine($series)
    s["merge"]["12 files"] =
        @benchmarkable ChunkManifest($SLICES; name=string.(1:12))

    s["serialize"]["save native"] = @benchmarkable ChunkManifests.save(
        joinpath(mktempdir(), "m"), $many, ZarrManifest()
    )
    saved = ChunkManifests.save(joinpath(DIR, "saved"), many, ZarrManifest())
    s["serialize"]["load native"] = @benchmarkable ChunkManifest($saved)

    return s
end

function main(args)
    @info "computing correctness checksums"
    sums = checksums()
    for (k, v) in pairs(sums)
        println("  ", rpad(k, 18), v isa AbstractVector ? v : repr(v))
    end

    @info "tuning and running"
    s = suite()
    tune!(s)
    results = run(s; verbose=false)

    flat = sort!(collect(BenchmarkTools.leaves(results)); by=first)
    baseline = isempty(args) ? nothing : Serialization.deserialize(args[1])

    println()
    println(rpad("benchmark", 46), rpad("time", 14), rpad("allocs", 12), "memory")
    for (path, trial) in flat
        label = join(path, " / ")
        t = minimum(trial)
        line = string(
            rpad(first(label, 45), 46),
            rpad(BenchmarkTools.prettytime(t.time), 14),
            rpad(string(t.allocs), 12),
            BenchmarkTools.prettymemory(t.memory),
        )
        if baseline !== nothing && haskey(baseline, path)
            b = baseline[path]
            dt = 100 * (t.time - b.time) / b.time
            da = b.allocs == 0 ? 0.0 : 100 * (t.allocs - b.allocs) / b.allocs
            line *= string("   ", sprint(show, round(dt; digits=1)), "% t  ",
                           sprint(show, round(da; digits=1)), "% a")
        end
        println(line)
    end

    out = joinpath(@__DIR__, "last.jls")
    Serialization.serialize(out, Dict(
        path => minimum(trial) for (path, trial) in flat
    ))
    Serialization.serialize(joinpath(@__DIR__, "last_checksums.jls"), sums)
    println("\nwrote ", out)
    return results
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
