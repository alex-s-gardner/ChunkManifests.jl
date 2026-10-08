module ChunkManifestsRastersExt

# A group returned by `scan` or `load` reaches Rasters as an already-open, lazy
# array rather than through Rasters' FileArray.
#
# FileArray holds a filename and reopens the dataset on every readblock!; it
# exists to defer opening a file. Such a group is a manifest already in memory
# plus byte-range transports, so there is nothing left to defer, and
# routing through FileArray would discard the store instance along with its
# transport containers and its warmed readahead cache. Handing Rasters an open
# lazy DiskArray is its ordinary Raster(A, dims) path, not a special case.
#
# Everything above the data array is Rasters' own CommonDataModel machinery
# applied to the same objects Rasters applies it to itself: dimensions and CRS
# from `_dims`, attributes from `_metadata`, and CF scaling and fill-value
# masking from a `ModifiedDiskArray` over the raw variable. A raster built from
# a manifest is therefore the same object `Raster(path; lazy=true)` builds from
# a real Zarr store, down to element type and `missingval`, and the `scaled`,
# `missingval`, `coerce` and `raw` keywords mean exactly what they mean there.
# Nothing here reimplements coordinate, calendar, scaling or grid-mapping
# handling.
#
# Internals of Rasters used deliberately, each pinned by a test in
# test/rasters.jl so that a Rasters upgrade moving one fails loudly rather than
# degrading quietly: `_dims`, `_layers`, `_metadata`, `_read_missingval_pair`,
# `_mod`, `_maybe_modify`, `_outer_missingval`, `_raw_check`, and the `nokw`
# sentinel. They are the set Rasters' own lazy CommonDataModel path calls, so
# they move only when that path moves.

using ChunkManifests
using ChunkManifests: ChunkManifest
import Rasters
import Zarr
import ZarrDatasets

const CDM = ZarrDatasets.CDM

# A manifest keys its arrays by full Zarr path ("gt1l/h_li"), while
# CommonDataModel addresses a variable by name within a dataset, so a key's
# leading groups are walked here rather than passed along as part of the name.
# The split is the store's own, so a key divides here exactly as it does when
# the store resolves one.
const _splitpath = ChunkManifests._splitkey

const ManifestGroup = Zarr.ZGroup{ChunkManifest}

# The array keys under `z`, relative to it, from the group tree Zarr.jl built
# when it opened `z`.
function _relkeys(z::Zarr.ZGroup, prefix::AbstractString = "")
    ks = [ChunkManifests._joinkey(prefix, k) for k in keys(z.arrays)]
    for (name, g) in z.groups
        append!(ks, _relkeys(g, ChunkManifests._joinkey(prefix, name)))
    end
    return sort!(ks)
end

# The dataset over `group` of `z`, `""` being `z` itself. A group is opened as
# a dataset of its own rather than reached through `CDM.group`: ZarrDatasets
# 0.1.6 builds a `CDM.group` dataset with an empty dimension table, so it
# reports no dimensions and Rasters' layer selection, which removes dimension
# variables by name, fails on it.
function _groupof(z::ManifestGroup, group::AbstractString)
    zg = z
    isempty(group) || for part in split(group, '/')
        zg = zg[String(part)]
    end
    return ZarrDatasets.ZarrDataset(zg)
end

# Array keys of `z` lying directly under `group`, with their leaf names.
function _leaves(z::ManifestGroup, group::AbstractString)
    out = Pair{String, String}[]
    for key in _relkeys(z)
        g, leaf = _splitpath(key)
        g == group && push!(out, key => String(leaf))
    end
    return out
end

function _groupnames(z::ManifestGroup)
    groups = Set{String}()
    for key in _relkeys(z)
        g, _ = _splitpath(key)
        isempty(g) || push!(groups, g)
    end
    return sort!(collect(groups))
end

_describe(group::AbstractString) =
    isempty(group) ? "at the group's root" : "under group $(repr(group))"

function _nolayers(z::ManifestGroup, group::AbstractString)
    groups = _groupnames(z)
    throw(
        ArgumentError(
            "RasterStack: no array lies $(_describe(group)). The group holds " *
                "$(_relkeys(z)). " *
                (
                isempty(groups) ? "It has no groups." :
                    "Name one of its groups to reach them: $(groups)"
            ),
        )
    )
end

_epsg(s::AbstractString) = startswith(s, "EPSG:") ? Rasters.EPSG(s) : Rasters.nokw
_epsg(_) = Rasters.nokw

# Build one lazy Raster over `var`, the raw CommonDataModel variable, following
# the order `Rasters._raster` uses: read the attributes, settle the inner and
# outer missing values from them, derive the scaling/masking modification, and
# wrap the variable in it. `_maybe_modify` returns a lazy ModifiedDiskArray, so
# no chunk is read for the data; `_dims` does read the coordinate variables,
# because a Sampled or Projected lookup is those coordinate values.
#
# The one step Rasters does not take itself is the CRS: GeoTIFFDriver records
# a page's CRS as a `"crs" => "EPSG:<code>"` attribute, which Rasters' CF
# handling does not read, so that form is turned into the default `crs` here.
function _raster(
        var, name; crs, mappedcrs, missingval, scaled, coerce, raw, verbose, kw...,
    )
    scaled1, missingval1 = Rasters._raw_check(raw, scaled, missingval, verbose)
    metadata = Rasters._metadata(var)
    crs === Rasters.nokw && (crs = _epsg(get(metadata, "crs", nothing)))
    mvpair = Rasters._read_missingval_pair(var, metadata, missingval1)
    mod = Rasters._mod(eltype(var), metadata, mvpair; scaled = scaled1, coerce)
    return Rasters.Raster(
        Rasters._maybe_modify(var, mod),
        Rasters._dims(var, crs, mappedcrs);
        name = Symbol(name),
        metadata,
        missingval = Rasters._outer_missingval(mod),
        crs, mappedcrs, kw...,
    )
end

"""
    Rasters.Raster(z::Zarr.ZGroup, name; kw...)

A lazy [`Rasters.Raster`](@extref) over one array of `z`, a group returned by
[`scan`](@ref) or [`load`](@ref), named by its key within `z`
(`"gt1l/land_ice_segments/h_li"`).

`parent(raster)` wraps the manifest behind `z` itself, so the store instance,
its transports and its readahead cache survive into the raster, and a windowed read fetches only
the chunks that window covers. Constructing one reads the *coordinate*
variables, because a `Sampled` or `Projected` lookup is those coordinate
values; it reads none of the data variable.

Coordinates, CRS, calendars, CF scaling and fill-value masking all come from
Rasters' own CommonDataModel machinery over the manifest's synthesized Zarr
metadata, so the result matches what `Raster(path; lazy=true)` builds from a
real Zarr store. `crs`, `mappedcrs`, `missingval`, `scaled`, `coerce` and `raw`
mean what they mean there.

[`Rasters.RasterStack`](@extref)`(z)` opens one dataset and shares it across
every layer, so that is the cheaper route to several arrays of one group.
"""
function Rasters.Raster(
        z::ManifestGroup, name;
        crs = Rasters.nokw,
        mappedcrs = Rasters.nokw,
        missingval = Rasters.nokw,
        scaled = Rasters.nokw,
        coerce = convert,
        raw::Bool = false,
        verbose::Bool = true,
        kw...,
    )
    key = String(string(name))
    ks = _relkeys(z)
    key in ks || throw(
        ArgumentError("Raster: the group has no array at $(repr(key)); it holds $ks")
    )
    group, leaf = _splitpath(key)
    ds = _groupof(z, group)
    return _raster(
        CDM.variable(ds, leaf), leaf;
        crs, mappedcrs, missingval, scaled, coerce, raw, verbose, kw...,
    )
end

"""
    Rasters.Raster(z::Zarr.ZGroup; kw...)

A lazy [`Rasters.Raster`](@extref) over the single array of `z`.

Errors if `z` holds more than one array, listing them, rather than choosing
one: which variable of a multi-variable granule was meant is not something to
guess at.
"""
function Rasters.Raster(z::ManifestGroup; kw...)
    ks = _relkeys(z)
    length(ks) == 1 || throw(
        ArgumentError(
            "Raster: the group holds $(length(ks)) arrays, so which one to build a " *
                "Raster from has to be named: $(ks). Use RasterStack to take them all",
        )
    )
    return Rasters.Raster(z, only(ks); kw...)
end

"""
    Rasters.RasterStack(z::Zarr.ZGroup; group=nothing, name, kw...)

A lazy [`Rasters.RasterStack`](@extref) with one layer per array of `z`, a
group returned by [`scan`](@ref) or [`load`](@ref).

Only the arrays lying directly at one level become layers: those directly in
`z` by default, or those directly under its subgroup `group`. A group whose
arrays all sit in subgroups — an HDF5 granule, or a merge of several files —
reports the subgroups available rather than flattening paths into layer names.

Layers are the arrays Rasters itself makes layers of a dataset: dimension
(coordinate) variables, their bounds and `grid_mapping` variables are left out,
since they describe the layers rather than being data.

`name` overrides the layer names, which default to each array's own leaf name.
Each layer is built as in [`Rasters.Raster`](@extref)`(z, name)` and accepts
the same keywords.
"""
function Rasters.RasterStack(
        z::ManifestGroup;
        group = nothing,
        name = nothing,
        crs = Rasters.nokw,
        mappedcrs = Rasters.nokw,
        missingval = Rasters.nokw,
        scaled = Rasters.nokw,
        coerce = convert,
        raw::Bool = false,
        verbose::Bool = true,
        kw...,
    )
    g = group === nothing ? "" : String(string(group))
    leaves = _leaves(z, g)
    isempty(leaves) && _nolayers(z, g)

    ds = _groupof(z, g)
    layernames = sort!(String.(Rasters._layers(ds).names))
    isempty(layernames) && throw(
        ArgumentError(
            "RasterStack: every array $(_describe(g)) — $(last.(leaves)) — is a dimension, " *
                "bounds or grid-mapping variable, so none is a layer",
        )
    )

    names = if name === nothing
        Symbol.(layernames)
    else
        ns = Symbol[Symbol(string(n)) for n in name]
        length(ns) == length(layernames) || throw(
            ArgumentError(
                "RasterStack: name has $(length(ns)) entries but $(length(layernames)) " *
                    "layers lie $(_describe(g)): $(layernames)",
            )
        )
        ns
    end

    layers = [
        _raster(
            CDM.variable(ds, leaf), leaf;
            crs, mappedcrs, missingval, scaled, coerce, raw, verbose,
        )
            for leaf in layernames
    ]

    return Rasters.RasterStack(
        NamedTuple{Tuple(names)}(Tuple(layers));
        metadata = Rasters._metadata(ds), kw...,
    )
end

end # module ChunkManifestsRastersExt
