# Kerchunk's JSON reference-set format, reimplemented from the published
# schema. Nothing here calls Python.
#
# `templates` substitution is implemented because a reader must accept it,
# but this writer never emits a `templates` section, matching every real
# kerchunk driver. `gen` (programmatic reference generation) is rejected
# outright: expanding a `gen` section into explicit references needs a
# Python-expression evaluator, which this package does not implement.
#
# A whole-object reference (`[url]`, no byte length) is recorded with
# `nbytes` set to `_WHOLE_OBJECT_NBYTES`, a sentinel no real chunk reaches.
# Nothing in this package's transport interface can resolve "the rest of
# the file" into a concrete length without reading the object first, so the
# sentinel is left in place rather than guessed at. Consequence: fetching
# such a chunk's bytes through a transport requests a byte range that
# exceeds the object's actual size and fails loudly there (`LocalTransport`
# raises `ArgumentError` for a range past end-of-file; a range request past
# the end of a remote object fails the same way) instead of silently
# returning fewer bytes than the chunk key promised. The value is
# `typemax(Int64)`, not `typemax(UInt64)`: the readahead
# planner and `Zarr.storagesize` both narrow a chunk's `nbytes` to `Int`
# while scanning neighboring chunks, including ones nobody asked to read
# yet, so a value `Int` cannot hold would throw there instead of at the
# fetch it is meant to fail.
const _WHOLE_OBJECT_NBYTES = UInt64(typemax(Int64))

function _substitutetemplates(url::AbstractString, templates::AbstractDict)
    for (k, v) in templates
        url = replace(url, "{{$k}}" => string(v))
    end
    return url
end

# `.zarray`/`.zattrs`/`.zgroup` entries are themselves JSON-encoded strings,
# not nested objects, per the kerchunk schema.
function _parsejsonstring(value, path, key)
    value isa AbstractString || throw(
        ArgumentError(
            "$path: \"$key\" must hold a JSON-encoded string, got $(typeof(value))"
        )
    )
    return JSON.parse(value)
end

# Byte-order markers numpy puts at the head of a dtype string. `'|'` spells
# "does not apply", `'='` the host's order.
const _DTYPE_BYTEORDERS = ('<', '>', '|', '=')

# Splits "<i4" into ('<', "i4"). A dtype carrying no marker reads as `'|'`.
function _splitdtype(dtype::AbstractString)
    isempty(dtype) && return ('|', dtype)
    order = first(dtype)
    return order in _DTYPE_BYTEORDERS ? (order, dtype[nextind(dtype, 1):end]) : ('|', dtype)
end

# A dtype can be read exactly when `zarr_dtype_string` emits it again, so the
# writer defines the readable set and the two cannot disagree.
#
# The byte-order marker is compared separately, because `Zarr.typestr` parses it
# and then discards it: ">i4" and "<i4" both give `Int32`. A big-endian dtype
# would therefore decode as little-endian and yield wrong values rather than
# fail, which is what `_checkbyteorder` refuses for HDF5 datasets and what this
# refuses here. Single-byte elements and strings have no byte order to get
# wrong, and numpy writes both "|u1" and "<u1", so the marker is not compared
# for those.
function _juliadtype(dtype, path, key)
    dtype isa AbstractString || throw(
        ArgumentError(
            "$path: $key: \"dtype\" must be a string, got $(typeof(dtype))"
        )
    )
    T = try
        Zarr.typestr(dtype)
    catch e
        e isa ArgumentError || rethrow()
        throw(ArgumentError("$path: $key: invalid Zarr v2 dtype string $(repr(dtype)): $(e.msg)"))
    end
    canonical = try
        zarr_dtype_string(T)
    catch e
        e isa ArgumentError || rethrow()
        throw(
            ArgumentError(
                "$path: $key: dtype $(repr(dtype)) parses to Julia type $T, which has no " *
                    "faithful round trip through this package's Zarr v2 dtype encoding",
            )
        )
    end
    order, kind = _splitdtype(dtype)
    _, canonicalkind = _splitdtype(canonical)
    kind == canonicalkind || throw(
        ArgumentError(
            "$path: $key: dtype $(repr(dtype)) parses to Julia type $T, which this " *
                "package encodes as $(repr(canonical)); reading under one spelling and " *
                "writing back under the other would change the declared type",
        )
    )
    byteordermatters = sizeof(T) > 1 && first(canonicalkind) != 'S'
    (order == '>' && byteordermatters) && throw(
        ArgumentError(
            "$path: $key: dtype $(repr(dtype)) is big-endian, which cannot be served " *
                "faithfully. This store passes a source's bytes through untouched and " *
                "Zarr.jl ignores the byte-order marker when decoding, so the values " *
                "would be wrong rather than refused. Use $(repr(canonical)) for " *
                "little-endian data.",
        )
    )
    return T
end

# parse_chunkkey only consults size/chunkshapeof, so a throwaway
# ManifestArray over an AffineChunkMap sized to match the real chunk grid is
# enough to reuse it before the real manifest exists; it is discarded once
# the chunk loop below finishes.
function _chunkkeyparser(shape::NTuple{N, Int}, chunkshape::NTuple{N, Int}) where {N}
    gridsize = cld.(shape, chunkshape)
    table = PathTable()
    push_uri!(table, "")
    manifest = AffineChunkMap(table, gridsize, UInt64(0), ntuple(_ -> UInt64(0), N), UInt32(0))
    return ManifestArray{UInt8}(manifest, shape, chunkshape)
end

# One ref value, decoded into one of the schema's four shapes and returned
# as a uniform 5-tuple `(kind, url, offset, nbytes, bytes)` — fields unused
# by `kind` are left at their zero value. Throws a bare ArgumentError on a
# shape mismatch; the caller attaches the offending file and key.
function _resolveref(raw, templates::AbstractDict)
    noturl = ""
    nobytes = UInt8[]
    if raw isa AbstractVector
        n = length(raw)
        if n == 3
            url, offset, nbytes = raw
            url isa AbstractString || throw(
                ArgumentError(
                    "a 3-element reference's first element must be a URL string, got $(typeof(url))"
                )
            )
            (offset isa Real && nbytes isa Real) || throw(
                ArgumentError(
                    "a 3-element reference's offset and length must be numbers, got " *
                        "$(typeof(offset)) and $(typeof(nbytes))",
                )
            )
            return (:range, _substitutetemplates(url, templates), UInt64(offset), UInt64(nbytes), nobytes)
        elseif n == 1
            url = raw[1]
            url isa AbstractString || throw(
                ArgumentError(
                    "a 1-element (whole-object) reference's element must be a URL string, got $(typeof(url))"
                )
            )
            return (:whole, _substitutetemplates(url, templates), UInt64(0), UInt64(0), nobytes)
        else
            throw(
                ArgumentError(
                    "an array-form reference must have 1 or 3 elements, got $n"
                )
            )
        end
    elseif raw isa AbstractString
        bytes = startswith(raw, "base64:") ?
            Base64.base64decode(chopprefix(raw, "base64:")) :
            Vector{UInt8}(codeunits(raw))
        return (:inline, noturl, UInt64(0), UInt64(0), bytes)
    else
        throw(
            ArgumentError(
                "a reference value must be a [url], [url, offset, length] array, or a " *
                    "string; got $(typeof(raw))",
            )
        )
    end
end

function _attrsanddimnames(zattrsdoc, N::Integer)
    zattrsdoc === nothing && return Dict{String, Any}(), ["dim_$i" for i in 1:N]
    attrs = Dict{String, Any}(zattrsdoc)
    dimnames = if haskey(attrs, "_ARRAY_DIMENSIONS")
        # _ARRAY_DIMENSIONS is C-order like shape/chunks; see zattrs_json.
        dims = reverse(collect(String, attrs["_ARRAY_DIMENSIONS"]))
        delete!(attrs, "_ARRAY_DIMENSIONS")
        dims
    else
        ["dim_$i" for i in 1:N]
    end
    return attrs, dimnames
end

function _buildarray(path, arraypath, zarraydoc, zattrsdoc, chunkleaves, table, templates)
    N = length(zarraydoc["shape"])
    length(zarraydoc["chunks"]) == N || throw(
        ArgumentError(
            "$path: array \"$arraypath\": \"shape\" has $N dimensions but \"chunks\" has " *
                "$(length(zarraydoc["chunks"]))",
        )
    )
    # Zarr v2 is C-ordered; ManifestArray is Julia (column-major) order. This
    # is the one point where shape/chunks are reversed back, the inverse of
    # zarray_json's own reversal on write.
    shape = NTuple{N, Int}(reverse(Int.(zarraydoc["shape"])))
    chunkshape = NTuple{N, Int}(reverse(Int.(zarraydoc["chunks"])))
    T = _juliadtype(zarraydoc["dtype"], path, "$arraypath/.zarray")

    gridsize = cld.(shape, chunkshape)
    index = fill(MISSING_INDEX, gridsize)
    offset = zeros(UInt64, gridsize)
    nbytes = zeros(UInt64, gridsize)
    inline = Dict{CartesianIndex{N}, Vector{UInt8}}()

    keyparser = _chunkkeyparser(shape, chunkshape)
    for (leaf, raw) in chunkleaves
        fullkey = "$arraypath/$leaf"
        I = parse_chunkkey(keyparser, leaf)
        I === nothing && throw(
            ArgumentError(
                "$path: chunk key \"$fullkey\" does not parse for array \"$arraypath\" " *
                    "with chunk grid $gridsize",
            )
        )
        kind, url, off, nb, bytes = try
            _resolveref(raw, templates)
        catch e
            e isa ArgumentError || rethrow()
            throw(ArgumentError("$path: refs[\"$fullkey\"]: $(e.msg)"))
        end
        if kind == :range
            index[I] = push_uri!(table, url)
            offset[I] = off
            nbytes[I] = nb
        elseif kind == :whole
            index[I] = push_uri!(table, url)
            offset[I] = 0
            nbytes[I] = _WHOLE_OBJECT_NBYTES
        else
            index[I] = INLINE_INDEX
            inline[I] = bytes
        end
    end

    manifest = ExplicitChunkMap(table, index, offset, nbytes; inline)
    attrs, dimnames = _attrsanddimnames(zattrsdoc, N)
    fillvalue = _fillvaluefromjson(get(zarraydoc, "fill_value", nothing), T)
    compressor = get(zarraydoc, "compressor", nothing)
    filters = get(zarraydoc, "filters", nothing)
    filters = filters === nothing ? Dict{String, Any}[] : Vector{Dict{String, Any}}(filters)

    return ManifestArray{T}(manifest, shape, chunkshape; fillvalue, compressor, filters, attrs, dimnames)
end

# Every "/"-separated proper prefix of an array path that is not itself an
# array path: the groups that exist only because an array lives under them.
function _implicitgroups(arraypaths)
    prefixes = Set{String}()
    for path in arraypaths
        parts = split(path, '/')
        for i in 1:(length(parts) - 1)
            push!(prefixes, join(parts[1:i], '/'))
        end
    end
    return prefixes
end

function _writechunks!(refs, arraypath, va::ManifestArray, fmt::KerchunkJSON, transport::AbstractTransport)
    m = chunkmapof(va)
    for I in CartesianIndices(chunkgridaxes(m))
        state = chunkstate(m, I)
        # Absent from refs is how kerchunk expresses "no chunk, use fill_value".
        state == MISSING_CHUNK && continue
        key = "$arraypath/" * chunkkey(va, I)
        if state == INLINE_CHUNK
            refs[key] = "base64:" * Base64.base64encode(inlinebytes(m, I))
            continue
        end
        uri, offset, nbytes = chunklocation(m, I)
        if fmt.inlinethreshold > 0 && nbytes < fmt.inlinethreshold
            bytes = fetchrange(transport, uri, ByteRange(offset, nbytes))
            refs[key] = "base64:" * Base64.base64encode(bytes)
        else
            refs[key] = Any[uri, offset, nbytes]
        end
    end
    return nothing
end

"""
    save(path, group::ChunkManifest, fmt::KerchunkJSON; transport=LocalTransport())

Write `group` as a kerchunk JSON reference-set document to `path`. `path`
names one document, not a directory: it is resolved to a store through
`Zarr.storefromstring`, the same mechanism the [`ZarrManifest`](@ref) method
uses, so `path` may equally be a local file path or an `s3://`, `gs://`,
`http://`, or `https://` URI.

Every array's `.zarray`/`.zattrs` and every group's `.zgroup`/`.zattrs` are
written as JSON-encoded string values under `refs`, matching the schema.
Missing chunks are left absent from `refs` rather than written with a null
or placeholder value. Chunks smaller than `fmt.inlinethreshold` bytes are
read through `transport` and embedded as `"base64:..."` rather than kept as
byte-range references; `inlinethreshold = 0` embeds nothing. No `templates`
section is written, matching real kerchunk drivers.
"""
function save(
        path::AbstractString, group::ChunkManifest, fmt::KerchunkJSON;
        transport::AbstractTransport = LocalTransport(),
    )
    store, key = _resolvefilestore(path, true)
    save(store, key, group, fmt; transport)
    return nothing
end

"""
    save(store::Zarr.AbstractStore, key::AbstractString, group::ChunkManifest, fmt::KerchunkJSON; transport=LocalTransport())

Write `group` as a kerchunk JSON reference-set document into `store` under
`key`, exactly as `save(path, group, fmt)` does once it has resolved `path`
to a store. Not part of the public interface; exists so a manifest's
store-agnosticism can be exercised directly against any `Zarr.AbstractStore`.
"""
function save(
        store::Zarr.AbstractStore, key::AbstractString, group::ChunkManifest, fmt::KerchunkJSON;
        transport::AbstractTransport = LocalTransport(),
    )
    arrays = arraysof(group)
    refs = Dict{String, Any}()

    refs[".zgroup"] = String(zgroup_json())
    refs[".zattrs"] = String(Vector{UInt8}(JSON.json(_jsonsafeattrs(attrsof(group)))))
    for g in _implicitgroups(keys(arrays))
        refs["$g/.zgroup"] = String(zgroup_json())
        refs["$g/.zattrs"] = String(Vector{UInt8}(JSON.json(Dict{String, Any}())))
    end

    for (arraypath, va) in arrays
        refs["$arraypath/.zarray"] = String(zarray_json(va))
        refs["$arraypath/.zattrs"] = String(zattrs_json(va))
        _writechunks!(refs, arraypath, va, fmt, transport)
    end

    doc = Dict{String, Any}("version" => KERCHUNK_REFERENCE_VERSION, "refs" => refs)
    store[key] = Vector{UInt8}(codeunits(JSON.json(doc)))
    return nothing
end

"""
    ChunkManifest(path, fmt::KerchunkJSON) -> ChunkManifest

Read a kerchunk JSON reference-set document from `path` into a
[`ChunkManifest`](@ref). `path` names one document, not a directory: it is
resolved to a store through `Zarr.storefromstring`, so `path` may equally be
a local file path or an `s3://`, `gs://`, `http://`, or `https://` URI.

Accepts all four `refs` entry shapes (`[url, offset, length]`, `[url]`,
a plain string, and a `"base64:..."` string) and substitutes `templates`
into the URL of the array-form entries. Rejects a document missing
`version`, a `version` other than 1, and a `gen` section. A chunk key
present in `refs` that does not parse against its array's chunk grid, and a
`.zarray` dtype with no faithful Julia type, both throw naming the file and
the offending key.
"""
function ChunkManifest(path::AbstractString, fmt::KerchunkJSON)
    store, key = _resolvefilestore(path, false)
    return _load_kerchunkjson(store, key, path, fmt)
end

"""
    ChunkManifest(store::Zarr.AbstractStore, key::AbstractString, fmt::KerchunkJSON) -> ChunkManifest

Read a kerchunk JSON reference-set document from `store` under `key`,
exactly as `ChunkManifest(path, fmt)` does once it has resolved `path` to a store.
Not part of the public interface; exists so a manifest's store-agnosticism
can be exercised directly against any `Zarr.AbstractStore`.
"""
function ChunkManifest(store::Zarr.AbstractStore, key::AbstractString, fmt::KerchunkJSON)
    return _load_kerchunkjson(store, key, key, fmt)
end

function _load_kerchunkjson(store::Zarr.AbstractStore, key::AbstractString, label::AbstractString, ::KerchunkJSON)
    bytes = store[key]
    bytes === nothing && throw(ArgumentError("load: \"$label\" does not exist"))
    doc = JSON.parse(String(bytes))
    doc isa AbstractDict || throw(
        ArgumentError(
            "$label: top-level kerchunk document must be a JSON object, got $(typeof(doc))"
        )
    )
    haskey(doc, "version") || throw(ArgumentError("$label: missing required \"version\" key"))
    doc["version"] == KERCHUNK_REFERENCE_VERSION || throw(
        ArgumentError(
            "$label: unsupported kerchunk reference-set version $(repr(doc["version"])); " *
                "only version $KERCHUNK_REFERENCE_VERSION is supported",
        )
    )
    haskey(doc, "gen") && throw(
        ArgumentError(
            "$label: \"gen\" (programmatic reference generation) is not supported; " *
                "expand it to explicit refs before loading",
        )
    )
    haskey(doc, "refs") || throw(ArgumentError("$label: missing required \"refs\" key"))
    refs = doc["refs"]
    templates = get(doc, "templates", Dict{String, Any}())

    rootattrs = Dict{String, Any}()
    zarraydocs = Dict{String, Any}()
    zattrsdocs = Dict{String, Any}()
    chunkleaves = Dict{String, Vector{Pair{String, Any}}}()

    for (refkey, value) in refs
        prefix, leaf = _splitkey(refkey)
        if leaf == ".zarray"
            zarraydocs[prefix] = _parsejsonstring(value, label, refkey)
        elseif leaf == ".zattrs"
            if prefix == ""
                rootattrs = Dict{String, Any}(_parsejsonstring(value, label, refkey))
            else
                zattrsdocs[prefix] = _parsejsonstring(value, label, refkey)
            end
        elseif leaf == ".zgroup"
            continue
        else
            push!(get!(() -> Pair{String, Any}[], chunkleaves, prefix), leaf => value)
        end
    end

    table = PathTable()
    arrays = Dict{String, ManifestArray}()
    for (arraypath, zarraydoc) in zarraydocs
        leaves = get(chunkleaves, arraypath, Pair{String, Any}[])
        zattrsdoc = get(zattrsdocs, arraypath, nothing)
        arrays[arraypath] = _buildarray(label, arraypath, zarraydoc, zattrsdoc, leaves, table, templates)
    end

    provenance = Dict{String, Any}("format" => "KerchunkJSON", "path" => String(label))
    return ChunkManifest(; arrays, attrs = rootattrs, provenance)
end
