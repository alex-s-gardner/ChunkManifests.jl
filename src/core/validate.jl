# Integrity validation (objectsize, validate) and manifest mutation (setchunk!).
#
# A manifest's PathTable holds one FileEntry per distinct file regardless of
# how many chunks reference it, so every check here costs one query per file,
# never one per chunk.

"""
    ObjectSizeUnsupportedError(transport)

Thrown by the generic [`objectsize`](@ref) fallback for a transport that has
not implemented it. [`validate`](@ref) catches this specifically and reports
the files reached through that transport as unverifiable rather than
treating the failure as a mismatch or crashing.
"""
struct ObjectSizeUnsupportedError <: Exception
    transport::Any
end

function Base.showerror(io::IO, e::ObjectSizeUnsupportedError)
    print(
        io,
        "objectsize is not implemented for transport ", typeof(e.transport),
        "; files reached through it cannot be checked by validate",
    )
end

function objectsize(t::AbstractTransport, uri)
    throw(ObjectSizeUnsupportedError(t))
end

"""
    objectsize(::LocalTransport, uri) -> UInt64

Size in bytes of the local file at `uri` (a plain path, or a `file://` URI),
read with `filesize` rather than opening the file.
"""
function objectsize(::LocalTransport, uri::AbstractString)
    path = startswith(uri, "file://") ? chop(uri; head=7, tail=0) : uri
    isfile(path) || throw(ArgumentError("no such file: $path"))
    return UInt64(filesize(path))
end

"""
    FileCheck(uri, reason)

One file reported by [`validate`](@ref) as unverifiable, missing or
mismatched. `reason` is empty only for entries that do not appear in
`FileCheck` form at all (a verified file is recorded as a bare uri).
"""
struct FileCheck
    uri::String
    reason::String
end

"""
    ConsistencyIssue(label, index, kind, reason)

One internal inconsistency found in a manifest without contacting any
transport: a chunk's byte range overruns its file's recorded size
(`kind = :offset_overflow`), its path table index is out of range
(`:bad_index`), or an `INLINE_CHUNK` has no inline bytes (`:empty_inline`).
`label` is the array name when the issue was found while validating a
[`VirtualGroup`](@ref), and `nothing` otherwise.
"""
struct ConsistencyIssue
    label::Union{Nothing,String}
    index::CartesianIndex
    kind::Symbol
    reason::String
end

"""
    ValidationReport(verified, unverifiable, missing_files, mismatched, consistency)

Result of [`validate`](@ref). `verified` lists the uris whose live size
matched the size recorded at scan time. `unverifiable` lists files that could
not be checked — no size was recorded, or the transport cannot report one —
distinct from `mismatched` (reachable but a different size now) and
`missing_files` (not reachable at all). `consistency` lists internal
manifest problems found without any network access.
"""
struct ValidationReport
    verified::Vector{String}
    unverifiable::Vector{FileCheck}
    missing_files::Vector{FileCheck}
    mismatched::Vector{FileCheck}
    consistency::Vector{ConsistencyIssue}
end

"""
    passed(report::ValidationReport) -> Bool

`true` only when at least one file was positively verified and nothing was
found missing, mismatched or inconsistent. A report where every file is
unverifiable (nothing could actually be checked) returns `false`: the absence
of evidence of corruption is not evidence of integrity.
"""
function passed(r::ValidationReport)
    isempty(r.missing_files) && isempty(r.mismatched) && isempty(r.consistency) || return false
    return !isempty(r.verified)
end

function Base.show(io::IO, r::ValidationReport)
    print(
        io,
        "ValidationReport(verified=", length(r.verified),
        ", unverifiable=", length(r.unverifiable),
        ", missing=", length(r.missing_files),
        ", mismatched=", length(r.mismatched),
        ", consistency=", length(r.consistency), ")",
    )
end

# Checks one file's live size against what was recorded at scan time. Returns
# a (status, reason) pair rather than throwing: a missing or mismatched file
# is an expected outcome for validate to report, not a program error.
function _probefile(entry::FileEntry, transport::AbstractTransport)
    sz = try
        objectsize(transport, entry.uri)
    catch err
        err isa ObjectSizeUnsupportedError &&
            return (:unverifiable, "cannot verify: $(sprint(showerror, err))")
        return (:missing, "could not stat \"$(entry.uri)\": $(sprint(showerror, err))")
    end

    # A field never recorded at scan time cannot be checked; that is not the
    # same outcome as checking it and finding a match.
    entry.size === nothing &&
        return (:unverifiable, "no size was recorded when this file was scanned")
    sz == entry.size && return (:verified, "")
    return (:mismatch, "recorded size $(entry.size) bytes, now $(sz) bytes")
end

# Internal consistency of a ChunkManifest's columns, needing no transport:
# every non-sentinel index must address an existing path table entry, every
# INLINE_CHUNK must have bytes, and every chunk's range must fit inside its
# file's recorded size (when a size was recorded at all).
function _consistency(m::ChunkManifest{N}, label=nothing) where {N}
    issues = ConsistencyIssue[]
    t = m.table
    ntable = length(t.entries)
    for I in CartesianIndices(chunkgridaxes(m))
        idx = m.index[I]
        if idx == MISSING_INDEX
            continue
        elseif idx == INLINE_INDEX
            bytes = get(m.inline, I, nothing)
            if bytes === nothing || isempty(bytes)
                push!(
                    issues,
                    ConsistencyIssue(
                        label, I, :empty_inline,
                        "chunk $(Tuple(I)) is INLINE_CHUNK but has no inline bytes",
                    ),
                )
            end
        elseif idx > ntable
            push!(
                issues,
                ConsistencyIssue(
                    label, I, :bad_index,
                    "chunk $(Tuple(I)) references path table index $idx, out of range 1:$ntable",
                ),
            )
        else
            entry = t.entries[idx]
            if entry.size !== nothing
                stop = UInt64(m.offset[I]) + UInt64(m.nbytes[I])
                if stop > entry.size
                    push!(
                        issues,
                        ConsistencyIssue(
                            label, I, :offset_overflow,
                            "chunk $(Tuple(I)) range [$(m.offset[I]), $stop) exceeds recorded " *
                            "size $(entry.size) of \"$(entry.uri)\"",
                        ),
                    )
                end
            end
        end
    end
    return issues
end

# AffineManifest's offsets are a closed-form function of the chunk index and
# its grid bounds are checked by chunkstate/chunklocation on every access, so
# there is no per-chunk state that can go inconsistent.
_consistency(::AffineManifest, label=nothing) = ConsistencyIssue[]

# Shared engine behind every validate(...) method: one FileEntry probe per
# path table entry (never per chunk) plus the chunk-level consistency check
# above. `label` tags issues and file reasons with an array name when called
# from validate(::VirtualGroup); nothing otherwise.
function _validate_core(m::AbstractManifest, transport::AbstractTransport, strict::Bool, label)
    consistency = _consistency(m, label)
    if strict && !isempty(consistency)
        c = first(consistency)
        prefix = label === nothing ? "" : "array \"$label\": "
        throw(ArgumentError(
            "validate: $(prefix)manifest is internally inconsistent at chunk " *
            "$(Tuple(c.index)): $(c.reason)",
        ))
    end

    verified = String[]
    unverifiable = FileCheck[]
    missing_files = FileCheck[]
    mismatched = FileCheck[]

    t = pathtable(m)
    for i in eachindex(t.entries)
        entry = t.entries[i]
        status, reason = _probefile(entry, transport)
        taggedreason = label === nothing ? reason : "array \"$label\": " * reason
        if status === :verified
            push!(verified, entry.uri)
        elseif status === :unverifiable
            push!(unverifiable, FileCheck(entry.uri, taggedreason))
        elseif status === :missing
            strict && throw(ArgumentError("validate: \"$(entry.uri)\" is missing: $taggedreason"))
            push!(missing_files, FileCheck(entry.uri, taggedreason))
        else
            strict && throw(ArgumentError(
                "validate: \"$(entry.uri)\" size mismatch: $taggedreason"
            ))
            push!(mismatched, FileCheck(entry.uri, taggedreason))
        end
    end

    return verified, unverifiable, missing_files, mismatched, consistency
end

"""
    validate(m::AbstractManifest, transport=LocalTransport(); strict=false) -> ValidationReport

Check `m`'s distinct files against live storage through `transport`, and
check `m`'s internal consistency. Cost is one [`objectsize`](@ref) query per
distinct file in `m`'s [`PathTable`](@ref) — independent of how many chunks
reference each file — plus one pass over the chunk grid for the
no-network consistency check.

With `strict=true`, throws on the first inconsistency or file problem found
instead of collecting a full report.
"""
function validate(m::AbstractManifest, transport::AbstractTransport=LocalTransport(); strict::Bool=false)
    verified, unverifiable, missing_files, mismatched, consistency =
        _validate_core(m, transport, strict, nothing)
    return ValidationReport(verified, unverifiable, missing_files, mismatched, consistency)
end

"""
    validate(a::VirtualArray, transport=LocalTransport(); strict=false) -> ValidationReport

Equivalent to `validate(manifestof(a), transport; strict)`.
"""
function validate(a::VirtualArray, transport::AbstractTransport=LocalTransport(); strict::Bool=false)
    return validate(manifestof(a), transport; strict)
end

"""
    validate(g::VirtualGroup, transport=LocalTransport(); strict=false) -> ValidationReport

Validates every array's manifest in `g` and merges the results. Each array is
checked independently, so a file shared by two arrays' manifests is queried
once per array rather than once overall; [`ConsistencyIssue`](@ref) and file
reasons carry the owning array's name.
"""
function validate(g::VirtualGroup, transport::AbstractTransport=LocalTransport(); strict::Bool=false)
    verified = String[]
    unverifiable = FileCheck[]
    missing_files = FileCheck[]
    mismatched = FileCheck[]
    consistency = ConsistencyIssue[]

    for (name, arr) in arraysof(g)
        v, u, mi, mm, c = _validate_core(manifestof(arr), transport, strict, name)
        append!(verified, v)
        append!(unverifiable, u)
        append!(missing_files, mi)
        append!(mismatched, mm)
        append!(consistency, c)
    end

    return ValidationReport(verified, unverifiable, missing_files, mismatched, consistency)
end

# Converts a MethodError from an attempted column write into a message naming
# the column, rather than letting setchunk! fail with an opaque MethodError
# from inside an AbstractArray that simply never defined setindex!.
function _setindex_checked!(col, v, I, label::AbstractString)
    try
        col[I] = v
    catch err
        err isa MethodError || err isa Base.CanonicalIndexError || rethrow()
        throw(ArgumentError(
            "setchunk!: the $label column ($(typeof(col))) does not support in-place " *
            "mutation; it is read-only or backed by a lazy/immutable array",
        ))
    end
    return nothing
end

"""
    setchunk!(m::ChunkManifest, I::CartesianIndex, uri, offset, nbytes;
              etag=nothing, size=nothing, mtime=nothing) -> m

Repoint chunk `I` to byte range `[offset, offset + nbytes)` of `uri`,
reusing [`push_uri!`](@ref) to add `uri` to `m`'s path table only if it is
not already present. No other chunk's columns are read or written.
"""
function setchunk!(
    m::ChunkManifest{N},
    I::CartesianIndex{N},
    uri::AbstractString,
    offset::Integer,
    nbytes::Integer;
    etag=nothing,
    size=nothing,
    mtime=nothing,
) where {N}
    idx = push_uri!(m.table, uri; etag, size, mtime)
    _setindex_checked!(m.index, idx, I, "index")
    _setindex_checked!(m.offset, offset, I, "offset")
    _setindex_checked!(m.nbytes, nbytes, I, "nbytes")
    delete!(m.inline, I)  # this cell is no longer INLINE_CHUNK, if it ever was
    return m
end

"""
    setchunk!(m::ChunkManifest, I::CartesianIndex, state::ChunkState) -> m

Set chunk `I` to [`MISSING_CHUNK`](@ref): its bytes are absent and read as
the array's fill value. `state` must be `MISSING_CHUNK`; `VIRTUAL_CHUNK`
needs the `(uri, offset, nbytes)` form of `setchunk!` and `INLINE_CHUNK`
needs the byte-vector form.
"""
function setchunk!(m::ChunkManifest{N}, I::CartesianIndex{N}, state::ChunkState) where {N}
    state == MISSING_CHUNK || throw(ArgumentError(
        "setchunk!: a bare ChunkState argument must be MISSING_CHUNK (got $state); " *
        "VIRTUAL_CHUNK needs (uri, offset, nbytes) and INLINE_CHUNK needs a byte vector",
    ))
    _setindex_checked!(m.index, MISSING_INDEX, I, "index")
    delete!(m.inline, I)
    return m
end

"""
    setchunk!(m::ChunkManifest, I::CartesianIndex, bytes::AbstractVector{UInt8}) -> m

Set chunk `I` to [`INLINE_CHUNK`](@ref), embedding `bytes` directly in the
manifest rather than referencing an external file.
"""
function setchunk!(m::ChunkManifest{N}, I::CartesianIndex{N}, bytes::AbstractVector{UInt8}) where {N}
    isempty(bytes) && throw(ArgumentError("setchunk!: inline bytes must be non-empty"))
    _setindex_checked!(m.index, INLINE_INDEX, I, "index")
    m.inline[I] = Vector{UInt8}(bytes)
    return m
end

"""
    setchunk!(m::AffineManifest, I, args...; kwargs...)

Throws unconditionally. An [`AffineManifest`](@ref) has no per-chunk storage
to repoint — its offsets are a closed-form function of the chunk index — so
there is no supported conversion to a [`ChunkManifest`](@ref); build one
directly instead.
"""
function setchunk!(::AffineManifest, I, args...; kwargs...)
    throw(ArgumentError(
        "setchunk!: AffineManifest has no per-chunk storage to repoint and cannot be " *
        "converted to a ChunkManifest; build a ChunkManifest directly instead",
    ))
end
