# Prefetching for HDF5: fetching the metadata libhdf5 is about to ask for
# before it asks.
#
# libhdf5 reads a file one structure at a time and learns where the next one
# is only from the one before, so over a network each costs a round trip. A
# version 1 B-tree node — the chunk index of a dataset in the HDF5 1.8 file
# format, which is what NetCDF4 writes — lists the addresses of all its
# children, and those nodes are written among the chunks they index, so they
# are scattered through the file. Once a node is read, all its children are
# fetched at once, and each child that is itself an internal node does the
# same when it arrives: an index costs one round trip per level instead of one
# per node.
#
# A prefetch is only a guess at what libhdf5 will read next. What it fetches
# is the file's own bytes at the address it names, so a wrong guess costs a
# request, never a wrong read.

# "TREE", the signature opening a version 1 B-tree node, read as a
# little-endian UInt32.
const _H5_BTREE1_SIGNATURE = 0x45455254

# Sizes of an address and of a length in the file, from its superblock, or
# `nothing` when the object does not begin with one. Addresses in the file are
# relative to its superblock and the driver is handed absolute ones, so only a
# file whose superblock is at offset 0 has the two agree.
function _h5sizes(source::_RangeSource)
    source.size >= 16 || return nothing
    head = Vector{UInt8}(undef, 16)
    GC.@preserve head _rangecopy!(source, pointer(head), UInt64(0), UInt64(16))
    head[1:8] == HDF5_MAGIC || return nothing
    # Superblock versions 0 and 1 give both sizes at bytes 13 and 14, later
    # versions at bytes 9 and 10.
    sa, sl = head[9] <= 1 ? (head[14], head[15]) : (head[10], head[11])
    (sa in (2, 4, 8) && sl in (2, 4, 8)) || return nothing
    return (Int(sa), Int(sl))
end

# Called for every read libhdf5 makes: a read of a B-tree node starts the
# prefetch of its children.
function _h5readhook(source::_RangeSource, buffer::Ptr{UInt8}, addr::UInt64, size::UInt64)
    size >= 8 || return nothing
    unsafe_load(Ptr{UInt32}(buffer)) == _H5_BTREE1_SIGNATURE || return nothing
    node = Vector{UInt8}(undef, size)
    GC.@preserve node unsafe_copyto!(pointer(node), buffer, size)
    for child in _h5btree1children(node, source.h5sizes..., source.size)
        # Only the reader may consult the blocks, so this check is made here
        # and not for children found by a prefetch.
        _inblocks(source, child, size) && continue
        _prefetch!(_h5onprefetch, source, child, size)
    end
    return nothing
end

# A prefetched node leads to its own children, which libhdf5 reads with the
# same size as their parent.
function _h5onprefetch(source::_RangeSource, ::UInt64, bytes::Vector{UInt8})
    length(bytes) >= 8 || return nothing
    for child in _h5btree1children(bytes, source.h5sizes..., source.size)
        _prefetch!(_h5onprefetch, source, child, UInt64(length(bytes)))
    end
    return nothing
end

# Whether every block covering [addr, addr + n) is already held.
function _inblocks(source::_RangeSource, addr::UInt64, n::UInt64)
    bs = source.blocksize
    bs == 0 && return false
    return all(b -> haskey(source.blocks, b), (addr ÷ bs):((addr + n - 1) ÷ bs))
end

_h5uint(bytes, pos, n) = foldl((v, k) -> v | (UInt64(bytes[pos + k]) << (8k)), 0:(n - 1); init = UInt64(0))

"""
    _h5btree1children(node, sa, sl, limit) -> Vector{UInt64}

Addresses of the child nodes listed in the version 1 B-tree node `node`, the
whole node as libhdf5 reads it, in a file whose addresses take `sa` bytes and
lengths `sl`. Empty for a leaf, whose children are chunks or symbol tables
rather than nodes, and for anything that does not parse as a node.

A node is a header of `8 + 2sa` bytes, then keys and child addresses
alternating, one more key than children. A group's key is one length. A chunk
index's key holds the chunk's size, its filter mask and one 8-byte offset per
dimension of the dataset plus one, and the node does not say how many
dimensions that is. Its total size does: a node holds `2K` children and
`2K + 1` keys, so only a few dimension counts divide it evenly, and the right
one is the one whose keys all end in the zero offset every chunk has along the
last, element-size dimension. Every child address must also lie below `limit`.
"""
function _h5btree1children(node::AbstractVector{UInt8}, sa::Int, sl::Int, limit::UInt64)
    Base.require_one_based_indexing(node)
    n = length(node)
    hdr = 8 + 2sa
    n > hdr || return UInt64[]
    nodetype, level = node[5], node[6]
    used = Int(node[7]) | (Int(node[8]) << 8)
    (level == 0 || used == 0) && return UInt64[]
    if nodetype == 0
        children = _h5btree1scan(node, hdr, sa, sl, used, limit, 0)
        return something(children, UInt64[])
    end
    nodetype == 1 || return UInt64[]
    for ndims in 2:33
        ks = 8 + 8ndims
        rest = n - hdr - ks
        (rest > 0 && rest % (2 * (sa + ks)) == 0) || continue
        used <= rest ÷ (sa + ks) || continue
        children = _h5btree1scan(node, hdr, sa, ks, used, limit, ndims)
        children === nothing || return children
    end
    return UInt64[]
end

# The `used` child addresses of a node whose keys are `ks` bytes, or `nothing`
# if they do not check out. `ndims` is the dimension count of a chunk index's
# keys, each of which must then end in a zero offset, and 0 for a group's.
function _h5btree1scan(node, hdr, sa, ks, used, limit, ndims)
    hdr + used * (ks + sa) + ks <= length(node) || return nothing
    children = Vector{UInt64}(undef, used)
    for i in 0:(used - 1)
        key = hdr + i * (ks + sa) + 1
        if ndims > 0
            _h5uint(node, key, 4) > 0 || return nothing
            _h5uint(node, key + 8 + 8 * (ndims - 1), 8) == 0 || return nothing
        end
        child = _h5uint(node, key + ks, sa)
        child < limit || return nothing
        children[i + 1] = child
    end
    return children
end
