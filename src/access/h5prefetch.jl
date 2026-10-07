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
# The same holds across the datasets of a group: libhdf5 opens them one at a
# time, and each one's header leads to its chunk index only once read. The
# scan hands over the header addresses of a group's members before walking
# it (`_h5prefetchobjects`); each header is fetched at once and parsed here
# for the continuation of the header and the root of the chunk index, which
# are fetched in turn. A remote scan does this for the group or dataset it
# starts from in a first pass, and waits for what it leads to without holding
# `HDF5_IO` (see `_scan_hdf5`), so scans of several files overlap there.
#
# A prefetch is only a guess at what libhdf5 will read next. What it fetches
# is the file's own bytes at the address it names, so a wrong guess costs a
# request, never a wrong read.

# "TREE", the signature opening a version 1 B-tree node, read as a
# little-endian UInt32.
const _H5_BTREE1_SIGNATURE = 0x45455254

# What prefetching needs from a file's superblock: the sizes of an address
# and of a length, and the `K` of its chunk-index B-trees, which sets the size
# of every node. `nothing` when the object does not begin with a superblock:
# addresses in the file are relative to it and the driver is handed absolute
# ones, so only a file whose superblock is at offset 0 has the two agree.
function _h5sizes(source::_RangeSource)
    source.size >= 28 || return nothing
    head = Vector{UInt8}(undef, 28)
    GC.@preserve head _rangecopy!(source, pointer(head), UInt64(0), UInt64(28))
    head[1:8] == HDF5_MAGIC || return nothing
    version = head[9]
    # Superblock versions 0 and 1 give both sizes at bytes 13 and 14, later
    # versions at bytes 9 and 10. Only version 1 records the chunk-index K;
    # every other version leaves it at libhdf5's default of 32, short of a
    # superblock extension saying otherwise.
    sa, sl = version <= 1 ? (head[14], head[15]) : (head[10], head[11])
    (sa in (2, 4, 8) && sl in (2, 4, 8)) || return nothing
    istorek = version == 1 ? Int(_h5uint(head, 25, 2)) : 32
    return (Int(sa), Int(sl), istorek)
end

# Called for every read libhdf5 makes: a read of a B-tree node starts the
# prefetch of its children.
function _h5readhook(source::_RangeSource, buffer::Ptr{UInt8}, addr::UInt64, size::UInt64)
    size >= 8 || return nothing
    unsafe_load(Ptr{UInt32}(buffer)) == _H5_BTREE1_SIGNATURE || return nothing
    node = Vector{UInt8}(undef, size)
    GC.@preserve node unsafe_copyto!(pointer(node), buffer, size)
    sa, sl, _ = source.h5sizes
    for child in _h5btree1children(node, sa, sl, source.size)
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
    sa, sl, _ = source.h5sizes
    for child in _h5btree1children(bytes, sa, sl, source.size)
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

# Bytes fetched at an object header's address: the first chunk of a dataset's
# header, which names its chunk index and any continuation, is rarely longer.
const _H5_HEADER_PREFETCH = 4096

# Header message types this follows: a continuation, which gives where the
# rest of the header is, and a data layout, which gives the chunk index.
const _H5_MSG_CONTINUATION = 0x0010
const _H5_MSG_LAYOUT = 0x0008

"""
    _h5prefetchobjects(fileid, addrs)

Prefetch the object headers at `addrs` in the file open as `fileid`, and the
chunk index each leads to. Does nothing for a file not open through the range
driver.
"""
function _h5prefetchobjects(fileid, addrs)
    source = @lock _RANGE_LOCK get(_RANGE_FILES, fileid, nothing)
    (source === nothing || source.h5sizes === nothing) && return nothing
    for addr in addrs
        _prefetch!(_h5onheader, source, UInt64(addr), UInt64(_H5_HEADER_PREFETCH))
    end
    return nothing
end

function _h5onheader(source::_RangeSource, addr::UInt64, bytes::Vector{UInt8})
    parsed = _h5headerprefix(bytes)
    parsed === nothing && return nothing
    version, start, stop, corder = parsed
    _h5followmessages(source, bytes, start, min(stop, length(bytes) + 1), version, corder)
    return nothing
end

# Where the messages of an object header's first chunk lie in `bytes`, the
# header as fetched: `(version, start, stop, corder)` with messages in
# `start:stop-1`, and whether each message records a creation order. A
# version 1 header has no signature and begins with its version number.
function _h5headerprefix(bytes::Vector{UInt8})
    length(bytes) >= 16 || return nothing
    if bytes[1:4] == b"OHDR"
        bytes[5] == 2 || return nothing
        flags = bytes[6]
        pos = 7
        flags & 0x20 != 0 && (pos += 16)  # access, modification, change, birth times
        flags & 0x10 != 0 && (pos += 4)   # attribute storage phase change values
        width = 1 << (flags & 0x03)
        pos + width - 1 <= length(bytes) || return nothing
        chunk0 = Int(_h5uint(bytes, pos, width))
        pos += width
        return 2, pos, pos + chunk0, flags & 0x04 != 0
    elseif bytes[1] == 1 && bytes[2] == 0
        # Version, a reserved byte, the message count, the reference count and
        # the size of the first chunk, padded to 16 bytes.
        return 1, 17, 17 + Int(_h5uint(bytes, 9, 4)), false
    end
    return nothing
end

# Walks the messages in `bytes[start:stop-1]`, prefetching what a continuation
# or a chunked layout points at.
function _h5followmessages(source, bytes, start, stop, version, corder)
    sa, sl, istorek = source.h5sizes
    pos = start
    while true
        if version == 2
            pos + 3 < stop || break
            mtype = UInt16(bytes[pos])
            msize = Int(_h5uint(bytes, pos + 1, 2))
            data = pos + 4 + (corder ? 2 : 0)
        else
            pos + 7 < stop || break
            mtype = UInt16(_h5uint(bytes, pos, 2))
            msize = Int(_h5uint(bytes, pos + 2, 2))
            data = pos + 8
        end
        data + msize <= stop || break
        if mtype == _H5_MSG_CONTINUATION && msize >= sa + sl
            at = _h5uint(bytes, data, sa)
            len = _h5uint(bytes, data + sa, sl)
            _prefetch!(source, at, len) do src, _, more
                # A version 2 continuation chunk opens with "OCHK" and ends
                # with a checksum; a version 1 one is messages alone.
                version == 2 && (length(more) < 8 || more[1:4] != b"OCHK") && return nothing
                lo, hi = version == 2 ? (5, length(more) - 3) : (1, length(more) + 1)
                _h5followmessages(src, more, lo, hi, version, corder)
            end
        elseif mtype == _H5_MSG_LAYOUT && msize >= 3 + sa
            _h5prefetchlayout(source, bytes, data, sa, istorek)
        end
        pos = version == 2 ? data + msize : data + ((msize + 7) & ~7)
    end
    return nothing
end

# A version 3 chunked layout names the root of a version 1 B-tree chunk index,
# whose node size follows from the dataset's dimension count and the file's K.
# Other layouts and index types are left for libhdf5 to read on its own.
function _h5prefetchlayout(source, bytes, data, sa, istorek)
    (bytes[data] == 3 && bytes[data + 1] == 2) || return nothing
    ndims = Int(bytes[data + 2])
    root = _h5uint(bytes, data + 3, sa)
    keysize = 8 + 8ndims
    nodesize = 8 + 2sa + 2istorek * sa + (2istorek + 1) * keysize
    _prefetch!(_h5onprefetch, source, root, UInt64(nodesize))
    return nothing
end

