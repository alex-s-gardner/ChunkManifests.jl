# The zip reader (`zip_members`, `ZipEntry`, `read_entry`, `read_member`, `extract_member`, `inflate_ranges!`):
# byte ranges of a stored and a deflated member read back as written, from a local file and over HTTP. Uses
# `_withserver` of `http.jl`.

import Random
import Zarr
using ChunkManifests: ZipMember, zip_members, ZipEntry, read_entry, read_member, extract_member, inflate_ranges!

# A zip of `members` (name => bytes, deflated when the flag is set), written field by field: local headers,
# then the central directory and its end record.
function _zip_fixture(members::Vector{Tuple{String,Vector{UInt8},Bool}})
    io = IOBuffer()
    central = IOBuffer()
    le16(x) = htol(UInt16(x))
    le32(x) = htol(UInt32(x))
    for (name, bytes, deflate) in members
        # Zarr's zlib stream less its two-byte header and four-byte checksum is the raw deflate a zip holds.
        data = deflate ? ZarrCore.zcompress(bytes, Zarr.ZlibCompressor())[3:(end - 4)] : bytes
        offset = position(io)
        method = deflate ? 8 : 0
        write(io, le32(0x04034b50), le16(20), le16(0), le16(method), le16(0), le16(0), le32(0),
              le32(length(data)), le32(length(bytes)), le16(ncodeunits(name)), le16(0), codeunits(name), data)
        write(central, le32(0x02014b50), le16(20), le16(20), le16(0), le16(method), le16(0), le16(0), le32(0),
              le32(length(data)), le32(length(bytes)), le16(ncodeunits(name)), le16(0), le16(0), le16(0),
              le16(0), le32(0), le32(offset), codeunits(name))
    end
    cd = take!(central)
    cd_offset = position(io)
    write(io, cd, le32(0x06054b50), le16(0), le16(0), le16(length(members)), le16(length(members)),
          le32(length(cd)), le32(cd_offset), le16(0))
    return take!(io)
end

@testset "zip members" begin
    # Mostly literals — sixteen symbols — so deflate closes a block every few tens of kilobytes and the
    # index has many access points to resume from.
    big = rand(Random.Xoshiro(7), 0x00:0x0f, 3_000_000)
    small = Vector{UInt8}("<annotation>small</annotation>")
    bytes = _zip_fixture([("SAFE/annotation/a.xml", small, true), ("SAFE/measurement/m.tiff", big, true),
                          ("SAFE/stored.bin", big[1:1000], false)])
    check(uri, transport) = begin
        ms = zip_members(uri; transport)
        @test [m.name for m in ms] == ["SAFE/annotation/a.xml", "SAFE/measurement/m.tiff", "SAFE/stored.bin"]
        @test [m.method for m in ms] == [8, 8, 0]
        @test read_member(uri, ms[1]; transport) == small
        @test read_member(uri, ms[3]; transport) == big[1:1000]
        e = ZipEntry(uri, ms[2]; transport, span = 64 * 1024)
        @test length(e.points) > 10
        for (offset, n) in ((0, 1), (0, 4096), (length(big) - 1, 1), (1_234_567, 100_000), (65_535, 2),
                            (2_000_001, 999_999))
            @test read_entry(e, offset, n) == big[(offset + 1):(offset + n)]
        end
        s = ZipEntry(uri, ms[3]; transport)
        @test read_entry(s, 10, 20) == big[11:30]
        @test_throws BoundsError read_entry(e, length(big) - 1, 2)
        mktempdir() do out
            # Blocks smaller than the member and fewer in flight than blocks, so batches follow batches.
            for (k, want) in ((2, big), (3, big[1:1000]))
                dest = joinpath(out, "m$k")
                @test extract_member(uri, ms[k], dest; transport, block = 100_000, inflight = 3) == dest
                @test read(dest) == want
                @test !isfile(dest * ".part")
            end
            # Ranges of a deflated and of a stored member, the last ending at the member's end.
            for (k, want, rs) in ((2, big, [(0, 10), (100_000, 250_000), (2_999_000, 1000)]),
                                  (3, big[1:1000], [(5, 10), (500, 500)]))
                ios = [IOBuffer() for _ in rs]
                @test inflate_ranges!(ios, uri, ms[k], rs; transport, block = 100_000, inflight = 3) === ios
                for (io, (o, n)) in zip(ios, rs)
                    @test take!(io) == want[(o + 1):(o + n)]
                end
            end
            @test read_member(uri, ms[2], 123, 4567; transport) == big[124:(123 + 4567)]
            @test read_member(uri, ms[3], 0, 0; transport) == UInt8[]
            @test_throws "must ascend without overlapping" inflate_ranges!(
                [IOBuffer(), IOBuffer()], uri, ms[2], [(100, 10), (105, 10)]; transport)
            @test_throws "one stream per range" inflate_ranges!([IOBuffer()], uri, ms[2], [(0, 1), (2, 1)]; transport)
            @test_throws BoundsError read_member(uri, ms[2], length(big) - 1, 2; transport)
        end
    end
    # Few, long blocks of long matches: the last access point is far before the end, and the last bytes come
    # out of matches inflate holds after its input is used up.
    periodic = UInt8[UInt8((i * 7 + (i >> 9)) % 251) for i in 1:3_000_000]
    mktempdir() do dir
        p = joinpath(dir, "periodic.zip")
        write(p, _zip_fixture([("m", periodic, true)]))
        e = ZipEntry(p, only(zip_members(p; transport = LocalTransport())); transport = LocalTransport(),
                     span = 64 * 1024)
        @test read_entry(e, length(periodic) - 1, 1) == periodic[end:end]
        @test read_entry(e, 2_900_000, 100_000) == periodic[2_900_001:end]
        @test read(extract_member(p, only(zip_members(p; transport = LocalTransport())), joinpath(dir, "out");
                                  transport = LocalTransport(), block = 4096)) == periodic
    end
    mktempdir() do dir
        path = joinpath(dir, "fixture.zip")
        write(path, bytes)
        check(path, LocalTransport())
        write(joinpath(dir, "not.zip"), zeros(UInt8, 100))
        @test_throws "no zip end-of-central-directory record" zip_members(joinpath(dir, "not.zip");
                                                                          transport = LocalTransport())
    end
    _withserver() do ts, base
        _addroute!(ts, "/fixture.zip", bytes)
        check(base * "/fixture.zip", HTTPTransport())
    end
end
