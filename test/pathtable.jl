@testset "PathTable" begin
    @testset "push_uri! dedup" begin
        t = PathTable()
        i1 = push_uri!(t, "s3://bucket/a.h5"; etag = "e1", size = 100, mtime = 1.0)
        @test i1 == 1
        @test length(t) == 1

        i2 = push_uri!(t, "s3://bucket/a.h5")
        @test i2 == i1
        @test length(t) == 1

        i3 = push_uri!(t, "s3://bucket/b.h5")
        @test i3 == 2
        @test length(t) == 2

        @test uriof(t, 1) == "s3://bucket/a.h5"
        @test uriof(t, 2) == "s3://bucket/b.h5"

        entry = t[1]
        @test entry.uri == "s3://bucket/a.h5"
        @test entry.etag == "e1"
        @test entry.size == UInt64(100)
        @test entry.mtime == 1.0
    end

    @testset "push_uri! conflicting metadata throws" begin
        t = PathTable()
        push_uri!(t, "a.h5"; etag = "e1")
        @test_throws "etag" push_uri!(t, "a.h5"; etag = "e2")

        push_uri!(t, "b.h5"; size = 10)
        @test_throws "size" push_uri!(t, "b.h5"; size = 20)

        push_uri!(t, "c.h5"; mtime = 1.0)
        @test_throws "mtime" push_uri!(t, "c.h5"; mtime = 2.0)

        # Filling in previously-unset metadata is not a conflict.
        i = push_uri!(t, "d.h5")
        @test push_uri!(t, "d.h5"; etag = "whatever") == i
        @test length(t) == 4
    end

    @testset "seturi! keeps lookup consistent" begin
        t = PathTable()
        i = push_uri!(t, "old/path.h5"; etag = "e1")
        seturi!(t, i, "new/path.h5")

        @test uriof(t, i) == "new/path.h5"
        @test t[i].etag == "e1"

        @test push_uri!(t, "new/path.h5") == i

        j = push_uri!(t, "old/path.h5")
        @test j != i
        @test length(t) == 2
    end

    @testset "seturi! rejects collision" begin
        t = PathTable()
        i1 = push_uri!(t, "a.h5")
        i2 = push_uri!(t, "b.h5")
        @test_throws "already names" seturi!(t, i1, "b.h5")
        @test uriof(t, i1) == "a.h5"
        @test uriof(t, i2) == "b.h5"
    end

    @testset "replace_prefix! keeps lookup consistent" begin
        t = PathTable()
        i1 = push_uri!(t, "s3://old-bucket/a.h5")
        i2 = push_uri!(t, "s3://old-bucket/b.h5")
        i3 = push_uri!(t, "s3://other-bucket/c.h5")

        # Returns the table, as Base.replace! does, so the call chains.
        @test replace_prefix!(t, "s3://old-bucket" => "s3://new-bucket") === t
        @test uriof(t, i1) == "s3://new-bucket/a.h5"
        @test uriof(t, i2) == "s3://new-bucket/b.h5"
        @test uriof(t, i3) == "s3://other-bucket/c.h5"

        @test push_uri!(t, "s3://new-bucket/a.h5") == i1

        j = push_uri!(t, "s3://old-bucket/a.h5")
        @test j != i1
        @test length(t) == 4
    end

    @testset "replace_prefix! rejects collision" begin
        t = PathTable()
        i1 = push_uri!(t, "old/a.h5")
        push_uri!(t, "new/a.h5")
        @test_throws "collides" replace_prefix!(t, "old" => "new")
        @test uriof(t, i1) == "old/a.h5"
    end
end
