# No network access: everything here runs offline. A real S3 integration
# test is gated behind VIRTUALZARR_TEST_S3 (see the bottom testset) and is
# off by default.

@testset "S3Transport" begin
    @testset "stub error without AWSS3" begin
        # Base.get_extension returns `nothing` until something `using`s
        # AWSS3; this testset must run before that happens anywhere in the
        # process, so it comes first in this file and no earlier include in
        # runtests.jl loads AWSS3.
        @test Base.get_extension(VirtualZarr, :VirtualZarrAWSS3Ext) === nothing
        @test_throws "AWSS3 must be loaded" VirtualZarr.S3Transport("my-bucket")
    end

    using AWSS3
    ext = Base.get_extension(VirtualZarr, :VirtualZarrAWSS3Ext)
    @test ext !== nothing

    # A config built from explicit, fake keys, never from the environment,
    # `~/.aws`, or instance metadata — so these tests pass on a machine with
    # no AWS credentials at all.
    fakeconfig() = AWSS3.AWS.AWSConfig(;
        creds=AWSS3.AWS.AWSCredentials("AKIAFAKEFAKEFAKEFAKE", "fakesecret"),
        region="us-west-2",
    )

    @testset "construction" begin
        config = fakeconfig()
        t = VirtualZarr.S3Transport("my-bucket"; aws=config)
        @test t isa VirtualZarr.S3Transport
        @test t.bucket == "my-bucket"
        @test t.aws === config

        # Two-field inner constructor stays usable directly.
        t2 = VirtualZarr.S3Transport("direct-bucket", config)
        @test t2.bucket == "direct-bucket"
        @test t2.aws === config

        # The no-aws= default path (current_aws_config()) needs real AWS
        # credentials on the machine to resolve and is not exercised here;
        # that's the caller's concern per the package contract, not
        # something a test can fake convincingly.
    end

    @testset "requester-pays header" begin
        config = fakeconfig()
        t = VirtualZarr.S3Transport("my-bucket"; aws=config, requesterpays=true)
        @test ext._awsconfig(t.aws) === config
        @test ext._headers(t.aws) == Dict("x-amz-request-payer" => "requester")

        plain = VirtualZarr.S3Transport("my-bucket"; aws=config)
        @test isempty(ext._headers(plain.aws))
    end

    @testset "byte_range mapping (the off-by-one that matters)" begin
        BR = VirtualZarr.ByteRange
        # 0-based half-open ByteRange(offset, nbytes) -> 1-based inclusive
        # AWSS3 byte_range. Checked against hand-computed values, not the
        # code under test, for offset 0, a nonzero offset, a single byte,
        # and the last byte of a 100-byte object.
        @test ext._awss3_byterange(BR(0, 10)) == 1:10
        @test ext._awss3_byterange(BR(100, 50)) == 101:150
        @test ext._awss3_byterange(BR(5, 1)) == 6:6
        @test ext._awss3_byterange(BR(99, 1)) == 100:100
    end

    @testset "uri resolution" begin
        t = VirtualZarr.S3Transport("default-bucket"; aws=fakeconfig())

        @test ext._s3_bucket_key(t, "some/key.h5") == ("default-bucket", "some/key.h5")
        @test ext._s3_bucket_key(t, "s3://other-bucket/some/key.h5") ==
            ("other-bucket", "some/key.h5")
        # A full s3:// uri resolves against its own bucket even when it
        # differs from the transport's, since one manifest can span buckets.
        @test ext._s3_bucket_key(t, "s3://other-bucket/key") != ("default-bucket", "key")

        @test_throws "malformed" ext._s3_bucket_key(t, "s3://no-key-bucket")
        @test_throws "malformed" ext._s3_bucket_key(t, "s3://")
    end

    @testset "zero-length range" begin
        # Must short-circuit before issuing any request: a 0-byte
        # byte_range would otherwise be the nonsensical "bytes=6-5".
        t = VirtualZarr.S3Transport("my-bucket"; aws=fakeconfig())
        @test VirtualZarr.fetchrange(t, "some/key", VirtualZarr.ByteRange(5, 0)) == UInt8[]
        @test VirtualZarr.fetchrange(t, "s3://other/key", VirtualZarr.ByteRange(0, 0)) ==
            UInt8[]
    end

    @testset "malformed uri fails before any network call" begin
        t = VirtualZarr.S3Transport("my-bucket"; aws=fakeconfig())
        @test_throws "malformed" VirtualZarr.fetchrange(
            t, "s3://no-key-bucket", VirtualZarr.ByteRange(0, 10)
        )
    end
end

@testset "S3Transport live integration (opt-in)" begin
    if get(ENV, "VIRTUALZARR_TEST_S3", "false") == "true"
        bucket = ENV["VIRTUALZARR_TEST_S3_BUCKET"]
        key = ENV["VIRTUALZARR_TEST_S3_KEY"]
        t = VirtualZarr.S3Transport(bucket)
        data = VirtualZarr.fetchrange(t, key, VirtualZarr.ByteRange(0, 16))
        @test length(data) == 16
    else
        @test_skip "set VIRTUALZARR_TEST_S3=true, VIRTUALZARR_TEST_S3_BUCKET and " *
            "VIRTUALZARR_TEST_S3_KEY (a real, readable object, >=16 bytes) " *
            "under valid AWS credentials to run this against live S3"
    end
end
