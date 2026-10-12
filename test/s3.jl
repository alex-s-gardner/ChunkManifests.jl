# No network access: everything here runs offline, against the local HTTP fixture of `http.jl` where
# a server is needed. A real S3 integration test is gated behind CHUNKMANIFESTS_TEST_S3 (see the
# bottom testset) and is off by default.

import Dates
import HTTP
import Random

# Explicit, fake keys, never those of the machine running the tests.
const _FAKE_ENV = ["AWS_ACCESS_KEY_ID" => "AKIAFAKEFAKEFAKEFAKE", "AWS_SECRET_ACCESS_KEY" => "fakesecret",
                   "AWS_SESSION_TOKEN" => nothing, "AWS_REGION" => "us-west-2", "AWS_DEFAULT_REGION" => nothing,
                   "AWS_PROFILE" => nothing, "AWS_ENDPOINT_URL_S3" => nothing]

@testset "S3Transport" begin
    S3 = ChunkManifests.S3Transport

    @testset "SigV4 matches AWS's published GET Object example" begin
        # docs.aws.amazon.com/AmazonS3/latest/API/sig-v4-header-based-auth.html, "GET Object".
        c = ChunkManifests._AWSCredentials("AKIAIOSFODNN7EXAMPLE", "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY", "", Inf)
        lines = ChunkManifests._sigv4(c, "us-east-1", "examplebucket.s3.amazonaws.com", "/test.txt",
                                      ["range" => "bytes=0-9"], ChunkManifests._EMPTY_SHA256,
                                      Dates.DateTime(2013, 5, 24))
        auth = only(filter(startswith("Authorization: "), lines))
        @test auth == "Authorization: AWS4-HMAC-SHA256 " *
                      "Credential=AKIAIOSFODNN7EXAMPLE/20130524/us-east-1/s3/aws4_request, " *
                      "SignedHeaders=host;range;x-amz-content-sha256;x-amz-date, " *
                      "Signature=f0e8bdb87c964420e857bd35b5d6ed310bd44f0170aba48dd91039c6036bdb41"
        @test "x-amz-date: 20130524T000000Z" in lines
        @test !any(startswith("host:"), lines)  # libcurl sends Host itself
    end

    @testset "a session token is sent and signed" begin
        c = ChunkManifests._AWSCredentials("AKID", "secret", "the-token", Inf)
        lines = ChunkManifests._sigv4(c, "us-west-2", "b.s3.us-west-2.amazonaws.com", "/k",
                                      Pair{String, String}[], "UNSIGNED-PAYLOAD", Dates.DateTime(2024, 1, 2, 3, 4, 5))
        @test "x-amz-security-token: the-token" in lines
        @test occursin("SignedHeaders=host;x-amz-content-sha256;x-amz-date;x-amz-security-token,",
                       only(filter(startswith("Authorization: "), lines)))
    end

    @testset "construction from the environment" begin
        withenv(_FAKE_ENV...) do
            t = S3("my-bucket")
            @test t.bucket == "my-bucket"
            @test t.region == "us-west-2"
            @test t.credentials.current.access_key == "AKIAFAKEFAKEFAKEFAKE"
            @test isempty(t.headers)
            @test S3("my-bucket"; region = "eu-central-1").region == "eu-central-1"
            @test S3("my-bucket"; requesterpays = true).headers == ["x-amz-request-payer" => "requester"]
        end
        @test S3("public-bucket"; anonymous = true).credentials.kind == ChunkManifests._CREDS_ANONYMOUS
        @test_throws "needs a bucket" S3("")
    end

    @testset "credentials and region from a profile" begin
        mktempdir() do dir
            creds = joinpath(dir, "credentials")
            config = joinpath(dir, "config")
            write(creds, """
                [default]
                aws_access_key_id = AKIADEFAULT
                aws_secret_access_key = defaultsecret

                [work]
                aws_access_key_id = AKIAWORK
                aws_secret_access_key = worksecret
                aws_session_token = worktoken

                [sso]
                credential_process = /usr/local/bin/get-creds
                """)
            write(config, """
                [profile work]
                region = ap-south-1
                """)
            env = ["AWS_ACCESS_KEY_ID" => nothing, "AWS_SECRET_ACCESS_KEY" => nothing, "AWS_REGION" => nothing,
                   "AWS_DEFAULT_REGION" => nothing, "AWS_SHARED_CREDENTIALS_FILE" => creds,
                   "AWS_CONFIG_FILE" => config, "AWS_PROFILE" => "work", "AWS_ENDPOINT_URL_S3" => nothing]
            withenv(env...) do
                t = S3("b")
                @test t.credentials.current.access_key == "AKIAWORK"
                @test t.credentials.current.token == "worktoken"
                @test t.region == "ap-south-1"
                d = S3("b"; profile = "default")
                @test d.credentials.current.access_key == "AKIADEFAULT"
                @test d.region == "us-east-1"
                @test_throws "credential_process" S3("b"; profile = "sso")
            end
        end
    end

    @testset "credential documents of the metadata endpoints" begin
        doc = """{"Code":"Success","AccessKeyId":"ASIAXYZ","SecretAccessKey":"s3cr3t",
                  "Token":"tok","Expiration":"2026-10-10T22:00:00Z"}"""
        c = ChunkManifests._credentials_from_json(doc, "a test")
        @test (c.access_key, c.secret_key, c.token) == ("ASIAXYZ", "s3cr3t", "tok")
        @test c.expires == Dates.datetime2unix(Dates.DateTime(2026, 10, 10, 22))
        @test_throws "carried no AccessKeyId" ChunkManifests._credentials_from_json("{}", "a test")
    end

    @testset "urls" begin
        withenv(_FAKE_ENV...) do
            t = S3("my-bucket")
            @test ChunkManifests._s3_url(t, "my-bucket", "a b/c.tif") ==
                  ("https://my-bucket.s3.us-west-2.amazonaws.com/a%20b/c.tif",
                   "my-bucket.s3.us-west-2.amazonaws.com", "/a%20b/c.tif")
            # A dotted bucket name would not match the virtual host's certificate.
            @test ChunkManifests._s3_url(t, "my.bucket", "k") ==
                  ("https://s3.us-west-2.amazonaws.com/my.bucket/k", "s3.us-west-2.amazonaws.com", "/my.bucket/k")
            e = S3("my-bucket"; endpoint = "http://127.0.0.1:9000/")
            @test ChunkManifests._s3_url(e, "my-bucket", "k") ==
                  ("http://127.0.0.1:9000/my-bucket/k", "127.0.0.1:9000", "/my-bucket/k")
        end
    end

    @testset "uri resolution" begin
        withenv(_FAKE_ENV...) do
            t = S3("default-bucket")
            @test ChunkManifests._s3_bucket_key(t, "some/key.h5") == ("default-bucket", "some/key.h5")
            # A full s3:// uri resolves against its own bucket even when it differs from the
            # transport's, since one manifest can span buckets.
            @test ChunkManifests._s3_bucket_key(t, "s3://other-bucket/some/key.h5") == ("other-bucket", "some/key.h5")
            @test_throws "malformed" ChunkManifests._s3_bucket_key(t, "s3://no-key-bucket")
            @test_throws "malformed" ChunkManifests._s3_bucket_key(t, "s3://")
        end
    end

    @testset "zero-length range issues no request" begin
        withenv(_FAKE_ENV...) do
            t = S3("my-bucket")
            @test ChunkManifests.fetchrange(t, "some/key", ChunkManifests.ByteRange(5, 0)) == UInt8[]
            @test ChunkManifests.fetchrange(t, "s3://other/key", ChunkManifests.ByteRange(0, 0)) == UInt8[]
            @test_throws "malformed" ChunkManifests.fetchrange(t, "s3://no-key-bucket", ChunkManifests.ByteRange(0, 10))
        end
    end

    @testset "signed reads from an S3-compatible endpoint" begin
        bytes = rand(Random.MersenneTwister(7), UInt8, 100_000)
        seen = Vector{Dict{String, String}}()
        seenlock = ReentrantLock()
        ts = _TestServer()
        _addroute!(ts, "/bucket/data.bin", bytes)
        server = HTTP.serve!("127.0.0.1", 0) do req
            @lock seenlock push!(seen, Dict(lowercase(String(k)) => String(v) for (k, v) in req.headers))
            _handle(ts, req)
        end
        try
            endpoint = "http://127.0.0.1:$(HTTP.port(server))"
            withenv(_FAKE_ENV...) do
                t = S3("bucket"; endpoint, requesterpays = true)
                @test ChunkManifests.fetchrange(t, "data.bin", ChunkManifests.ByteRange(10, 20)) == bytes[11:30]
                @test ChunkManifests.fetchrange(t, "s3://bucket/data.bin", ChunkManifests.ByteRange(0, 1)) == bytes[1:1]
                ranges = [ChunkManifests.ByteRange(o, 1000) for o in (0, 50_000, 99_000, 20_000)]
                @test ChunkManifests.fetchranges(t, "data.bin", ranges) ==
                      [bytes[(r.offset + 1):(r.offset + r.nbytes)] for r in ranges]
                @test ChunkManifests.objectsize(t, "data.bin") == length(bytes)
                head, tail, total = ChunkManifests._fetchends(t, "data.bin", 64, 32)
                @test (head, tail, total) == (bytes[1:64], bytes[(end - 31):end], length(bytes))
                @test_throws "404" ChunkManifests.fetchrange(t, "missing.bin", ChunkManifests.ByteRange(0, 1))
            end
            @test !isempty(seen)
            @test all(h -> startswith(get(h, "authorization", ""), "AWS4-HMAC-SHA256 Credential=AKIAFAKEFAKEFAKEFAKE/"), seen)
            @test all(h -> get(h, "x-amz-request-payer", "") == "requester", seen)
            @test all(h -> occursin("x-amz-request-payer", get(h, "authorization", "")), seen)
        finally
            close(server)
        end
    end
end

@testset "S3Transport live integration (opt-in)" begin
    if get(ENV, "CHUNKMANIFESTS_TEST_S3", "false") == "true"
        bucket = ENV["CHUNKMANIFESTS_TEST_S3_BUCKET"]
        key = ENV["CHUNKMANIFESTS_TEST_S3_KEY"]
        t = ChunkManifests.S3Transport(bucket)
        data = ChunkManifests.fetchrange(t, key, ChunkManifests.ByteRange(0, 16))
        @test length(data) == 16
        total = ChunkManifests.objectsize(t, key)
        head, tail, size = ChunkManifests._fetchends(t, key, 8, 8)
        @test size == total
        @test head == data[1:8]
        @test tail == ChunkManifests.fetchrange(t, key, ChunkManifests.ByteRange(total - 8, 8))
    else
        @test_skip "set CHUNKMANIFESTS_TEST_S3=true, CHUNKMANIFESTS_TEST_S3_BUCKET and " *
            "CHUNKMANIFESTS_TEST_S3_KEY (a real, readable object, >=16 bytes) " *
            "under valid AWS credentials to run this against live S3"
    end
end
