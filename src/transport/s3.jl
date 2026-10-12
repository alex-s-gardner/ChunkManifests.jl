# S3Transport: byte ranges from Amazon S3, or an S3-compatible endpoint, read with ranged GETs on the
# libcurl client (`curl.jl`) and signed with AWS Signature Version 4.

"""
    _AWSCredentials

One set of AWS credentials. `token` is empty for long-lived keys; `expires` is a Unix time, `Inf` for
credentials that do not expire.
"""
struct _AWSCredentials
    access_key::String
    secret_key::String
    token::String
    expires::Float64
end

# Where a transport's credentials come from, so expiring ones can be fetched again.
const _CREDS_GIVEN = 0      # passed by the caller, or read from the environment or a profile file
const _CREDS_ECS = 1        # the ECS / AWS Batch container credentials endpoint
const _CREDS_IMDS = 2       # the EC2 instance metadata service (IMDSv2)
const _CREDS_ANONYMOUS = 3  # unsigned requests

mutable struct _CredentialSource
    kind::Int
    current::_AWSCredentials
    lock::ReentrantLock
end

_CredentialSource(kind::Int, c::_AWSCredentials) = _CredentialSource(kind, c, ReentrantLock())

const _NO_CREDENTIALS = _AWSCredentials("", "", "", Inf)

# The metadata endpoints are link-local: reached directly, never through a proxy, and given little
# time to answer, so a machine without them learns that within a second.
const _METADATA_HOSTS = "169.254.169.254,169.254.170.2"
_metadata_client() = _CurlClient(; connect_timeout = 1, read_idle_timeout = 2, noproxy = _METADATA_HOSTS)

# A string member of a flat JSON object, as the credential endpoints return them.
function _jsonfield(body::AbstractString, name::AbstractString)
    m = match(Regex(string("\"", name, "\"\\s*:\\s*\"([^\"]*)\"")), body)
    return m === nothing ? "" : String(something(m[1], ""))
end

# The Unix time of an ISO 8601 UTC timestamp `yyyy-mm-ddTHH:MM:SS...`, as the credential endpoints
# write their `Expiration`; `Inf` for an empty or unreadable one, which is then never refreshed.
function _iso8601_unix(s::AbstractString)
    length(s) >= 19 || return Inf
    fields = Int[]
    for r in (1:4, 6:7, 9:10, 12:13, 15:16, 18:19)
        v = tryparse(Int, s[r])
        v === nothing && return Inf
        push!(fields, v)
    end
    return Dates.datetime2unix(Dates.DateTime(fields[1], fields[2], fields[3], fields[4], fields[5], fields[6]))
end

function _credentials_from_json(body::AbstractString, where::AbstractString)
    akid, secret = _jsonfield(body, "AccessKeyId"), _jsonfield(body, "SecretAccessKey")
    (isempty(akid) || isempty(secret)) &&
        error("the AWS credentials from $where carried no AccessKeyId and SecretAccessKey")
    token = _jsonfield(body, "Token")
    expiration = _jsonfield(body, "Expiration")
    expires = _iso8601_unix(expiration)
    return _AWSCredentials(akid, secret, token, expires)
end

function _get_text(client::_CurlClient, url::String, headers::Vector{String}; method::String = "GET")
    resp = only(_curl_get(client, [_CurlRequest(url, headers; method)]; concurrency = 1))
    isempty(resp.error) || return 0, resp.error
    return resp.status, String(copy(resp.body))
end

function _ecs_credentials()
    full = get(ENV, "AWS_CONTAINER_CREDENTIALS_FULL_URI", "")
    relative = get(ENV, "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI", "")
    url = isempty(relative) ? full : string("http://169.254.170.2", relative)
    isempty(url) && return nothing
    headers = String[]
    token = get(ENV, "AWS_CONTAINER_AUTHORIZATION_TOKEN", "")
    tokenfile = get(ENV, "AWS_CONTAINER_AUTHORIZATION_TOKEN_FILE", "")
    isempty(tokenfile) || (token = strip(read(tokenfile, String)))
    isempty(token) || push!(headers, string("Authorization: ", token))
    status, body = _get_text(_metadata_client(), url, headers)
    status == 200 || error("the container credentials endpoint $url answered $status: $body")
    return _credentials_from_json(body, "the container credentials endpoint")
end

function _imds_credentials()
    lowercase(get(ENV, "AWS_EC2_METADATA_DISABLED", "")) == "true" && return nothing
    client = _metadata_client()
    base = "http://169.254.169.254/latest"
    status, token = _get_text(client, string(base, "/api/token"),
                              ["X-aws-ec2-metadata-token-ttl-seconds: 21600"]; method = "PUT")
    status == 200 || return nothing
    auth = [string("X-aws-ec2-metadata-token: ", token)]
    status, roles = _get_text(client, string(base, "/meta-data/iam/security-credentials/"), auth)
    status == 200 || return nothing
    role = String(strip(first(split(roles, '\n'))))
    isempty(role) && return nothing
    status, body = _get_text(client, string(base, "/meta-data/iam/security-credentials/", role), auth)
    status == 200 || error("the instance metadata service answered $status for role $role: $body")
    return _credentials_from_json(body, "the instance metadata service")
end

# The `key = value` entries of section `section` of an AWS INI file (`credentials` or `config`).
function _ini_section(path::AbstractString, section::AbstractString)
    found = Dict{String, String}()
    isfile(path) || return found
    current = ""
    for raw in eachline(path)
        line = strip(raw)
        (isempty(line) || startswith(line, '#') || startswith(line, ';')) && continue
        if startswith(line, '[') && endswith(line, ']')
            current = String(strip(line[2:(end - 1)]))
            continue
        end
        current == section || continue
        i = findfirst('=', line)
        i === nothing && continue
        found[String(strip(line[1:(i - 1)]))] = String(strip(line[(i + 1):end]))
    end
    return found
end

_aws_profile() = get(ENV, "AWS_PROFILE", get(ENV, "AWS_DEFAULT_PROFILE", "default"))
_aws_credentials_file() = get(ENV, "AWS_SHARED_CREDENTIALS_FILE", joinpath(homedir(), ".aws", "credentials"))
_aws_config_file() = get(ENV, "AWS_CONFIG_FILE", joinpath(homedir(), ".aws", "config"))
_config_section(profile::AbstractString) = profile == "default" ? "default" : string("profile ", profile)

function _profile_credentials(profile::AbstractString)
    entries = merge(_ini_section(_aws_config_file(), _config_section(profile)),
                    _ini_section(_aws_credentials_file(), profile))
    akid, secret = get(entries, "aws_access_key_id", ""), get(entries, "aws_secret_access_key", "")
    if isempty(akid) || isempty(secret)
        for key in ("credential_process", "sso_session", "sso_start_url", "role_arn", "web_identity_token_file")
            haskey(entries, key) && error(
                "AWS profile \"$profile\" obtains its credentials through `$key`, which S3Transport " *
                    "does not support; export the resulting AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY and " *
                    "AWS_SESSION_TOKEN to the environment instead",
            )
        end
        return nothing
    end
    return _AWSCredentials(akid, secret, get(entries, "aws_session_token", ""), Inf)
end

"""
    _default_credentials(profile) -> _CredentialSource

The credentials an AWS SDK would find, searched in its order: the `AWS_ACCESS_KEY_ID` /
`AWS_SECRET_ACCESS_KEY` / `AWS_SESSION_TOKEN` environment variables; `profile` in
`~/.aws/credentials` and `~/.aws/config`; the ECS container credentials endpoint
(`AWS_CONTAINER_CREDENTIALS_RELATIVE_URI` or `_FULL_URI`), which AWS Batch uses; and the EC2
instance metadata service. Throws, naming what was tried, when none has credentials.
"""
function _default_credentials(profile::AbstractString)
    akid, secret = get(ENV, "AWS_ACCESS_KEY_ID", ""), get(ENV, "AWS_SECRET_ACCESS_KEY", "")
    isempty(akid) || isempty(secret) ||
        return _CredentialSource(_CREDS_GIVEN, _AWSCredentials(akid, secret, get(ENV, "AWS_SESSION_TOKEN", ""), Inf))
    c = _profile_credentials(profile)
    c === nothing || return _CredentialSource(_CREDS_GIVEN, c)
    e = _ecs_credentials()
    e === nothing || return _CredentialSource(_CREDS_ECS, e)
    i = _imds_credentials()
    i === nothing || return _CredentialSource(_CREDS_IMDS, i)
    error(
        "no AWS credentials found: none in AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY, none for " *
            "profile \"$profile\" in $(_aws_credentials_file()) or $(_aws_config_file()), no ECS " *
            "container credentials endpoint, and no EC2 instance role. For a public bucket pass " *
            "`anonymous = true`",
    )
end

# The credentials to sign with now, fetched again when they expire within five minutes.
function _credentials(src::_CredentialSource)
    src.kind == _CREDS_ANONYMOUS && return src.current
    return @lock src.lock begin
        if time() > src.current.expires - 300
            fresh = src.kind == _CREDS_ECS ? _ecs_credentials() :
                    src.kind == _CREDS_IMDS ? _imds_credentials() : nothing
            fresh === nothing && error("the AWS credentials expired and could not be fetched again")
            src.current = fresh
        end
        src.current
    end
end

function _default_region(profile::AbstractString)
    for name in ("AWS_REGION", "AWS_DEFAULT_REGION")
        r = get(ENV, name, "")
        isempty(r) || return r
    end
    r = get(_ini_section(_aws_config_file(), _config_section(profile)), "region", "")
    return isempty(r) ? "us-east-1" : r
end

"""
    S3Transport(bucket; region, requesterpays = false, profile, anonymous = false, endpoint,
                retries = 3, aws = nothing)

Reads byte ranges from objects in the S3 bucket `bucket`, with ranged GETs signed with AWS Signature
Version 4. A bare key is read from `bucket`; a full `s3://bucket/key` uri names its own bucket,
which must be in the same region.

Credentials are found as an AWS SDK finds them (environment, the `profile` of `~/.aws`, the ECS /
AWS Batch container endpoint, the EC2 instance role); `profile` defaults to `AWS_PROFILE`, and
temporary credentials are fetched again before they expire. `anonymous = true` sends unsigned
requests, for a public bucket. `region` defaults to `AWS_REGION`, then `AWS_DEFAULT_REGION`, then
the profile's, then `us-east-1`; a bucket in another region answers `301`. `requesterpays = true`
sends `x-amz-request-payer: requester` with every request, as NASA/USGS/ESA archive buckets require.
`endpoint` (default `AWS_ENDPOINT_URL_S3`, else Amazon's) addresses an S3-compatible server in path
style. `aws`, with AWS.jl and AWSS3.jl loaded, takes the credentials and region of an
`AWS.AbstractAWSConfig` instead.

As [`HTTPTransport`](@ref), the ranges of one [`fetchranges`](@ref) call run concurrently on one
libcurl multi handle, and transient failures are retried up to `retries` times.
"""
struct S3Transport <: AbstractTransport
    bucket::String
    region::String
    endpoint::String
    credentials::_CredentialSource
    headers::Vector{Pair{String, String}}
    client::_CurlClient
    retries::Int
end

function S3Transport(bucket::AbstractString; region = nothing, requesterpays::Bool = false,
                     profile = nothing, anonymous::Bool = false, endpoint = nothing,
                     retries::Integer = 3, aws = nothing)
    isempty(bucket) && throw(ArgumentError("S3Transport needs a bucket name"))
    retries >= 0 || throw(ArgumentError("retries must be nonnegative, got $retries"))
    prof = profile === nothing ? _aws_profile() : String(profile)
    creds, awsregion = if aws !== nothing
        c, r = _s3_from_aws(aws)
        _CredentialSource(_CREDS_GIVEN, c), r
    elseif anonymous
        _CredentialSource(_CREDS_ANONYMOUS, _NO_CREDENTIALS), ""
    else
        _default_credentials(prof), ""
    end
    reg = region !== nothing ? String(region) : !isempty(awsregion) ? awsregion : _default_region(prof)
    ep = endpoint !== nothing ? String(endpoint) : get(ENV, "AWS_ENDPOINT_URL_S3", "")
    headers = requesterpays ? ["x-amz-request-payer" => "requester"] : Pair{String, String}[]
    return S3Transport(String(bucket), reg, rstrip(ep, '/'), creds, headers, _CurlClient(), Int(retries))
end

"""
    _s3_from_aws(aws) -> (_AWSCredentials, region)

The credentials and region of an AWS.jl configuration. Defined by the AWSS3 extension.
"""
function _s3_from_aws(aws)
    throw(ArgumentError("S3Transport(...; aws) needs AWS.jl and AWSS3.jl loaded; without them, " *
                        "credentials come from the environment, ~/.aws, or the instance role"))
end

# A bare key is read from `t.bucket`. A full `s3://bucket/key` uri is read from the bucket it names,
# which may differ from `t.bucket` — a manifest's PathTable can legitimately span buckets. A
# malformed `s3://` uri (no key component) throws rather than guessing.
function _s3_bucket_key(t::S3Transport, uri::AbstractString)
    if startswith(uri, "s3://")
        rest = chop(uri; head = 5, tail = 0)
        parts = split(rest, '/'; limit = 2)
        length(parts) == 2 && !isempty(parts[1]) && !isempty(parts[2]) || throw(
            ArgumentError(
                "malformed s3:// uri, expected s3://bucket/key, got $(repr(uri))"
            )
        )
        return String(parts[1]), String(parts[2])
    else
        return t.bucket, String(uri)
    end
end

# An object key percent-encoded as SigV4 signs it: every byte but the unreserved characters and `/`.
function _s3_encode(key::AbstractString)
    io = IOBuffer()
    for b in codeunits(key)
        c = Char(b)
        if ('A' <= c <= 'Z') || ('a' <= c <= 'z') || ('0' <= c <= '9') || c in ('-', '_', '.', '~', '/')
            write(io, b)
        else
            write(io, '%', uppercase(string(b; base = 16, pad = 2)))
        end
    end
    return String(take!(io))
end

# The URL of `bucket/key` and the `Host` it is signed for: virtual-hosted on Amazon unless the bucket
# name holds a dot (its certificate would not match), path style on a custom endpoint.
function _s3_url(t::S3Transport, bucket::String, key::String)
    path = _s3_encode(key)
    if !isempty(t.endpoint)
        host = replace(t.endpoint, r"^https?://" => "")
        return string(t.endpoint, "/", bucket, "/", path), host, string("/", bucket, "/", path)
    elseif occursin('.', bucket)
        host = string("s3.", t.region, ".amazonaws.com")
        return string("https://", host, "/", bucket, "/", path), host, string("/", bucket, "/", path)
    else
        host = string(bucket, ".s3.", t.region, ".amazonaws.com")
        return string("https://", host, "/", path), host, string("/", path)
    end
end

_hmac(key::Vector{UInt8}, data::String) = SHA.hmac_sha256(key, Vector{UInt8}(codeunits(data)))

const _EMPTY_SHA256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"

"""
    _sigv4(c, region, host, path, headers, payloadhash, now) -> Vector{String}

The header lines of a SigV4-signed bodiless GET of `path` on `host`: `headers` (lowercase names)
plus `x-amz-date`, `x-amz-content-sha256`, `x-amz-security-token` when the credentials carry one,
and `Authorization`. Every header given is signed; `Host` is signed and left for libcurl to send.
"""
function _sigv4(c::_AWSCredentials, region::String, host::String, path::String,
                headers::Vector{Pair{String, String}}, payloadhash::String, now::Dates.DateTime)
    amzdate = Dates.format(now, Dates.dateformat"yyyymmdd\THHMMSS\Z")
    date = amzdate[1:8]
    signed = Pair{String, String}["host" => host, "x-amz-content-sha256" => payloadhash, "x-amz-date" => amzdate]
    append!(signed, headers)
    isempty(c.token) || push!(signed, "x-amz-security-token" => c.token)
    sort!(signed; by = first)
    canonical_headers = join(string(k, ":", strip(v), "\n") for (k, v) in signed)
    names = join((k for (k, _) in signed), ";")
    canonical = string("GET\n", path, "\n\n", canonical_headers, "\n", names, "\n", payloadhash)
    scope = string(date, "/", region, "/s3/aws4_request")
    to_sign = string("AWS4-HMAC-SHA256\n", amzdate, "\n", scope, "\n", bytes2hex(SHA.sha256(canonical)))
    k = _hmac(Vector{UInt8}(codeunits(string("AWS4", c.secret_key))), date)
    k = _hmac(k, region)
    k = _hmac(k, "s3")
    k = _hmac(k, "aws4_request")
    signature = bytes2hex(_hmac(k, to_sign))
    lines = [string(k, ": ", v) for (k, v) in signed if k != "host"]
    push!(lines, string("Authorization: AWS4-HMAC-SHA256 Credential=", c.access_key, "/", scope,
                        ", SignedHeaders=", names, ", Signature=", signature))
    return lines
end

# The URL of `uri` and the header lines of a request for it, signed now. `Range` is not signed, so
# one set of lines serves every range of a batch.
function _s3_request(t::S3Transport, uri::AbstractString)
    bucket, key = _s3_bucket_key(t, uri)
    url, host, path = _s3_url(t, bucket, key)
    c = _credentials(t.credentials)
    if t.credentials.kind == _CREDS_ANONYMOUS
        return url, [string(k, ": ", v) for (k, v) in t.headers]
    end
    return url, _sigv4(c, t.region, host, path, t.headers, "UNSIGNED-PAYLOAD", Dates.now(Dates.UTC))
end

function fetchall(t::S3Transport, uri::AbstractString)
    url, lines = _s3_request(t, uri)
    return _getall(t.client, lines, t.retries, url)
end

# See `concurrency` and `maxblock` for why these differ from the defaults.
concurrency(::S3Transport) = 32
maxblock(::S3Transport) = 16 * 1024 * 1024

"""
    fetchrange(t::S3Transport, uri, r::ByteRange) -> Vector{UInt8}

Read `r` from the S3 object named by `uri` (a bare key resolved against `t.bucket`, or a full
`s3://bucket/key` uri naming its own bucket). Throws if the object is missing, access is denied, or
the response is shorter than `r.nbytes`; never substitutes empty or truncated bytes for a failed read.
"""
function fetchrange(t::S3Transport, uri::AbstractString, r::ByteRange)
    r.nbytes == 0 && (_s3_bucket_key(t, uri); return UInt8[])
    url, lines = _s3_request(t, uri)
    return only(_fetchblocks(t.client, lines, t.retries, url, [r], 1))
end

function fetchranges(t::S3Transport, uri::AbstractString, ranges::AbstractVector{ByteRange})
    merged, mapping = coalesce_ranges(ranges; maxgap = maxgap(t), maxblock = maxblock(t))
    url, lines = _s3_request(t, uri)
    blocks = _fetchblocks(t.client, lines, t.retries, url, merged, concurrency(t))
    return _assemble(ranges, mapping, blocks)
end

"""
    objectsize(t::S3Transport, uri) -> UInt64

Size of the object at `uri`, from the `Content-Range` of a one-byte ranged GET — a GET rather than a
`HEAD` so that it carries the same signed headers, the requester-pays one included, as every read.
"""
function objectsize(t::S3Transport, uri::AbstractString)
    url, lines = _s3_request(t, uri)
    status, contentrange, body = _curlget(t.client, lines, t.retries, url, "bytes=0-0")
    status == 206 || error("HTTP $status sizing $(repr(uri)) ($url)$(_bodyexcerpt(body))")
    m = match(r"^bytes\s+\d+-\d+/(\d+)$", contentrange)
    m === nothing && error("sizing $(repr(uri)): could not read a total from Content-Range $(repr(contentrange))")
    return parse(UInt64, m[1])
end

# Both ends without a request spent sizing the object: the head request and a suffix request for the
# tail run concurrently and each states the object's size. On an object shorter than both together
# the two overlap, and the tail returned is only what the head does not hold.
function _fetchends(t::S3Transport, uri::AbstractString, head::Integer, tail::Integer)
    (head > 0 || tail > 0) || return UInt8[], UInt8[], objectsize(t, uri)
    url, lines = _s3_request(t, uri)
    tailtask = tail > 0 ? Threads.@spawn(_clipped(t.client, lines, t.retries, url, "bytes=-$tail")) : nothing
    headpart = head > 0 ? _clipped(t.client, lines, t.retries, url, "bytes=0-$(head - 1)") : nothing
    tailpart = tailtask === nothing ? nothing : fetch(tailtask)::Tuple{Vector{UInt8}, UInt64, UInt64}
    total = headpart !== nothing ? headpart[3] : something(tailpart)[3]
    (headpart === nothing || tailpart === nothing || tailpart[3] == total) || error(
        "$(repr(uri)) reported sizes $total and $(tailpart[3]) in two responses; it " *
            "changed while being read",
    )
    h = min(UInt64(head), total)
    tl = min(UInt64(tail), total - h)
    headbytes = h == 0 ? UInt8[] : _bodyspan(something(headpart)[1], something(headpart)[2], UInt64(0), h, uri)
    tailbytes = tl == 0 ? UInt8[] : _bodyspan(something(tailpart)[1], something(tailpart)[2], total - tl, tl, uri)
    return headbytes, tailbytes, total
end
