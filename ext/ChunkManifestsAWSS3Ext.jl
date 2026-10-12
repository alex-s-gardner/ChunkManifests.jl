module ChunkManifestsAWSS3Ext

# Lets `S3Transport(bucket; aws)` take its credentials and region from an AWS.jl configuration. The
# transport itself signs and sends its requests without AWS.jl.

using ChunkManifests
import AWS
import AWSS3
import Dates

function ChunkManifests._s3_from_aws(aws::AWS.AbstractAWSConfig)
    c = AWS.credentials(aws)
    expires = c.expiry == typemax(Dates.DateTime) ? Inf : Dates.datetime2unix(c.expiry)
    creds = ChunkManifests._AWSCredentials(c.access_key_id, c.secret_key, c.token, expires)
    return creds, String(AWS.region(aws))
end

end # module ChunkManifestsAWSS3Ext
