import boto3
import os


class BucketHelper():

    def __init__(self):
        # Connect to s3.
        s3 = boto3.resource(
            's3',
            aws_access_key_id=os.environ.get('AWSAccessKeyId'),
            aws_secret_access_key=os.environ.get('AWSSecretKey')
        )
        # Connect to bucket.
        self.bucket = s3.Bucket(os.environ.get('AWS_BUCKET_NAME'))

    def upload_img(self, img_bytes, filename):
        """Uploads file to the bucket, deletes it locally
        """
        tmp_filename = 'tmp/' + filename
        with open(tmp_filename, 'wb') as f:
            f.write(img_bytes)
        self.bucket.upload_file(tmp_filename, filename)
        os.remove(tmp_filename)

    def download_img(self, filename):
        self.bucket.download_file(filename, 'tmp/'+filename)
