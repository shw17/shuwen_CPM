import base64
import json

import boto3


class AWSSecretManager:
    """ Class for retrieve secrets from AWS Secret Manager """

    def __init__(self, region):
        self.client = boto3.client("secretsmanager", region_name=region)

    def get_secret(self, secret_name):
        response = self.client.get_secret_value(SecretId=secret_name)

        if "SecretString" in response:
            secret = response["SecretString"]
            return json.loads(secret)

        secret = response["SecretBinary"]
        return base64.b64decode(secret)