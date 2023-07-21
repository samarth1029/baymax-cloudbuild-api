import os
from google.cloud import secretmanager
from dotenv import load_dotenv


def get_api_key():
    if os.getenv("GAE_INSTANCE") is None:
        load_dotenv()
        return os.getenv("OPENAI_API_KEY")
    else:
        client = secretmanager.SecretManagerServiceClient()
        secret_version_name = "projects/148496638216/secrets/OPENAI_API_KEY/versions/1"
        response = client.access_secret_version(request={"name": secret_version_name})
        return response.payload.data.decode("UTF-8")
