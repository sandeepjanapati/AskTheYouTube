import os
import logging
from google.cloud import secretmanager
import google.auth

logger = logging.getLogger(__name__)

class SecretManager:
    """
    Utility to fetch secure keys from Google Cloud Secret Manager.
    """
    _client = None
    _project_id = None

    @classmethod
    def get_client(cls):
        if cls._client is None:
            cls._client = secretmanager.SecretManagerServiceClient()
        return cls._client

    @classmethod
    def get_project_id(cls):
        """
        Auto-detects the Google Cloud Project ID.
        """
        if cls._project_id is None:
            # Try env var first (Standard in Cloud Run)
            cls._project_id = os.getenv("GOOGLE_CLOUD_PROJECT_ID")
            
            # Fallback to default auth credentials if running locally/hybrid
            if not cls._project_id:
                _, cls._project_id = google.auth.default()
                
        return cls._project_id

    @staticmethod
    def get_secret(secret_id: str, version_id: str = "latest") -> str:
        """
        Fetches a secret payload.
        Args:
            secret_id: The name of the secret (e.g., 'pinecone-api-key').
            version_id: version (default 'latest').
        """
        try:
            client = SecretManager.get_client()
            project_id = SecretManager.get_project_id()
            
            name = f"projects/{project_id}/secrets/{secret_id}/versions/{version_id}"
            response = client.access_secret_version(request={"name": name})
            
            payload = response.payload.data.decode("UTF-8")
            return payload
        except Exception as e:
            logger.error(f"Failed to access secret '{secret_id}': {e}")
            raise RuntimeError(f"Critical Security Error: Could not retrieve secret {secret_id}")