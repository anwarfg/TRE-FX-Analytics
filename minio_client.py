import requests
import minio
from minio import Minio
import xml.etree.ElementTree as ET
from typing import Optional, List, Dict, Any
import time
import os


class TokenExpiredError(Exception):
    """Exception raised when the provided token has expired."""
    pass


class MinIOClient:
    """
    Handles MinIO operations including token exchange and object retrieval.
    """
    
    def __init__(self, token: str, 
                 sts_endpoint: str = None,
                 minio_endpoint: str = None):
        """
        Initialize the MinIO client.
        
        Args:
            token (str): OIDC token for authentication
            sts_endpoint (str): STS endpoint URL
            minio_endpoint (str): MinIO endpoint URL
        """
        self.token = token
        # Use environment variables - required
        self.sts_endpoint = sts_endpoint or os.getenv('MINIO_STS_ENDPOINT')
        if not self.sts_endpoint:
            raise ValueError("MINIO_STS_ENDPOINT environment variable is required")
        
        self.minio_endpoint = minio_endpoint or os.getenv('MINIO_ENDPOINT')
        if not self.minio_endpoint:
            raise ValueError("MINIO_ENDPOINT environment variable is required")
        
        self._client = None
        self._credentials = None
        self.credentials = None
        self.credentials_expiry = None
    
    def _exchange_token_for_credentials(self) -> Dict[str, str]:
        """
        Exchange OIDC token for temporary AWS credentials.
        
        Returns:
            Dict[str, str]: Dictionary containing access_key, secret_key, and session_token
            
        Raises:
            TokenExpiredError: If the token has expired
            Exception: If token exchange fails
        """
        headers = {
            'Authorization': f'Bearer {self.token}',
            'Content-Type': 'application/x-www-form-urlencoded'
        }
        
        data = {
            'Action': 'AssumeRoleWithWebIdentity',
            'Version': '2011-06-15',
            'WebIdentityToken': self.token,
            'DurationSeconds': '3600'
        }
        
        print("Exchanging token for credentials...")
        response = requests.post(self.sts_endpoint, headers=headers, data=data)
        
        if response.status_code != 200:
            print(f"STS Response Status: {response.status_code}")
            print(f"STS Response Headers: {response.headers}")
            print(f"STS Response Content: {response.text}")
            
            # Check for expired token
            if response.status_code == 400:
                error_content = response.text.lower()
                if "expired" in error_content or "token expired" in error_content:
                    raise TokenExpiredError("Token has expired")
            
            raise Exception(f"Failed to exchange token: {response.status_code} - {response.text}")
        
        # Parse the STS response
        root = ET.fromstring(response.text)
        ns = {'sts': 'https://sts.amazonaws.com/doc/2011-06-15/'}
        
        credentials = root.find('.//sts:Credentials', ns)
        if credentials is None:
            raise Exception("No credentials found in STS response")
            
        access_key = credentials.find('sts:AccessKeyId', ns).text
        secret_key = credentials.find('sts:SecretAccessKey', ns).text
        session_token = credentials.find('sts:SessionToken', ns).text
        
        return {
            'access_key': access_key,
            'secret_key': secret_key,
            'session_token': session_token
        }
    
    def _get_client(self) -> Minio:
        """
        Get or create MinIO client with valid credentials.
        
        Returns:
            Minio: MinIO client instance
        """
        if self._client is None or self._credentials is None:
            self._credentials = self._exchange_token_for_credentials()
            
            self._client = Minio(
                self.minio_endpoint,
                access_key=self._credentials['access_key'],
                secret_key=self._credentials['secret_key'],
                session_token=self._credentials['session_token'],
                secure=False
            )
        
        return self._client
    
    def refresh_credentials(self):
        """Force refresh of credentials."""
        self._credentials = None
        self._client = None
    
    def get_object(self, bucket: str, object_path: str) -> Optional[str]:
        """
        Get object content from MinIO.
        
        Args:
            bucket (str): Name of the MinIO bucket
            object_path (str): Path to the object within the bucket
            
        Returns:
            Optional[str]: Contents of the file as a string, or None if not found
        """
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                client = self._get_client()
                
                print(f"Getting object '{object_path}' from bucket '{bucket}'...")
                response = client.get_object(bucket, object_path)
                
                # Read and decode the content
                content = response.read().decode('utf-8')
                response.close()
                response.release_conn()
                
                return content
                
            except minio.error.S3Error as e:
                if e.code == 'NoSuchKey':
                    print(f"Object not found: {object_path} in bucket {bucket}")
                    return None
                elif e.code == 'ExpiredTokenException':
                    print("Token expired, refreshing credentials...")
                    self.refresh_credentials()
                    retry_count += 1
                    continue
                else:
                    print(f"MinIO error: {e}")
                    retry_count += 1
                    continue
                    
            except Exception as e:
                print(f"Error accessing MinIO: {str(e)}")
                print(f"Error type: {type(e).__name__}")
                
                if retry_count < max_retries - 1:
                    retry_count += 1
                    print(f"\nRetrying... (Attempt {retry_count + 1} of {max_retries})")
                    time.sleep(2 ** retry_count)  # Exponential backoff
                    continue
                else:
                    print(f"Failed after {max_retries} attempts")
                    return None
        
        return None
    
    def list_objects(self, bucket: str, prefix: str = "") -> List[str]:
        """
        List objects in a bucket.
        
        Args:
            bucket (str): Name of the MinIO bucket
            prefix (str): Prefix to filter objects
            
        Returns:
            List[str]: List of object names
        """
        try:
            client = self._get_client()
            objects = client.list_objects(bucket, prefix=prefix)
            return [obj.object_name for obj in objects]
        except Exception as e:
            print(f"Error listing objects: {str(e)}")
            return []
    
    def list_buckets(self) -> List[str]:
        """
        List available buckets.
        
        Returns:
            List[str]: List of bucket names
        """
        try:
            client = self._get_client()
            buckets = client.list_buckets()
            return [bucket.name for bucket in buckets]
        except Exception as e:
            print(f"Error listing buckets: {str(e)}")
            return []
    
    def bucket_exists(self, bucket: str) -> bool:
        """
        Check if a bucket exists.
        
        Args:
            bucket (str): Name of the bucket
            
        Returns:
            bool: True if bucket exists, False otherwise
        """
        try:
            client = self._get_client()
            return client.bucket_exists(bucket)
        except Exception as e:
            print(f"Error checking bucket existence: {str(e)}")
            return False
    
    def object_exists(self, bucket: str, object_path: str) -> bool:
        """
        Check if an object exists.
        
        Args:
            bucket (str): Name of the bucket
            object_path (str): Path to the object
            
        Returns:
            bool: True if object exists, False otherwise
        """
        try:
            client = self._get_client()
            client.stat_object(bucket, object_path)
            return True
        except minio.error.S3Error as e:
            if e.code == 'NoSuchKey':
                return False
            else:
                print(f"Error checking object existence: {e}")
                return False
        except Exception as e:
            print(f"Error checking object existence: {str(e)}")
            return False
    
    def get_object_info(self, bucket: str, object_path: str) -> Optional[Dict]:
        """
        Get information about an object.
        
        Args:
            bucket (str): Name of the bucket
            object_path (str): Path to the object
            
        Returns:
            Optional[Dict]: Object information or None if not found
        """
        try:
            client = self._get_client()
            stat = client.stat_object(bucket, object_path)
            return {
                'size': stat.size,
                'last_modified': stat.last_modified,
                'etag': stat.etag,
                'content_type': stat.content_type
            }
        except minio.error.S3Error as e:
            if e.code == 'NoSuchKey':
                return None
            else:
                print(f"Error getting object info: {e}")
                return None
        except Exception as e:
            print(f"Error getting object info: {str(e)}")
            return None
    
    def wait_for_object(self, bucket: str, object_path: str, timeout: int = 300, check_interval: int = 10) -> Optional[str]:
        """
        Wait for an object to appear and return its content.
        
        Args:
            bucket (str): Name of the bucket
            object_path (str): Path to the object
            timeout (int): Maximum time to wait in seconds
            check_interval (int): Time between checks in seconds
            
        Returns:
            Optional[str]: Object content or None if timeout
        """
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            content = self.get_object(bucket, object_path)
            if content is not None:
                return content
            
            print(f"Object {object_path} not found, waiting {check_interval} seconds...")
            time.sleep(check_interval)
        
        print(f"Timeout waiting for object {object_path}")
        return None


# Legacy function for backward compatibility
def access_minio_with_token(token: str, bucket: str, object_path: str, 
                           list_objects: bool = False, list_buckets: bool = False) -> str:
    """
    Legacy function for backward compatibility.
    
    Args:
        token (str): OIDC token for authentication
        bucket (str): Name of the MinIO bucket (required)
        object_path (str): Path to the object within the bucket (required)
        list_objects (bool): Whether to list objects in bucket
        list_buckets (bool): Whether to list available buckets
        
    Returns:
        str: Contents of the file as a string
    """
    client = MinIOClient(token)
    
    if list_buckets:
        buckets = client.list_buckets()
        print("Available buckets:", buckets)
        
        if not buckets:
            raise Exception("No buckets available")
    
    if list_objects:
        objects = client.list_objects(bucket)
        print(f"Listing objects in bucket '{bucket}':")
        for obj in objects:
            print(f"- {obj}")
    
    content = client.get_object(bucket, object_path)
    if content is None:
        raise Exception(f"Failed to retrieve object {object_path} from bucket {bucket}")
    
    return content 