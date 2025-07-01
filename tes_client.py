import json
import os
import requests
from typing import Dict, Any, List, Tuple
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class TESClient:
    """
    Handles TES (Task Execution Service) operations including task generation and submission.
    """
    
    def __init__(self, 
                 base_url: str = None,
                 default_image: str = None,
                 default_db_config: Dict[str, str] = None,
                 default_db_port: str = None):
        """
        Initialize the TES client.
        
        Args:
            base_url (str): Base URL for the TES API
            default_image (str): Default Docker image to use
            default_db_config (Dict[str, str]): Default database configuration
            default_db_port (str): Default database port
        """
        # Use environment variables - required
        self.base_url = base_url or os.getenv('TES_BASE_URL')
        if not self.base_url:
            raise ValueError("TES_BASE_URL environment variable is required")
        
        self.default_image = default_image or os.getenv('TES_DOCKER_IMAGE')
        if not self.default_image:
            raise ValueError("TES_DOCKER_IMAGE environment variable is required")
        
        default_db_port = default_db_port or os.getenv('DB_PORT')
        if not default_db_port:
            raise ValueError("DB_PORT environment variable is required")
        
        if default_db_config is None:
            db_host = os.getenv('DB_HOST')
            if not db_host:
                raise ValueError("DB_HOST environment variable is required")
            
            db_username = os.getenv('DB_USERNAME')
            if not db_username:
                raise ValueError("DB_USERNAME environment variable is required")
            
            db_password = os.getenv('DB_PASSWORD')
            if not db_password:
                raise ValueError("DB_PASSWORD environment variable is required")
            
            db_name = os.getenv('DB_NAME')
            if not db_name:
                raise ValueError("DB_NAME environment variable is required")
            
            self.default_db_config = {
                "host": db_host,
                "username": db_username,
                "password": db_password,
                "name": db_name,
                "port": default_db_port
            }
        else:
            self.default_db_config = default_db_config
    
    def generate_tes_task(self,
                         query: str,
                         name: str = "analysis test",
                         image: str = None,
                         db_config: Dict[str, str] = None,
                         output_path: str = "/outputs") -> Dict[str, Any]:
        """
        Generate a TES task JSON configuration.
        
        Args:
            query (str): SQL query to execute
            name (str): Name of the analysis task
            image (str): Docker image to use (uses default if None)
            db_config (Dict[str, str]): Database configuration (uses default if None)
            output_path (str): Path for output files
            
        Returns:
            Dict[str, Any]: TES task configuration
        """
        if image is None:
            image = self.default_image
        
        if db_config is None:
            db_config = self.default_db_config
        
        task = {
            "name": name,
            "inputs": [],
            "outputs": [
                {
                    "url": "s3://beacon7283outputtre",
                    "path": output_path,
                    "type": "DIRECTORY",
                    "name": "workdir"
                }
            ],
            "executors": [
                {
                    "image": image,
                    "command": [
                        f"--Connection=Host={db_config['host']}:{db_config['port']};Username={db_config['username']};Password={db_config['password']};Database={db_config['name']}",
                        f"--Output={output_path}/output.csv",
                        f"--Query={query}"
                    ],
                    "env": {
                        "DATASOURCE_DB_DATABASE": db_config['name'],
                        "DATASOURCE_DB_HOST": db_config['host'],
                        "DATASOURCE_DB_PASSWORD": db_config['password'],
                        "DATASOURCE_DB_USERNAME": db_config['username']
                    },
                    "workdir": "/app"
                }
            ]
        }
        return task
    
    def save_tes_task(self, task: Dict[str, Any], output_file: str):
        """
        Save the TES task configuration to a JSON file.
        
        Args:
            task (Dict[str, Any]): TES task configuration
            output_file (str): Path to save the JSON file
        """
        with open(output_file, 'w') as f:
            json.dump(task, f, indent=4)
    
    def generate_submission_template(
        self,
        name: str = "Analysis Submission Test",
        description: str = "Federated analysis task",
        tres: list = ["Nottingham"],
        project: str = None,
        output_bucket: str = None,
        output_path: str = "/outputs",
        image: str = None,
        db_config: dict = None,
        query: str = None
    ) -> tuple[dict, int]:
        """
        Generate a submission template JSON configuration.
        
        Args:
            name (str): Name of the analysis submission
            description (str): Description of the analysis task
            tres (list): List of TREs to run the analysis on
            project (str): Project name (defaults to TRE_FX_PROJECT env var)
            output_bucket (str): S3 bucket name for outputs (defaults to MINIO_OUTPUT_BUCKET env var)
            output_path (str): Path for output files
            image (str): Docker image to use
            db_config (dict): Database configuration
            query (str): SQL query to execute
        
        Returns:
            tuple[dict, int]: Submission template configuration and number of TREs
        """
        # Use environment variables for project and output bucket if not provided
        project = project or os.getenv('TRE_FX_PROJECT')
        if not project:
            raise ValueError("TRE_FX_PROJECT environment variable is required when project parameter is not provided")
        
        output_bucket = output_bucket or os.getenv('MINIO_OUTPUT_BUCKET')
        if not output_bucket:
            raise ValueError("MINIO_OUTPUT_BUCKET environment variable is required when output_bucket parameter is not provided")
        
        if image is None:
            image = self.default_image
        
        if db_config is None:
            db_config = self.default_db_config
        
        # If a TES task is provided, use its executor configuration
        if query is None:
            query = f"SELECT COUNT(*) FROM measurement WHERE measurement_concept_id = 3037532"
        
        executors = [{
            "image": image,
            "command": [
                f"--Connection=Host={db_config['host']}:{db_config['port']};Username={db_config['username']};Password={db_config['password']};Database={db_config['name']}",
                f"--Output={output_path}/output.csv",
                f"--Query={query}"
            ],
            "env": {
                "DATASOURCE_DB_DATABASE": db_config['name'],
                "DATASOURCE_DB_HOST": db_config['host'],
                "DATASOURCE_DB_PASSWORD": db_config['password'],
                "DATASOURCE_DB_USERNAME": db_config['username']
            },
            "workdir": "/app"
        }]
        
        template = {
            "id": None,
            "state": 1,
            "name": name,
            "description": description,
            "inputs": None,
            "outputs": [
                {
                    "name": "workdir",
                    "description": "analysis test output",
                    "url": f"s3://{output_bucket}",
                    "path": output_path,
                    "type": "DIRECTORY"
                }
            ],
            "resources": None,
            "executors": executors,
            "volumes": None,
            "tags": {
                "Project": project,
                "tres": "|".join(tres)
            },
            "logs": None,
            "creation_time": None,
            "tesTask": {
                "id": None,
                "state": 1,
                "name": name,
                "description": description,
                "inputs": None,
                "outputs": [
                    {
                        "name": "workdir",
                        "description": "analysis test output",
                        "url": f"s3://{output_bucket}",
                        "path": output_path,
                        "type": "DIRECTORY"
                    }
                ],
                "resources": None,
                "executors": executors,
                "volumes": None,
                "tags": {
                    "Project": project,
                    "tres": "|".join(tres)
                },
                "logs": None,
                "creation_time": None
            }
        }
        return template, len(tres)
    
    def save_submission_template(self, template: Dict[str, Any], output_file: str):
        """
        Save the submission template configuration to a JSON file.
        
        Args:
            template (Dict[str, Any]): Submission template configuration
            output_file (str): Path to save the JSON file
        """
        with open(output_file, 'w') as f:
            json.dump(template, f, indent=4)
    
    def generate_curl_command(self, template: Dict[str, Any]) -> str:
        """
        Generate a curl command for submitting the template.
        
        Args:
            template (Dict[str, Any]): Submission template configuration
            
        Returns:
            str: Formatted curl command
        """
        # Convert template to JSON string with proper escaping
        template_json = json.dumps(template).replace('"', '\\"')
        
        curl_command = f"""curl -X 'POST' \\
  '{self.base_url}' \\
  -H 'accept: text/plain' \\
  -H 'Authorization: Bearer **TOKEN-HERE**' \\
  -H 'Content-Type: application/json' \\
  -d '{template_json}'"""
        
        return curl_command
    
    def submit_task(self, template: Dict[str, Any], token: str) -> Dict[str, Any]:
        """
        Submit a TES task using the requests library.
        
        Args:
            template (Dict[str, Any]): The TES task template
            token (str): Authentication token
            
        Returns:
            Dict[str, Any]: Response from the server
            
        Raises:
            requests.exceptions.RequestException: If the request fails
        """
        headers = {
            'accept': 'text/plain',
            'Authorization': f'Bearer {token}',
            'Content-Type': 'application/json'
        }
        
        try:
            response = requests.post(self.base_url, headers=headers, json=template)
            
            # Debug: Print response details for 400 errors
            if response.status_code == 400:
                print(f"400 Bad Request Response:")
                print(f"Status Code: {response.status_code}")
                print(f"Response Headers: {dict(response.headers)}")
                print(f"Response Content: {response.text}")
            
            response.raise_for_status()  # Raise an exception for bad status codes
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error submitting task: {str(e)}")
            if hasattr(e.response, 'text'):
                print(f"Response content: {e.response.text}")
            raise
    
    def get_task_status(self, task_id: str, token: str) -> Dict[str, Any]:
        """
        Get the status of a submitted task.
        
        Args:
            task_id (str): Task ID
            token (str): Authentication token
            
        Returns:
            Dict[str, Any]: Task status information
        """
        headers = {
            'accept': 'application/json',
            'Authorization': f'Bearer {token}'
        }
        
        try:
            response = requests.get(f"{self.base_url}/{task_id}", headers=headers)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error getting task status: {str(e)}")
            raise
    
    def list_tasks(self, token: str, limit: int = 100) -> List[Dict[str, Any]]:
        """
        List recent tasks.
        
        Args:
            token (str): Authentication token
            limit (int): Maximum number of tasks to return
            
        Returns:
            List[Dict[str, Any]]: List of tasks
        """
        headers = {
            'accept': 'application/json',
            'Authorization': f'Bearer {token}'
        }
        
        params = {'limit': limit}
        
        try:
            response = requests.get(self.base_url, headers=headers, params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error listing tasks: {str(e)}")
            raise


if __name__ == "__main__":
    # Example usage
    client = TESClient()
    
    # Generate a simple task
    task = client.generate_tes_task(
        query="SELECT COUNT(*) FROM measurement WHERE measurement_concept_id = 3037532",
        name="Test Analysis"
    )
    
    # Save to file
    client.save_tes_task(task, "tes-task.json")
    
    # Generate submission template
    template, n_tres = client.generate_submission_template(
        name="Test Submission",
        tres=["Nottingham", "Nottingham 2"],
        project="TestProject"
    )
    
    # Save template
    client.save_submission_template(template, "submission_template.json")
    
    # Generate curl command
    curl_cmd = client.generate_curl_command(template)
    print("\nCurl command for submission:")
    print(curl_cmd) 