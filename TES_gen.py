### update teh TES Json to update the name, query, and output?
### name and output are just to be able to identify the query, different filenames can be done when they're retrieved

import json
import requests
import os

def generate_tes_task(
    query: str,
    name: str = "analysis task",
    image: str = None,
    db_host: str = None,
    db_port: str = None,
    db_username: str = None,
    db_password: str = None,
    db_name: str = None,
    output_path: str = "/outputs",
    output_bucket: str = None
) -> dict:
    """
    Generate a TES task JSON configuration.
    
    Args:
        query (str): SQL query to execute
        name (str): Name of the analysis task
        image (str): Docker image to use
        db_host (str): Database host
        db_port (str): Database port
        db_username (str): Database username
        db_password (str): Database password
        db_name (str): Database name
        output_path (str): Path for output files
        output_bucket (str): S3 bucket for outputs (defaults to MINIO_OUTPUT_BUCKET env var)
    
    Returns:
        dict: TES task configuration
    """
    # Use environment variables - required
    image = image or os.getenv('TES_DOCKER_IMAGE')
    if not image:
        raise ValueError("TES_DOCKER_IMAGE environment variable is required")
    
    db_host = db_host or os.getenv('DB_HOST')
    if not db_host:
        raise ValueError("DB_HOST environment variable is required")
    
    db_port = db_port or os.getenv('DB_PORT')
    if not db_port:
        raise ValueError("DB_PORT environment variable is required")
    
    db_username = db_username or os.getenv('DB_USERNAME')
    if not db_username:
        raise ValueError("DB_USERNAME environment variable is required")
    
    db_password = db_password or os.getenv('DB_PASSWORD')
    if not db_password:
        raise ValueError("DB_PASSWORD environment variable is required")
    
    db_name = db_name or os.getenv('DB_NAME')
    if not db_name:
        raise ValueError("DB_NAME environment variable is required")
    
    # Use environment variable for output bucket if not provided
    output_bucket = output_bucket or os.getenv('MINIO_OUTPUT_BUCKET')
    if not output_bucket:
        raise ValueError("MINIO_OUTPUT_BUCKET environment variable is required when output_bucket parameter is not provided")
    
    task = {
        "name": name,
        "inputs": [],
        "outputs": [
            {
                "url": f"s3://{output_bucket}",
                "path": output_path,
                "type": "DIRECTORY",
                "name": "workdir"
            }
        ],
        "executors": [
            {
                "image": image,
                "command": [
                    f"--Connection=Host={db_host}:{db_port};Username={db_username};Password={db_password};Database={db_name}",
                    f"--Output={output_path}/output.csv",
                    f"--Query={query}"
                ],
                "env": {
                    "DATASOURCE_DB_DATABASE": db_name,
                    "DATASOURCE_DB_HOST": db_host,
                    "DATASOURCE_DB_PASSWORD": db_password,
                    "DATASOURCE_DB_USERNAME": db_username
                },
                "workdir": "/app"
            }
        ]
    }
    return task

def save_tes_task(task: dict, output_file: str):
    """
    Save the TES task configuration to a JSON file.
    
    Args:
        task (dict): TES task configuration
        output_file (str): Path to save the JSON file
    """
    with open(output_file, 'w') as f:
        json.dump(task, f, indent=4)

def generate_submission_template(
    tes_task: dict = None,
    name: str = "Analysis Submission Test",
    description: str = "Federated analysis task",
    tres: list = ["Nottingham"],
    project: str = None,
    output_bucket: str = None,
    output_path: str = "/outputs",
    image: str = None,
    db_host: str = None,
    db_port: str = None,
    db_username: str = None,
    db_password: str = None,
    db_name: str = None,
    query: str = None) -> tuple[dict, int]:
    """
    Generate a submission template JSON configuration.
    
    Args:
        tes_task (dict, optional): Existing TES task to use for executor configuration
        name (str): Name of the analysis submission
        description (str): Description of the analysis task
        tres (list): List of TREs to run the analysis on
        project (str): Project name (defaults to TRE_FX_PROJECT env var)
        output_bucket (str): S3 bucket name for outputs (defaults to MINIO_OUTPUT_BUCKET env var)
        output_path (str): Path for output files
        image (str): Docker image to use
        db_host (str): Database host
        db_port (str): Database port
        db_username (str): Database username
        db_password (str): Database password
        db_name (str): Database name
        query (str): SQL query to execute
    
    Returns:
        tuple[dict, int]: Submission template configuration and number of TREs
    """
    # Use environment variables - required
    image = image or os.getenv('TES_DOCKER_IMAGE')
    if not image:
        raise ValueError("TES_DOCKER_IMAGE environment variable is required")
    
    db_host = db_host or os.getenv('DB_HOST')
    if not db_host:
        raise ValueError("DB_HOST environment variable is required")
    
    db_port = db_port or os.getenv('DB_PORT')
    if not db_port:
        raise ValueError("DB_PORT environment variable is required")
    
    db_username = db_username or os.getenv('DB_USERNAME')
    if not db_username:
        raise ValueError("DB_USERNAME environment variable is required")
    
    db_password = db_password or os.getenv('DB_PASSWORD')
    if not db_password:
        raise ValueError("DB_PASSWORD environment variable is required")
    
    db_name = db_name or os.getenv('DB_NAME')
    if not db_name:
        raise ValueError("DB_NAME environment variable is required")
    
    # Use environment variables for project and output bucket if not provided
    project = project or os.getenv('TRE_FX_PROJECT')
    if not project:
        raise ValueError("TRE_FX_PROJECT environment variable is required when project parameter is not provided")
    
    output_bucket = output_bucket or os.getenv('MINIO_OUTPUT_BUCKET')
    if not output_bucket:
        raise ValueError("MINIO_OUTPUT_BUCKET environment variable is required when output_bucket parameter is not provided")
    
    # If a TES task is provided, use its executor configuration
    if tes_task is not None:
        executors = tes_task.get("executors", [])
    else:
        # Create executor configuration using provided parameters
        executors = [{
            "image": image,
            "command": [
                f"--Connection=Host={db_host}:{db_port};Username={db_username};Password={db_password};Database={db_name}",
                f"--Output={output_path}/output.csv",
                f"--Query={query}" if query else ""
            ],
            "env": {
                "DATASOURCE_DB_DATABASE": db_name,
                "DATASOURCE_DB_HOST": db_host,
                "DATASOURCE_DB_PASSWORD": db_password,
                "DATASOURCE_DB_USERNAME": db_username
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

def save_submission_template(template: dict, output_file: str):
    """
    Save the submission template configuration to a JSON file.
    
    Args:
        template (dict): Submission template configuration
        output_file (str): Path to save the JSON file
    """
    with open(output_file, 'w') as f:
        json.dump(template, f, indent=4)

def generate_curl_command(template: dict, base_url: str = None) -> str:
    """
    Generate a curl command for submitting the template.
    
    Args:
        template (dict): Submission template configuration
        base_url (str): Base URL for the API endpoint
    
    Returns:
        str: Formatted curl command
    """
    # Use environment variable - required
    base_url = base_url or os.getenv('TES_BASE_URL')
    if not base_url:
        raise ValueError("TES_BASE_URL environment variable is required")
    
    # Convert template to JSON string with proper escaping
    template_json = json.dumps(template).replace('"', '\"')
    
    curl_command = f"""curl -X 'POST' \\
  '{base_url}' \\
  -H 'accept: text/plain' \\
  -H 'Authorization: Bearer **TOKEN-HERE**' \\
  -H 'Content-Type: application/json-patch+json' \\
  -d '{template_json}'"""
    
    return curl_command

def submit_tes_task(template: dict, token: str, base_url: str = None) -> dict:
    """
    Submit a TES task using the requests library.
    
    Args:
        template (dict): The TES task template
        token (str): Authentication token
        base_url (str): Base URL for the API endpoint
    
    Returns:
        dict: Response from the server
    
    Raises:
        requests.exceptions.RequestException: If the request fails
    """
    # Use environment variable - required
    base_url = base_url or os.getenv('TES_BASE_URL')
    if not base_url:
        raise ValueError("TES_BASE_URL environment variable is required")
    
    headers = {
        'accept': 'text/plain',
        'Authorization': f'Bearer {token}',
        'Content-Type': 'application/json-patch+json'
    }
    
    try:
        response = requests.post(base_url, headers=headers, json=template)
        response.raise_for_status()  # Raise an exception for bad status codes
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error submitting task: {str(e)}")
        if hasattr(e.response, 'text'):
            print(f"Response content: {e.response.text}")
        raise

# Example usage:
if __name__ == "__main__":
    # Example parameters
    task = generate_tes_task(
        query="SELECT g.concept_name AS gender_name, r.concept_name AS race_name, COUNT(*) AS n FROM person p JOIN concept g ON p.gender_concept_id = g.concept_id JOIN concept r ON p.race_concept_id = r.concept_id WHERE p.race_concept_id IN (38003574, 38003584) GROUP BY g.concept_name, r.concept_name ORDER BY g.concept_name, r.concept_name;",
    )
    
    # Save to file
    save_tes_task(task, "TRE-FX Analytics/TES/tes-task.json")
    
    # Generate and save submission template using the TES task
    template = generate_submission_template(tes_task=task)
    save_submission_template(template, "TRE-FX Analytics/TES/submission_template.json")
    
    # Generate curl command
    curl_cmd = generate_curl_command(template)
    print("\nCurl command for submission:")
    print(curl_cmd)

