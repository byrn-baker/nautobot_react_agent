import requests
import json
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# URL to the OpenAPI specification
openapi_url = 'https://demo.nautobot.com/api/docs/?format=openapi'

# API token from environment variable
api_token = os.getenv("DEMO_NAUTOBOT_API")

# HTTP headers for authentication
headers = {
    'Authorization': f'Token {api_token}'
}

# Fetch the OpenAPI specification
response = requests.get(openapi_url, headers=headers)

if response.status_code == 200:
    # Parse the JSON content
    openapi_spec = response.json()
    
    # Extract paths from the OpenAPI specification
    api_paths = openapi_spec.get("paths", {})
    simplified_paths = {}

    for path in api_paths:
        # Skip paths containing {id} or /notes/
        if "/{id}/" in path or "/notes/" in path:
            continue
        
        # Ensure the path ends with a forward slash
        full_path = f"/api{path.rstrip('/')}/"
        
        # Generate a name based on the last part of the URL path
        path_parts = path.strip('/').split('/')
        if path_parts:
            name = path_parts[-1].replace('-', ' ').capitalize()
        else:
            name = "Unnamed Endpoint"
        
        # Consolidate duplicate paths by using the base path
        base_path = full_path.rstrip('/').split('/{')[0]
        if base_path not in simplified_paths:
            simplified_paths[base_path] = {
                "endpoint": base_path + '/',  # Ensure base path ends with a forward slash
                "Name": name
            }
        else:
            # Avoid duplicate names by appending the last segment
            existing_name = simplified_paths[base_path]["Name"]
            if existing_name.lower() != name.lower():
                simplified_paths[base_path]["Name"] += f" / {name}"
    
    # Convert the dictionary to a list for output
    output_list = list(simplified_paths.values())
    
    # Save the simplified JSON content to a file
    output_file_path = './nautobot_react_agent/nautobot_apis.json'
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    with open(output_file_path, 'w') as json_file:
        json.dump(output_list, json_file, indent=4)
    
    print(f'Simplified API paths saved to {output_file_path}')
else:
    print(f'Failed to fetch OpenAPI specification. Status code: {response.status_code}')
    print(f'Response content: {response.text}')
