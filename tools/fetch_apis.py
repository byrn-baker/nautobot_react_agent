import requests
import json

# URL to the OpenAPI specification
openapi_url = 'https://demo.nautobot.com/api/docs/?format=openapi'

# Your API token (replace 'your_api_token' with the actual token)
api_token = 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa'

# HTTP headers for authentication
headers = {
    'Authorization': f'Token {api_token}'
}

# Fetch the OpenAPI specification
response = requests.get(openapi_url, headers=headers)

if response.status_code == 200:
    # Parse the JSON content
    openapi_spec = response.json()
    
    # Extract paths and format them
    api_paths = openapi_spec.get("paths", {})
    simplified_paths = {}

    for path in api_paths:
        # Skip paths containing {id} or /notes/
        if "/{id}/" in path or "/notes/" in path:
            continue
        
        # Ensure the path ends with a forward slash and prepend /api/
        full_path = f"/api{path.rstrip('/')}/"
        
        # Generate a human-readable name
        name_parts = path.strip('/').split('/')
        name = ' '.join(part.capitalize() for part in name_parts[-3:])
        
        # Consolidate duplicate paths
        base_path = full_path.rstrip('/').split('/{')[0]
        if base_path not in simplified_paths:
            simplified_paths[base_path] = {
                "URL": base_path + '/',  # Ensure the base path also ends with a forward slash
                "Name": name
            }
    
    # Convert the dictionary to a list
    output_list = list(simplified_paths.values())
    
    # Save the simplified JSON content to a file
    with open('./nautobot_react_agent/nautobot_apis.json', 'w') as json_file:
        json.dump(output_list, json_file, indent=4)
    
    print('Simplified API paths saved to simplified_api_paths.json')
else:
    print(f'Failed to fetch OpenAPI specification. Status code: {response.status_code}')
    print(f'Response content: {response.text}')