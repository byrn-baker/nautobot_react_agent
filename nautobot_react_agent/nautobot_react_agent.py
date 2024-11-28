import os
import time
import json
import logging
import requests
import difflib
import tiktoken
import streamlit as st
from langchain_community.chat_models import ChatOpenAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from langchain_core.tools import tool, render_text_description
import urllib3
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Global variables for lazy initialization
llm = None
agent_executor = None

# Attempting to limit the responses so chatgpt can handle larger responses
def filter_fields(data, keys_to_keep):
    """
    Filter the fields of a list of dictionaries.
    Args:
        data (list): List of dictionaries to filter.
        keys_to_keep (list): Keys to retain in each dictionary.
    Returns:
        list: Filtered list of dictionaries.
    """
    return [{key: obj[key] for key in keys_to_keep if key in obj} for obj in data]

def chunk_large_data(data, chunk_size):
    """
    Split a large list into smaller chunks.
    Args:
        data (list): The large dataset to chunk.
        chunk_size (int): The maximum size of each chunk.
    Returns:
        list: A list of smaller chunks.
    """
    return [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]

def summarize_large_data(data, max_items=50):
    """
    Summarize large data by selecting only a subset of records.
    Args:
        data (list): List of dictionaries.
        max_items (int): Maximum number of items to include in the summary.
    Returns:
        list: Summarized data.
    """
    return data[:max_items]

def estimate_token_usage(data, model="gpt-4"):
    """
    Estimate the number of tokens used by the given data.
    Args:
        data (list or dict): Data to estimate token usage for.
        model (str): The OpenAI model name.
    Returns:
        int: Estimated token count.
    """
    encoder = tiktoken.encoding_for_model(model)
    return len(encoder.encode(str(data)))

import time

def openai_api_call(data, retries=3):
    """
    Mock OpenAI API call with retry logic for 429 errors.
    Replace this with actual OpenAI API call logic.
    Args:
        data (list or dict): Data to send to OpenAI.
        retries (int): Number of retries on failure.
    Returns:
        dict: Simulated API response.
    """
    for attempt in range(retries):
        try:
            # Mock API Call
            token_count = estimate_token_usage(data)
            if token_count > 30000:
                raise ValueError(f"Data exceeds token limit: {token_count} tokens")

            # Simulate successful API response
            return {"response": f"Processed {len(data)} items"}
        except ValueError as ve:
            raise ve
        except Exception as e:
            if "429" in str(e):  # Handle rate limit error
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                    continue
                else:
                    raise e
            else:
                raise e

def process_large_data_for_openai(data, chunk_size=100, summarize=True, max_summary_items=50):
    """
    Prepare large data for OpenAI by summarizing or chunking.
    Args:
        data (list): The data to process.
        chunk_size (int): The size of each chunk.
        summarize (bool): Whether to summarize the data.
        max_summary_items (int): Max items to include in the summary.
    Returns:
        list: Processed chunks ready for OpenAI.
    """
    # Summarize if necessary
    if summarize:
        data = summarize_large_data(data, max_items=max_summary_items)
    
    # Chunk the data
    chunks = chunk_large_data(data, chunk_size)
    responses = []
    
    for chunk in chunks:
        # Ensure chunk is within token limits
        token_count = estimate_token_usage(chunk)
        if token_count > 30000:  # gpt-4 token limit
            raise ValueError(f"Chunk exceeds token limit: {token_count} tokens")
        
        # Call OpenAI API for each chunk
        response = openai_api_call(chunk)
        responses.append(response)
    
    return responses

  
# NautobotController for CRUD Operations
class NautobotController:
    def __init__(self, nautobot_url, api_token):
        self.nautobot = nautobot_url.rstrip('/')
        self.api_token = api_token
        self.headers = {
            'Accept': 'application/json',
            'Authorization': f"Token {self.api_token}",
        }

    def get_api(self, api_endpoint: str, params: dict = None, keys_to_keep: list = None):
        response = requests.get(
        f"{self.nautobot}{api_endpoint}",
        headers=self.headers,
        params=params,
        verify=False
    )
        response.raise_for_status()
        data = response.json()

        # If keys_to_keep is provided, filter the fields
        if keys_to_keep:
            data["results"] = filter_fields(data.get("results", []), keys_to_keep)
        
        return data

    def post_api(self, api_endpoint: str, payload: dict):
        response = requests.post(
            f"{self.nautobot}{api_endpoint}",
            headers=self.headers,
            json=payload,
            verify=False
        )
        response.raise_for_status()
        return response.json()

    def delete_api(self, api_endpoint: str):
        response = requests.delete(
            f"{self.nautobot}{api_endpoint}",
            headers=self.headers,
            verify=False
        )
        response.raise_for_status()
        return response.json()


# Function to load supported endpoints with their names from a JSON file
def load_endpoints(file_path='nautobot_apis.json'):
    if not os.path.exists(file_path):
        return {"error": f"endpoints file '{file_path}' not found."}
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        return [(entry['endpoint'], entry.get('Name', '')) for entry in data]
    except Exception as e:
        return {"error": f"Error loading endpoints: {str(e)}"}


def check_endpoint_support(api_endpoint: str) -> dict:
    endpoint_list = load_endpoints()
    if "error" in endpoint_list:
        return endpoint_list  # Return error if loading endpoints failed

    endpoints = [entry[0] for entry in endpoint_list]
    names = [entry[1] for entry in endpoint_list]

    # Get close matches
    close_endpoint_matches = difflib.get_close_matches(api_endpoint, endpoints, n=5, cutoff=0.6)
    close_name_matches = difflib.get_close_matches(api_endpoint, names, n=5, cutoff=0.6)

    matches = []

    # Add matching endpoints
    for endpoint in close_endpoint_matches:
        name = next(entry[1] for entry in endpoint_list if entry[0] == endpoint)
        matches.append({"endpoint": endpoint, "name": name, "match_type": "endpoint"})

    # Add matching names
    for name in close_name_matches:
        endpoint = next(entry[0] for entry in endpoint_list if entry[1] == name)
        if endpoint not in [m["endpoint"] for m in matches]:
            matches.append({"endpoint": endpoint, "name": name, "match_type": "name"})

    if len(matches) == 1:
        return {"status": "supported", "closest_endpoint": matches[0]["endpoint"], "closest_name": matches[0]["name"]}
    elif len(matches) > 1:
        # Return multiple matches for disambiguation
        return {"status": "ambiguous", "matches": matches, "message": "Multiple matches found. Please refine your input."}
    else:
        return {"status": "unsupported", "message": f"The input '{api_endpoint}' is not supported."}


# Tools for interacting with Nautobot
@tool
def discover_apis(dummy_input: str = None) -> dict:
    """Discover available Nautobot APIs from a local JSON file."""
    try:
        if not os.path.exists("nautobot_apis.json"):
            return {"error": "API JSON file not found. Please ensure 'nautobot_apis.json' exists in the project directory."}
        
        with open("nautobot_apis.json", "r") as f:
            data = json.load(f)
        return {"apis": data, "message": "APIs successfully loaded from JSON file"}
    except Exception as e:
        return {"error": f"An error occurred while loading the APIs: {str(e)}"}

@tool
def check_supported_endpoint_tool(api_endpoint: str) -> dict:
    """Check if an API endpoint or Name is supported by Nautobot."""
    result = check_endpoint_support(api_endpoint)
    if result.get("status") == "supported":
        closest_endpoint = result['closest_endpoint']
        closest_name = result['closest_name']
        return {
            "status": "supported",
            "message": f"The closest supported API endpoint is '{closest_endpoint}' ({closest_name}).",
            "action": {
                "next_tool": "get_nautobot_data_tool",
                "input": closest_endpoint
            }
        }
    elif result.get("status") == "ambiguous":
        # Return multiple matches for user clarification
        matches = result["matches"]
        match_list = "\n".join([f"- {m['name']} ({m['endpoint']})" for m in matches])
        return {
            "status": "ambiguous",
            "message": f"Multiple matches found for '{api_endpoint}':\n{match_list}\nPlease refine your query."
        }
    return result

@tool
def get_nautobot_data_tool(api_endpoint: str, chunk_size: int = 100) -> dict:
    """Fetch data from Nautobot, chunk it, and prepare for OpenAI."""
    try:
        nautobot_controller = NautobotController(
            nautobot_url=os.getenv("NAUTOBOT_URL"),
            api_token=os.getenv("DEMO_NAUTOBOT_API")
        )
        # Fetch data from the API
        api_response = nautobot_controller.get_api(api_endpoint)

        # Extract results and count
        results = api_response.get("results", [])
        total_count = api_response.get("count", len(results))

        # Process large data
        processed_responses = process_large_data_for_openai(results, chunk_size=chunk_size)

        return {"count": total_count, "responses": processed_responses}
    except requests.HTTPError as e:
        return {"error": f"Failed to fetch data from Nautobot: {str(e)}"}
    except Exception as e:
        return {"error": f"An unexpected error occurred: {str(e)}"}


@tool
def create_nautobot_data_tool(input: str) -> dict:
    """Create new data in Nautobot."""
    try:
        data = json.loads(input)
        api_endpoint = data.get("api_endpoint")
        payload = data.get("payload")

        if not api_endpoint or not payload:
            raise ValueError("Both 'api_endpoint' and 'payload' must be provided.")

        if not isinstance(payload, dict):
            raise ValueError("Payload must be a dictionary.")

        nautobot_controller = NautobotController(
            nautobot_url=os.getenv("NAUTOBOT_URL"),
            api_token=os.getenv("DEMO_NAUTOBOT_API")
        )
        return nautobot_controller.post_api(api_endpoint, payload)
    except Exception as e:
        return {"error": f"An error occurred in create_nautobot_data_tool: {str(e)}"}

@tool
def delete_nautobot_data_tool(api_endpoint: str) -> dict:
    """Delete data from Nautobot."""
    try:
        nautobot_controller = NautobotController(
            nautobot_url=os.getenv("NAUTOBOT_URL"),
            api_token=os.getenv("DEMO_NAUTOBOT_API")
        )
        return nautobot_controller.delete_api(api_endpoint)
    except requests.HTTPError as e:
        return {"error": f"Failed to delete data from Nautobot: {str(e)}"}
    except Exception as e:
        return {"error": f"An unexpected error occurred: {str(e)}"}

def process_agent_response(response):
    if response.get("status") == "ambiguous":
        st.warning(response["message"])
        return {"status": "ambiguous", "message": "Clarification requested from user."}

    if response.get("status") == "supported" and "next_tool" in response.get("action", {}):
        next_tool = response["action"]["next_tool"]
        tool_input = response["action"]["input"]

        # Automatically invoke the next tool
        result = agent_executor.invoke({
            "input": tool_input,
            "chat_history": st.session_state.chat_history,
            "agent_scratchpad": "",
            "tool": next_tool
        })

        # Handle chunked data in the result
        if "chunks" in result:
            for chunk in result["chunks"]:
                # Optionally process each chunk separately (e.g., summarize or send to OpenAI)
                st.write(f"Processing chunk: {chunk}")
        return result
    else:
        return response



# ============================================================
# Streamlit App
# ============================================================

def configure_page():
    st.title("Configure API Settings")
    
    # Pre-fill with environment variables if available
    base_url = os.getenv("NAUTOBOT_URL", "")
    openai_key = os.getenv("OPENAI_API", "")
    api_token = os.getenv("DEMO_NAUTOBOT_API", "")

    # Allow user to input or override tokens
    nautobot_url_input = st.text_input("Nautobot URL", value=base_url, placeholder="https://demo.nautobot.com")
    openai_token_input = st.text_input("OpenAI API Token", value=openai_key, type="password")
    nautobot_token_input = st.text_input("Nautobot API Token", value=api_token, type="password")

    # Button to save configuration
    if st.button("Save and Continue"):
        if not nautobot_url_input or not openai_token_input or not nautobot_token_input:
            st.error("All fields are required.")
        else:
            # Save to session state and environment
            st.session_state['NAUTOBOT_URL'] = base_url
            st.session_state['DEMO_NAUTOBOT_API'] = api_token
            st.session_state['OPENAI_API_KEY'] = openai_key
            os.environ['NAUTOBOT_URL'] = base_url
            os.environ['DEMO_NAUTOBOT_API'] = api_token
            os.environ['OPENAI_API_KEY'] = openai_key
            st.success("Configuration saved! Redirecting to chat...")
            st.session_state['page'] = "chat"

    # Allow skipping if environment variables are already configured
    if base_url and openai_key and api_token and st.button("Use Existing Configuration"):
        st.session_state['page'] = "chat"
        st.success("Using existing configuration from environment variables.")



            
def initialize_agent():
    global llm, agent_executor
    if not llm:
        # Initialize the LLM with the API key from session state
        llm = ChatOpenAI(model_name="gpt-4o", openai_api_key=st.session_state['OPENAI_API_KEY'])

        # Define tools
        tools = [discover_apis, check_supported_endpoint_tool, get_nautobot_data_tool, create_nautobot_data_tool, delete_nautobot_data_tool]

        # Create the prompt template
        tool_descriptions = render_text_description(tools)
        # Create the PromptTemplate
        template = """
        Assistant is a network assistant capable of managing Nautobot data using CRUD operations.

        TOOLS:
        - discover_apis: Discovers available Nautobot APIs from a local JSON file.
        - check_supported_endpoint_tool: Checks if an API endpoint or Name is supported by Nautobot.
        - get_nautobot_data_tool: Fetches data from Nautobot using the specified API endpoint.
        - create_nautobot_data_tool: Creates new data in Nautobot using the specified API endpoint and payload.
        - delete_nautobot_data_tool: Deletes data from Nautobot using the specified API endpoint.

        GUIDELINES:
        1. **Matching Endpoint Context**: Use user input context (e.g., "device types" refers to hardware) to select the most relevant endpoint.
        2. **Handling Ambiguity**:
        - If multiple endpoints match, use the most specific match relevant to the context.
        - If ambiguity remains, clearly communicate the options and ask the user for clarification.
        3. **Fallback on Clear Intent**:
        - If the user mentions "device types" explicitly, prioritize `/api/dcim/device-types/`.
        - For terms like "devices," use `/api/dcim/devices/` unless context specifies otherwise.
        4. For questions about totals or quantities (e.g., "How many devices do I have?"):
        - Look for the `count` key in the API response.
        - If `count` exists, provide it as the answer.
        - If `count` is missing, calculate the total based on the length of the `results` array.

        FORMAT:
        Thought: [Your thought process]
        Action: [Tool Name]
        Action Input: [Tool Input]
        Observation: [Tool Response]
        Final Answer: [Your response to the user]

        Begin:

        Previous conversation history:
        {chat_history}

        New input: {input}

        {agent_scratchpad}
        """
        prompt_template = PromptTemplate(
            template=template,
            input_variables=["input", "chat_history", "agent_scratchpad"],
            partial_variables={
                "tools": tool_descriptions,
                "tool_names": ", ".join([t.name for t in tools])
            }
        )

        # Create the ReAct agent
        agent = create_react_agent(llm=llm, tools=tools, prompt=prompt_template)

        # Create the AgentExecutor
        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            handle_parsing_errors=True,
            verbose=True,
        )

def chat_page():
    st.title("Chat with Nautobot AI Agent")
    user_input = st.text_input("Ask Nautobot a question:", key="user_input")

    # Ensure the agent is initialized
    if "OPENAI_API_KEY" not in st.session_state:
        st.error("Please configure Nautobot and OpenAI settings first!")
        st.session_state['page'] = "configure"
        return

    initialize_agent()

    # Initialize session state variables if not already set
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = ""

    if "conversation" not in st.session_state:
        st.session_state.conversation = []

    # Handle user input
    if st.button("Send"):
        if user_input:
            # Add the user input to the conversation history
            st.session_state.conversation.append({"role": "user", "content": user_input})

            try:
                # Invoke the agent to process user input
                response = agent_executor.invoke({
                    "input": user_input,
                    "chat_history": st.session_state.chat_history,
                    "agent_scratchpad": ""
                })

                # Handle large responses if "chunks" are present
                if "chunks" in response:
                    st.write("Processing large response in chunks...")
                    processed_chunks = process_large_data_for_openai(response["chunks"], chunk_size=100, summarize=True)
                    
                    # Display each processed chunk
                    for chunk_response in processed_chunks:
                        st.write(chunk_response)
                else:
                    # Process the normal response
                    final_response = process_agent_response(response)
                    final_answer = final_response.get("output", "No answer provided.")
                    st.write(f"**Answer:** {final_answer}")

                # Update chat history
                st.session_state.chat_history += f"User: {user_input}\n"
                st.session_state.chat_history += f"Assistant: {final_answer}\n"

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

    # Display conversation history
    if st.session_state.conversation:
        st.markdown("### Conversation History")
        for entry in st.session_state.conversation:
            role = entry["role"].capitalize()
            content = entry["content"]
            st.markdown(f"**{role}:** {content}")



# Page Navigation
if 'page' not in st.session_state:
    st.session_state['page'] = "configure"

if st.session_state['page'] == "configure":
    configure_page()
elif st.session_state['page'] == "chat":
    chat_page()