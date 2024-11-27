import os
import yaml
import logging
import requests
import difflib
import streamlit as st
from langchain_community.llms import Ollama
from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from langchain_core.tools import tool, render_text_description
import urllib3

# Configure logging
logging.basicConfig(level=logging.INFO)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Global variables for lazy initialization
llm = None
agent_executor = None

# NautobotController for CRUD Operations
class NautobotController:
    def __init__(self, nautobot_url, api_token):
        self.nautobot = nautobot_url.rstrip('/')
        self.api_token = api_token
        self.headers = {
            'Accept': 'application/json',
            'Authorization': f"Token {self.api_token}",
        }

    def get_api(self, api_endpoint: str, params: dict = None):
        response = requests.get(
            f"{self.nautobot}{api_endpoint}",
            headers=self.headers,
            params=params,
            verify=False
        )
        response.raise_for_status()
        return response.json()

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


# Function to load supported endpoints with their names from a YAML file
def load_endpoints(file_path='nautobot_apis.yaml'):
    if not os.path.exists(file_path):
        return {"error": f"Endpoints file '{file_path}' not found."}
    try:
        with open(file_path, 'r') as f:
            data = yaml.safe_load(f)
        return [(entry['endpoint'], entry.get('Name', '')) for entry in data]
    except Exception as e:
        return {"error": f"Error loading endpoints: {str(e)}"}


def check_endpoint_support(api_endpoint: str) -> dict:
    endpoint_list = load_endpoints()
    if "error" in endpoint_list:
        return endpoint_list  # Return error if loading endpoints failed

    endpoints = [entry[0] for entry in endpoint_list]
    names = [entry[1] for entry in endpoint_list]

    close_endpoint_matches = difflib.get_close_matches(api_endpoint, endpoints, n=1, cutoff=0.6)
    close_name_matches = difflib.get_close_matches(api_endpoint, names, n=1, cutoff=0.6)

    if close_endpoint_matches:
        closest_endpoint = close_endpoint_matches[0]
        matching_name = [entry[1] for entry in endpoint_list if entry[0] == closest_endpoint][0]
        return {"status": "supported", "closest_endpoint": closest_endpoint, "closest_name": matching_name}
    elif close_name_matches:
        closest_name = close_name_matches[0]
        closest_endpoint = [entry[0] for entry in endpoint_list if entry[1] == closest_name][0]
        return {"status": "supported", "closest_endpoint": closest_endpoint, "closest_name": closest_name}
    else:
        return {"status": "unsupported", "message": f"The input '{api_endpoint}' is not supported."}


# Tools for interacting with Nautobot
@tool
def discover_apis(dummy_input: str = None) -> dict:
    """Discover available Nautobot APIs from a local YAML file."""
    try:
        if not os.path.exists("nautobot_apis.yaml"):
            return {"error": "API YAML file not found. Please ensure 'nautobot_apis.yaml' exists in the project directory."}
        
        with open("nautobot_apis.yaml", "r") as f:
            data = yaml.safe_load(f)
        return {"apis": data, "message": "APIs successfully loaded from YAML file"}
    except Exception as e:
        return {"error": f"An error occurred while loading the APIs: {str(e)}"}


@tool
def check_supported_endpoint_tool(api_endpoint: str) -> dict:
    """Check if an API endpoint or Name is supported by Nautobot."""
    result = check_endpoint_support(api_endpoint)
    if result.get('status') == 'supported':
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
    return result


@tool
def get_nautobot_data_tool(api_endpoint: str) -> dict:
    """Fetch data from Nautobot."""
    try:
        nautobot_controller = NautobotController(
            nautobot_url=os.getenv("NAUTOBOT_URL"),
            api_token=os.getenv("NAUTOBOT_TOKEN")
        )
        data = nautobot_controller.get_api(api_endpoint)
        
        # Convert the data to YAML
        yaml_output = yaml.dump(data, default_flow_style=False)
        
        return {
            "status": "supported",
            "data": yaml_output,
            "message": f"Successfully fetched and converted data from endpoint: {api_endpoint} to YAML format.",
            "action": {
                # No next tool; just provide the data.
            }
        }
    except requests.HTTPError as e:
        return {"error": f"Failed to fetch data from Nautobot: {str(e)}"}
    except Exception as e:
        return {"error": f"An unexpected error occurred: {str(e)}"}


@tool
def create_nautobot_data_tool(input: str) -> dict:
    """Create new data in Nautobot."""
    try:
        data = yaml.safe_load(input)
        api_endpoint = data.get("api_endpoint")
        payload = data.get("payload")

        if not api_endpoint or not payload:
            raise ValueError("Both 'api_endpoint' and 'payload' must be provided.")

        if not isinstance(payload, dict):
            raise ValueError("Payload must be a dictionary.")

        nautobot_controller = NautobotController(
            nautobot_url=os.getenv("NAUTOBOT_URL"),
            api_token=os.getenv("NAUTOBOT_TOKEN")
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
            api_token=os.getenv("NAUTOBOT_TOKEN")
        )
        return nautobot_controller.delete_api(api_endpoint)
    except requests.HTTPError as e:
        return {"error": f"Failed to delete data from Nautobot: {str(e)}"}
    except Exception as e:
        return {"error": f"An unexpected error occurred: {str(e)}"}


def process_agent_response(response):
    if response and response.get("status") == "supported":
        # If there's YAML data in the response, convert it back to Python object if needed
        if "data" in response:
            try:
                # Convert YAML to dictionary to work with it
                data_dict = yaml.safe_load(response["data"])
                # You can perform further processing on data_dict if required
                # Example: Just display the YAML as final output
                return {"output": response["data"]}
            except yaml.YAMLError as e:
                return {"error": f"Failed to parse YAML data: {str(e)}"}

        if "next_tool" in response.get("action", {}):
            next_tool = response["action"]["next_tool"]
            tool_input = response["action"]["input"]

            # Automatically invoke the next tool
            next_response = agent_executor.invoke({
                "input": tool_input,
                "chat_history": st.session_state.chat_history,
                "agent_scratchpad": "",
                "tool": next_tool
            })
            return process_agent_response(next_response)
    
    # Return the current response if no further tool action is required
    return response


# ============================================================
# Streamlit App
# ============================================================

def configure_page():
    st.title("Nautobot Configuration")
    base_url = st.text_input("Nautobot URL", placeholder="https://demo.nautobot.com")
    api_token = st.text_input("Nautobot API Token", type="password", placeholder="Your API Token")

    if st.button("Save and Continue"):
        if not base_url or not api_token:
            st.error("All fields are required.")
        else:
            st.session_state['NAUTOBOT_URL'] = base_url
            st.session_state['NAUTOBOT_TOKEN'] = api_token
            os.environ['NAUTOBOT_URL'] = base_url
            os.environ['NAUTOBOT_TOKEN'] = api_token
            
            st.session_state['redirect_to_chat'] = True
            st.success("Configuration saved! Redirecting to chat...")

def initialize_agent():
    global llm, agent_executor
    if not llm:
        # Initialize the LLM with the API key from session state
        llm = Ollama(model="llama3.2", base_url="http://ollama:11434")

        # Define tools
        tools = [discover_apis, check_supported_endpoint_tool, get_nautobot_data_tool]

        # Create the prompt template
        tool_descriptions = render_text_description(tools)
        # Create the PromptTemplate
        template = """
        Assistant is a network assistant capable of managing Nautobot data using CRUD operations.

        TOOLS:
        - discover_apis: Discovers available Nautobot APIs from a local JSON file.
        - check_supported_endpoint_tool: Checks if an API URL or Name is supported by Nautobot.
        - get_nautobot_data_tool: Fetches data from Nautobot using the specified API URL and provides the response in YAML format.

        GUIDELINES:
        1. Use 'check_supported_url_tool' to validate ambiguous or unknown URLs or Names.
        2. If certain about the URL, directly use 'get_nautobot_data_tool', 'create_nautobot_data_tool', or 'delete_nautobot_data_tool'.
        3. Follow a structured response format to ensure consistency.
        4. When reading the output from 'get_nautobot_data_tool', it will be in YAML format.

        FORMAT:
        Thought: [Your thought process]
        Action: [Tool Name]
        Action Input: [Tool Input]
        Observation: [Tool Response in YAML format]
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
            max_iterations=10
        )


def chat_page():
    st.title("Chat with Nautobot AI Agent")
    user_input = st.text_input("Ask Nautobot a question:", key="user_input")

    # Ensure the agent is initialized
    if "NAUTOBOT_TOKEN" not in st.session_state:
        st.error("Please configure Nautobot settings first!")
        st.session_state['page'] = "configure"
        return

    initialize_agent()

    # Initialize session state variables if not already set
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = ""

    if "conversation" not in st.session_state:
        st.session_state.conversation = []

    # Button to submit the question
    if st.button("Send"):
        if user_input:
            # Add the user input to the conversation history
            st.session_state.conversation.append({"role": "user", "content": user_input})

            # Invoke the agent with the user input and current chat history
            try:
                response = agent_executor.invoke({
                    "input": user_input,
                    "chat_history": st.session_state.chat_history,
                    "agent_scratchpad": ""  # Initialize agent scratchpad as an empty string
                })

                # Process the agent's response
                final_response = process_agent_response(response)

                # Handle the response appropriately
                if isinstance(final_response, dict):
                    if "message" in final_response:
                        final_answer = final_response.get("message")
                    else:
                        # If response includes data from Nautobot, display it directly
                        final_answer = final_response.get('output', str(final_response))
                else:
                    final_answer = str(final_response)

                # Display the question and answer
                st.write(f"**Question:** {user_input}")
                st.write(f"**Answer:** {final_answer}")

                # Add the response to the conversation history
                st.session_state.conversation.append({"role": "assistant", "content": final_answer})

                # Truncate conversation history to avoid excessive length
                max_conversation_turns = 5
                if len(st.session_state.conversation) > max_conversation_turns:
                    st.session_state.conversation = st.session_state.conversation[-max_conversation_turns:]

                # Update chat history with the new conversation
                st.session_state.chat_history = "\n".join(
                    [f"{entry['role'].capitalize()}: {entry['content']}" for entry in st.session_state.conversation]
                )
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

    # Display conversation history
    if st.session_state.conversation:
        st.markdown("### Conversation History")
        for entry in st.session_state.conversation:
            if entry["role"] == "user":
                st.markdown(f"**User:** {entry['content']}")
            elif entry["role"] == "assistant":
                st.markdown(f"**Nautobot AI ReAct Agent:** {entry['content']}")



# ============================================================
# Main Navigation Logic
# ============================================================

# Page Navigation - Ensure 'page' is initialized in session state
if 'page' not in st.session_state:
    st.session_state['page'] = "configure"

# Check if user wants to be redirected to chat
if st.session_state.get('redirect_to_chat'):
    # Redirect to chat page
    st.session_state['page'] = "chat"
    # Remove the redirect flag so it doesn't keep redirecting on every rerun
    del st.session_state['redirect_to_chat']

# Show appropriate page based on session state
if st.session_state['page'] == "configure":
    configure_page()
elif st.session_state['page'] == "chat":
    chat_page()