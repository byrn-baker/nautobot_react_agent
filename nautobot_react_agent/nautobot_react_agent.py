import os
import json
import logging
import requests
import difflib
import streamlit as st
from langchain_community.llms import Ollama
from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from langchain_core.tools import tool, render_text_description
import urllib3
from dotenv import load_dotenv

# Load ENV VARs
load_dotenv()
NAUTOBOT_URL = os.getenv("NAUTOBOT_URL")
NAUTOBOT_TOKEN = os.getenv("NAUTOBOT_TOKEN")

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
    
# Function to load supported URLs with their names from a JSON file
def load_urls(file_path='nautobot_apis.json'):
    if not os.path.exists(file_path):
        return {"error": f"endpoints file '{file_path}' not found."}
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        return [(entry['endpoint'], entry.get('Name', '')) for entry in data]
    except Exception as e:
        return {"error": f"Error loading endpoints: {str(e)}"}


def check_endpoint_support(api_endpoint: str) -> dict:
    endpoint_list = load_urls()
    if "error" in endpoint_list:
        return endpoint_list  # Return error if loading URLs failed

    urls = [entry[0] for entry in endpoint_list]
    names = [entry[1] for entry in endpoint_list]

    close_endpoint_matches = difflib.get_close_matches(api_endpoint, urls, n=1, cutoff=0.6)
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

# Tools for interacting with nautobot
@tool
def discover_apis(dummy_input: str = None) -> dict:
    """Discover available nautobot APIs from a local JSON file."""
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
    """Check if an API URL or Name is supported by nautobot."""
    result = check_endpoint_support(api_endpoint)
    if result.get('status') == 'supported':
        closest_endpoint = result['closest_endpoint']
        closest_name = result['closest_name']
        return {
            "status": "supported",
            "message": f"The closest supported API URL is '{closest_endpoint}' ({closest_name}).",
            "action": {
                "next_tool": "get_nautobot_data_tool",
                "input": closest_endpoint
            }
        }
    logging.info(f"check_supported_endpoint_tool result: {result}")
    return result


@tool
def get_nautobot_data_tool(api_endpoint: str) -> dict:
    """Fetch data from nautobot."""
    try:
        nautobot_controller = NautobotController(
            nautobot_url=os.getenv("NAUTOBOT_URL"),
            api_token=os.getenv("NAUTOBOT_TOKEN")
        )
        data = nautobot_controller.get_api(api_endpoint)
        return data
    except requests.HTTPError as e:
        return {"error": f"Failed to fetch data from nautobot: {str(e)}"}
    except Exception as e:
        return {"error": f"An unexpected error occurred: {str(e)}"}


def process_agent_response(response):
    logging.info(f"Agent response: {response}")
    if response and response.get("status") == "supported" and "next_tool" in response.get("action", {}):
        next_tool = response["action"]["next_tool"]
        tool_input = response["action"]["input"]

        # Sanitize the tool input to ensure it's a clean string
        tool_input = str(tool_input).strip()

        # Automatically invoke the next tool
        return agent_executor.invoke({
            "input": tool_input,
            "chat_history": st.session_state.chat_history,
            "agent_scratchpad": "",
            "tool": next_tool
        })
    else:
        return response



# ============================================================
# Streamlit App
# ============================================================

def configure_page():
    st.title("Nautobot Configuration")
    base_url = st.text_input("Nautobot URL", placeholder="https://demo.netbox.com")
    api_token = st.text_input("Nautobot API Token", type="password", placeholder="Your API Token")

    if st.button("Save and Continue"):
        if not base_url or not api_token:
            st.error("All fields are required.")
        else:
            st.session_state['NAUTOBOT_URL'] = base_url
            st.session_state['NAUTOBOT_TOKEN'] = api_token
            os.environ['NAUTOBOT_URL'] = base_url
            os.environ['NAUTOBOT_TOKEN'] = api_token
            st.success("Configuration saved! Redirecting to chat...")
            st.session_state['page'] = "chat"

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
        Assistant is a Nautobot network assistant. It answers questions by using tools to fetch live data from Nautobot. It cannot solve tasks without using these tools.

        TOOLS:
        - discover_apis: Discover available Nautobot API endpoints from a local JSON file.
        - check_supported_endpoint_tool: Check if an API URL or Name is supported by Nautobot.
        - get_nautobot_data_tool: Fetch data from Nautobot using a specified API endpoint.

        TASK:
        1. Understand the user's question and extract relevant keywords (e.g., "devices").
        2. Use tools to validate and fetch data:
        - Always use `check_supported_endpoint_tool` to validate an API endpoint.
        - Always use `get_nautobot_data_tool` to fetch data from a validated endpoint.
        3. Process the data to answer the user's question. For example:
        - To count devices, count the entries in the `results` key of the JSON response.
        4. Provide a clear, concise answer based on the processed data.

        IMPORTANT RULES:
        - Do not fabricate answers. Use only the data from the tools.
        - If no results are found, inform the user: "There are no devices in your Nautobot instance."

        FORMAT:
        Thought: [Your thought process]
        Action: [Tool Name]
        Action Input: [Tool Input]
        Observation: [Tool Response]
        Final Answer: [Your response based on the observation]

        EXAMPLE:
        User Input: "How many devices do I have?"
        Thought: I recognize the keyword "devices" and need to validate the API endpoint.
        Action: check_supported_endpoint_tool
        Action Input: "/api/dcim/devices/"
        Observation: The endpoint is valid and corresponds to the keyword "devices."
        Thought: I need to fetch the actual data from Nautobot.
        Action: get_nautobot_data_tool
        Action Input: "/api/dcim/devices/"
        Observation: The API response contains a `results` key with 10 entries.
        Final Answer: "You have 10 devices in your Nautobot instance."

        Begin:
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

                # Extract the final answer
                if "results" in final_response and isinstance(final_response["results"], list):
                    total_results = len(final_response["results"])
                    if total_results > 100:
                        st.write(f"**Note:** Too many results ({total_results}). Showing the first 100 entries.")
                        st.write(final_response["results"][:100])
                    else:
                        st.write(final_response["results"])
                else:
                    final_answer = final_response.get('output', 'No answer provided.')
                    st.write(f"**Question:** {user_input}")
                    st.write(f"**Answer:** {final_answer}")

                # Display the question and answer
                st.write(f"**Question:** {user_input}")
                st.write(f"**Answer:** {final_answer}")

                # Update chat history with the new conversation
                st.session_state.chat_history += f"User: {user_input}\nAssistant: {final_answer}\n"

            except Exception as e:
                logging.error(f"Error during agent invocation: {str(e)}")
                st.error(f"An error occurred: {str(e)}. Please check your inputs or server status.")


# Page Navigation
if 'page' not in st.session_state:
    st.session_state['page'] = "configure"

if st.session_state['page'] == "configure":
    configure_page()
elif st.session_state['page'] == "chat":
    chat_page()