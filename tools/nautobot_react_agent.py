import os
import json
import logging
import requests
import difflib
import streamlit as st
from langchain_ollama import OllamaLLM
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

# Load supported URLs from JSON
def load_urls(file_path='nautobot_apis.json'):
    if not os.path.exists(file_path):
        return {"error": f"Endpoints file '{file_path}' not found."}
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        return [(entry['endpoint'], entry.get('Name', '')) for entry in data]
    except Exception as e:
        return {"error": f"Error loading endpoints: {str(e)}"}

def check_endpoint_support(api_endpoint: str) -> dict:
    endpoint_list = load_urls()
    if "error" in endpoint_list:
        return endpoint_list

    urls = [entry[0] for entry in endpoint_list]
    names = [entry[1] for entry in endpoint_list]

    close_endpoint_matches = difflib.get_close_matches(api_endpoint, urls, n=1, cutoff=0.6)
    close_name_matches = difflib.get_close_matches(api_endpoint, names, n=1, cutoff=0.6)

    if close_endpoint_matches:
        closest_endpoint = close_endpoint_matches[0]
        matching_name = [entry[1] for entry in endpoint_list if entry[0] == closest_endpoint][0]
        return {"status": "supported", "endpoint": closest_endpoint, "name": matching_name}
    elif close_name_matches:
        closest_name = close_name_matches[0]
        closest_endpoint = [entry[0] for entry in endpoint_list if entry[1] == closest_name][0]
        return {"status": "supported", "endpoint": closest_endpoint, "name": closest_name}
    else:
        return {"status": "unsupported", "message": f"The input '{api_endpoint}' is not supported."}

# Tools for interacting with Nautobot
@tool
def discover_apis(dummy_input: str = None) -> dict:
    """Discover available Nautobot API endpoints from a local JSON file."""
    try:
        with open("nautobot_apis.json", "r") as f:
            data = json.load(f)
        return {"apis": data, "message": "APIs successfully loaded from JSON file"}
    except Exception as e:
        return {"error": f"An error occurred while loading the APIs: {str(e)}"}

@tool
def check_supported_endpoint_tool(api_endpoint: str) -> dict:
    """Check if an API URL or Name is supported by Nautobot."""
    result = check_endpoint_support(api_endpoint)
    logging.info(f"check_supported_endpoint_tool result: {result}")
    return result

@tool
def get_nautobot_data_tool(api_endpoint: str) -> dict:
    """Fetch data from Nautobot using a specified API endpoint."""
    try:
        nautobot_controller = NautobotController(NAUTOBOT_URL, NAUTOBOT_TOKEN)
        data = nautobot_controller.get_api(api_endpoint)
        return data
    except requests.HTTPError as e:
        return {"error": f"Failed to fetch data from Nautobot: {str(e)}"}
    except Exception as e:
        return {"error": f"An unexpected error occurred: {str(e)}"}

# Streamlit App
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
            st.success("Configuration saved! Redirecting to chat...")
            st.session_state['page'] = "chat"

def initialize_agent():
    global agent_executor
    if not agent_executor:
        llm = OllamaLLM(
            model="llama3.3",
            base_url="http://192.168.100.32:11434"
        )
        tools = [discover_apis, check_supported_endpoint_tool, get_nautobot_data_tool]
        tool_descriptions = render_text_description(tools)
        tool_names = ", ".join([tool.name for tool in tools])

        template = """
        You are a Nautobot assistant that answers questions by querying live data from Nautobot using tools. You cannot answer without using tools.

        Available Tools:
        {tools}
        Tool Names: {tool_names}

        Instructions:
        1. For every query, start by using `discover_apis` to list all available Nautobot endpoints from `nautobot_apis.json`.
        2. Analyze the user's question to identify key entities (e.g., device name 'ams01-asw-01', 'primary IPv4 address').
        3. From the `discover_apis` output, select the most relevant endpoint that matches the query (e.g., '/api/dcim/devices/' for device details).
        4. Use `check_supported_endpoint_tool` to validate the selected endpoint.
        5. If supported, use `get_nautobot_data_tool` with appropriate filters (e.g., '?name=ams01-asw-01&depth=1&exclude_m2m=false') to fetch data.
        6. Process the response to extract the requested information (e.g., 'primary_ip4.address').
        7. If no data is found, say: "No results found in your Nautobot instance."
        8. If an error occurs, report it clearly.
        9. Always follow this sequence: discover endpoints, validate, fetch data, then answer.

        Response Format for Each Step:
        ```json
        {{
          "thought": "Your reasoning here",
          "action": "tool_name",
          "input": "tool_input",
          "observation": "tool_output",
          "final_answer": "Your answer here (only in the last step)"
        }}
        ```

        Chat History:
        {chat_history}

        User Input:
        {input}

        Agent Scratchpad:
        {agent_scratchpad}
        """

        prompt_template = PromptTemplate(
            template=template,
            input_variables=["input", "chat_history", "agent_scratchpad"],
            partial_variables={
                "tools": tool_descriptions,
                "tool_names": tool_names
            }
        )

        agent = create_react_agent(llm=llm, tools=tools, prompt=prompt_template)
        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            handle_parsing_errors=True,
            verbose=True,
            max_iterations=15
        )

def chat_page():
    st.title("Chat with Nautobot AI Agent")
    user_input = st.text_input("Ask Nautobot a question:", key="user_input")

    if "NAUTOBOT_TOKEN" not in st.session_state:
        st.error("Please configure Nautobot settings first!")
        st.session_state['page'] = "configure"
        return

    initialize_agent()

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if "conversation" not in st.session_state:
        st.session_state.conversation = []

    if st.button("Send") and user_input:
        st.session_state.conversation.append({"role": "user", "content": user_input})
        try:
            chat_history_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state.chat_history])

            response = agent_executor.invoke({
                "input": user_input,
                "chat_history": chat_history_str,
                "agent_scratchpad": ""
            })

            if isinstance(response, dict) and "output" in response:
                output = response["output"]
                try:
                    parsed_output = json.loads(output) if isinstance(output, str) else output
                    final_answer = parsed_output.get("final_answer", "No answer provided.")
                    observation = parsed_output.get("observation", {})
                except json.JSONDecodeError:
                    final_answer = output
                    observation = {}
            else:
                final_answer = str(response)
                observation = {}

            st.write(f"**Question:** {user_input}")
            st.write(f"**Answer:** {final_answer}")

            if isinstance(observation, dict) and "results" in observation:
                results = observation["results"]
                total_results = len(results)
                if total_results > 100:
                    st.write(f"**Note:** Too many results ({total_results}). Showing first 100.")
                    st.write(results[:100])
                elif total_results > 0:
                    st.write(results)
                else:
                    st.write("No results found.")

            st.session_state.chat_history.append({"role": "user", "content": user_input})
            st.session_state.chat_history.append({"role": "assistant", "content": final_answer})

        except Exception as e:
            logging.error(f"Error during agent invocation: {str(e)}")
            st.error(f"An error occurred: {str(e)}. Ensure Nautobot URL and token are correct.")

# Page Navigation
if 'page' not in st.session_state:
    st.session_state['page'] = "configure"

if st.session_state['page'] == "configure":
    configure_page()
elif st.session_state['page'] == "chat":
    chat_page()