import os
import json
import logging
import requests
import streamlit as st
from langchain_community.agent_toolkits.openapi.spec import reduce_openapi_spec
from langchain_community.chat_models import ChatOpenAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from langchain_core.tools import tool, render_text_description
import urllib3
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.DEBUG)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Global variables for lazy initialization
llm = None
agent_executor = None

@tool
def execute_graphql(query: str, variables: dict = None) -> dict:
    """Execute a dynamically constructed GraphQL query."""
    nautobot_url = os.getenv("NAUTOBOT_URL")
    api_token = os.getenv("NAUTOBOT_TOKEN")
    headers = {"Authorization": f"Token {api_token}"}
    endpoint = f"{nautobot_url.rstrip('/')}/api/graphql/"

    # Log the raw input exactly as received
    logging.debug(f"Raw query input: {repr(query)}")

    # Ensure no double-escaping by replacing any erroneous escapes (only if needed)
    cleaned_query = query.replace('\\"', '"').strip()
    logging.debug(f"Cleaned query: {repr(cleaned_query)}")

    try:
        payload = {"query": cleaned_query, "variables": variables or {}}
        logging.debug(f"GraphQL query payload: {payload}")
        response = requests.post(endpoint, headers=headers, json=payload, verify=False)
        response.raise_for_status()
        return response.json()
    except requests.HTTPError as e:
        if response.content:
            logging.error(f"GraphQL error response: {response.content.decode()}")
        raise
    except Exception as e:
        logging.error(f"Failed to execute GraphQL query: {str(e)}", exc_info=True)
        return {"error": str(e)}

@tool
def execute_rest_call(input_str: str) -> dict:
    """Execute a dynamically constructed REST API call."""
    nautobot_url = os.getenv("NAUTOBOT_URL")
    api_token = os.getenv("NAUTOBOT_TOKEN")
    headers = {"Authorization": f"Token {api_token}", "Accept": "application/json"}

    try:
        # Parse the JSON string into a dict
        input_dict = json.loads(input_str)
        endpoint = input_dict["endpoint"]
        method = input_dict.get("method", "GET")
        payload = input_dict.get("payload", None)

        url = f"{nautobot_url.rstrip('/')}{endpoint}"
        logging.debug(f"Executing REST API call: {method} {url} with payload {payload}")
        if method.upper() == "GET":
            response = requests.get(url, headers=headers, params=payload, verify=False)
        elif method.upper() == "POST":
            response = requests.post(url, headers=headers, json=payload, verify=False)
        elif method.upper() == "DELETE":
            response = requests.delete(url, headers=headers, verify=False)
        else:
            raise ValueError(f"Unsupported HTTP method: {method}")
        response.raise_for_status()
        result = response.json()
        logging.debug(f"REST API response: {result}")
        return result
    except json.JSONDecodeError as e:
        logging.error(f"Failed to parse Action Input: {input_str} - {str(e)}")
        return {"error": f"Invalid Action Input format: {str(e)}"}
    except Exception as e:
        logging.error(f"Failed to execute REST call: {str(e)}", exc_info=True)
        return {"error": str(e)}

# Helper: Load and reduce OpenAPI spec
def load_and_reduce_spec(spec_path, paths_to_include=None):
    """
    Load and reduce an OpenAPI specification file.
    Args:
        spec_path (str): Path to the OpenAPI spec file.
        paths_to_include (list): Specific paths to include in the reduced spec.
    Returns:
        dict: Reduced OpenAPI specification.
    """
    if not os.path.exists(spec_path):
        raise FileNotFoundError(f"OpenAPI spec file '{spec_path}' not found.")
    with open(spec_path, 'r') as f:
        full_spec = json.load(f)
    return reduce_openapi_spec(full_spec, paths=paths_to_include)

# Load reduced OpenAPI spec
def get_reduced_spec():
    try:
        spec_path = "nautobot_openapi.json"
        reduced_spec = load_and_reduce_spec(spec_path, paths_to_include=["/api/graphql/"])
        return reduced_spec
    except Exception as e:
        logging.error(f"Failed to load or reduce OpenAPI spec: {str(e)}")
        return None

# NautobotController for CRUD Operations
class NautobotController:
    def __init__(self, nautobot_url, api_token):
        self.nautobot = nautobot_url.rstrip('/')
        self.api_token = api_token
        self.headers = {
            'Accept': 'application/json',
            'Authorization': f"Token {self.api_token}",
        }

    def graphql_query(self, query: str, variables: dict = None):
        endpoint = f"{self.nautobot}/api/graphql/"
        payload = {"query": query, "variables": variables or {}}
        try:
            response = requests.post(endpoint, headers=self.headers, json=payload, verify=False)
            response.raise_for_status()
            return response.json()
        except requests.HTTPError as e:
            if response.content:
                print(f"GraphQL error response: {response.content.decode()}")
            raise

# Tools for interacting with Nautobot
@tool
def query_nautobot(query: str, variables: dict = None) -> dict:
    """Fetch data from Nautobot using a GraphQL query."""
    logging.debug(f"Executing query_nautobot with query: {query} and variables: {variables}")
    try:
        nautobot_controller = NautobotController(
            nautobot_url=os.getenv("NAUTOBOT_URL"),
            api_token=os.getenv("NAUTOBOT_TOKEN")
        )
        result = nautobot_controller.graphql_query(query, variables)
        logging.debug(f"Result from Nautobot: {result}")
        return result
    except Exception as e:
        logging.error(f"Failed to execute query_nautobot: {str(e)}", exc_info=True)
        return {"error": f"Failed to query Nautobot: {str(e)}"}

# Initialize the LangChain agent
def initialize_agent():
    global llm, agent_executor
    if not llm:
        llm = ChatOpenAI(model_name="gpt-4", openai_api_key=st.session_state['OPENAI_API_KEY'])

        tools = [execute_graphql, execute_rest_call]
        tool_descriptions = render_text_description(tools)

        template = """
        Assistant is a Nautobot agent capable of dynamically constructing and executing REST and GraphQL queries based on user requests.

        TOOLS:
        {tools}

        GUIDELINES:
        1. Use 'execute_rest_call' for:
           - Counting objects (e.g., devices, IPs) by querying REST endpoints like '/api/dcim/devices/' or '/api/ipam/ip-addresses/' with method 'GET'. The 'count' field in the response provides the total.
           - Simple CRUD operations (e.g., delete a device).
           - Use 'limit=0' in the payload to get just the count without fetching all items.
           - For filtering (e.g., by manufacturer), use the appropriate REST filter parameters (e.g., 'manufacturer' with the slug value).
        2. Use 'execute_graphql' for:
           - Queries requiring specific data about individual items (e.g., a device's primary IP or an interface's status).
           - Nested or complex queries involving relationships between objects.
        3. Dynamically construct queries or endpoints based on the user's intent. Interpret whether the request is for a count or specific item data.
        4. For GraphQL:
           - Use 'devices' (plural) to query multiple devices or filter by name (e.g., devices(name: "device_name")).
           - Use 'device' (singular) only when querying by UUID (e.g., device(id: "uuid")).
           - Do NOT invent fields like 'count' or 'manufacturer'—stick to valid schema fields.
        5. Examples:
           - User: "How many devices are there?"
             Action: execute_rest_call
             Action Input: {{"endpoint": "/api/dcim/devices/", "method": "GET", "payload": {{"limit": 0}}}}
             Observation: Look for 'count' in the response (e.g., {{"count": 42, ...}})
           - User: "How many IP addresses are there?"
             Action: execute_rest_call
             Action Input: {{"endpoint": "/api/ipam/ip-addresses/", "method": "GET", "payload": {{"limit": 0}}}}
             Observation: Look for 'count' in the response
           - User: "How many Arista devices do I have?"
             Action: execute_rest_call
             Action Input: {{"endpoint": "/api/dcim/devices/", "method": "GET", "payload": {{"manufacturer": "arista", "limit": 0}}}}
             Observation: Look for 'count' in the response
           - User: "List all devices"
             Action: execute_graphql
             Action Input: "query {{ devices {{ name }} }}"
           - User: "Show primary IP for device ams01-asw-01"
             Action: execute_graphql
             Action Input: "query {{ devices(name: \"ams01-asw-01\") {{ name primary_ip4 {{ address }} }} }}"
           - User: "Show interfaces for device with name R1"
             Action: execute_graphql
             Action Input: "query {{ devices(name: \"R1\") {{ name interfaces {{ name status }} }} }}"
           - User: "Delete device R1"
             Action: execute_rest_call
             Action Input: {{"endpoint": "/api/dcim/devices/R1/", "method": "DELETE"}}

        FORMAT:
        Thought: [Reasoning]
        Action: [Tool Name]
        Action Input: ["query string" for GraphQL, or {{"endpoint": "...", "method": "...", "payload": {{...}}}} for REST]
        Observation: [Tool Response]
        Final Answer: [Answer to user]

        Previous conversation:
        {chat_history}

        New input: {input}

        {agent_scratchpad}
        """
        prompt_template = PromptTemplate(
            template=template,
            input_variables=["input", "chat_history", "agent_scratchpad"],
            partial_variables={
                "tools": tool_descriptions,
                "tool_names": ", ".join([tool.name for tool in tools]),
            }
        )

        agent = create_react_agent(llm=llm, tools=tools, prompt=prompt_template)
        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            handle_parsing_errors=True,
            verbose=True,
        )

# Streamlit App
def configure_page():
    st.title("Configure API Settings")
    nautobot_url = os.getenv("NAUTOBOT_URL", "")
    openai_key = os.getenv("OPENAI_API", "")
    api_token = os.getenv("NAUTOBOT_TOKEN", "")

    nautobot_url_input = st.text_input("Nautobot URL", value=nautobot_url, placeholder="https://demo.nautobot.com")
    openai_token_input = st.text_input("OpenAI API Token", value=openai_key, type="password")
    nautobot_token_input = st.text_input("Nautobot API Token", value=api_token, type="password")

    if st.button("Save and Continue"):
        if not nautobot_url_input or not openai_token_input or not nautobot_token_input:
            st.error("All fields are required.")
        else:
            st.session_state['NAUTOBOT_URL'] = nautobot_url_input
            st.session_state['NAUTOBOT_TOKEN'] = nautobot_token_input
            st.session_state['OPENAI_API_KEY'] = openai_token_input
            os.environ['NAUTOBOT_URL'] = nautobot_url_input
            os.environ['NAUTOBOT_TOKEN'] = nautobot_token_input
            os.environ['OPENAI_API_KEY'] = openai_token_input
            st.success("Configuration saved! Redirecting to chat...")
            st.session_state['page'] = "chat"

def chat_page():
    st.title("Chat with Nautobot AI Agent")
    user_input = st.text_input("Ask Nautobot a question:")

    if "OPENAI_API_KEY" not in st.session_state:
        st.error("Please configure settings first.")
        st.session_state['page'] = "configure"
        return

    initialize_agent()

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = ""

    if st.button("Send"):
        if user_input:
            try:
                response = agent_executor.invoke({
                    "input": user_input,
                    "chat_history": st.session_state.chat_history,
                    "agent_scratchpad": "",
                })
                logging.debug(f"AI Response: {response}")
                st.session_state.chat_history += f"\nUser: {user_input}\nAssistant: {response.get('output', 'No response')}"
                st.write(f"Assistant: {response.get('output', 'No response')}")
            except Exception as e:
                logging.error(f"Error during chat: {str(e)}", exc_info=True)
                st.error(f"Error: {str(e)}")

# Page Navigation
if 'page' not in st.session_state:
    st.session_state['page'] = "configure"

if st.session_state['page'] == "configure":
    configure_page()
elif st.session_state['page'] == "chat":
    chat_page()
