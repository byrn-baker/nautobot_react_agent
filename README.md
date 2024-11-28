# netbox_react_agent
forked from https://github.com/automateyournetwork/netbox_react_agent

An artificial intelligence ReAct Agent for NetBox 

Welcome to the NetBox AI Agent project! This application provides a natural language interface for interacting with NetBox APIs, enabling CRUD (Create, Read, Update, Delete) operations through an intuitive chat-like interface powered by AI.

This project simplifies network management by combining AI-driven agents with the NetBox API to streamline and automate common network tasks.

## Branches Overview

### Main Branch

Powered by ChatGPT (gpt-4o)

Requires OpenAI API Key

Offers high accuracy and performance for handling natural language queries.

Recommended for production use.

API costs apply.

### Ollama Branch

Powered by Local LLM using Ollama

Completely free and private: All computations happen locally.

No external API calls required.

Performance: Works well for basic tasks but is less sophisticated compared to the ChatGPT-based version.

Recommended for personal or offline use cases.

## Features

Natural Language Interface: Interact with NetBox APIs using plain English commands.

CRUD Operations: Perform Create, Read, Update, and Delete tasks on your NetBox data.

API Validation: Ensures commands align with supported NetBox API endpoints.

Dynamic Tools: Auto-detects and leverages the appropriate tools for each task.

Local or Cloud Options: Choose between the main branch for high performance or the Ollama branch for privacy and offline capabilities.

## Setup Instructions

### Prerequisites
Docker and Docker Compose installed.

OpenAI API Key (for the main branch).

Optional: Ollama installed for the local branch.

## Quick Start

### Clone the Repository

``` bash
git clone https://github.com/<your-repo-name>/netbox-ai-agent.git
cd netbox-ai-agent
```

### Run the Application

```bash
docker-compose up
```

### Access the App

Open your browser and go to http://localhost:8501.

Configure API Keys

## For the main branch:

Provide your NetBox API URL, NetBox Token, and OpenAI API Key in the configuration page.

## For the Ollama branch:
Provide only your NetBox API URL and NetBox Token.

# Start Chatting

Use natural language to manage your NetBox data. Example commands:

"Fetch all devices in the DC1 site."

"Create a new VLAN in site DC2."

## Key Components

NetBoxController: Manages interactions with the NetBox API.

LangChain ReAct Agent: Dynamically selects tools to process natural language queries.

Streamlit Interface: Provides an intuitive chat-like web UI.

## FAQs

Q: Which branch should I use?

Use the main branch for production-grade performance and OpenAI's latest capabilities.

Use the Ollama branch for offline and private operations, but expect reduced performance.

Q: How do I switch between branches?

To use the Ollama branch, run:

```bash
git checkout ollama
```

Then re-run the Docker setup.

## Troubleshooting

Docker Issues: Ensure Docker is running and your system meets the necessary prerequisites.

OpenAI Key Errors: Check that your API key is valid and added correctly.

NetBox API Errors: Verify your NetBox instance is accessible, and the API token has the required permissions.