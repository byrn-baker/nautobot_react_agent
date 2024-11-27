# nautobot_react_agent
An artificial intelligence Nautobot ReAct Agent

This has been forked from https://github.com/automateyournetwork/Nautobot_react_agent. Please follow John Capobianco at the following to see what he is doing with AI. Great stuff on his channels

https://bsky.app/profile/automateyournetwork.ca
https://www.youtube.com/@johncapobianco2527
https://x.com/John_Capobianco
https://www.linkedin.com/in/john-capobianco-644a1515/

Welcome to the Nautobot AI Agent project! This application provides a natural language interface for interacting with Nautobot APIs, enabling CRUD (Create, Read, Update, Delete) operations through an intuitive chat-like interface powered by AI.

This project simplifies network management by combining AI-driven agents with the Nautobot API to streamline and automate common network tasks.

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
git clone https://github.com/<your-repo-name>/nautobot_react_agent.git
cd nautobot_react_agent
```

### If you want to build your own container
Update the docker-compose.yml with your values.
``` bash
docker-compose build
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
