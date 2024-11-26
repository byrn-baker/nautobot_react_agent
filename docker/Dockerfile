FROM ubuntu:latest

ARG DEBIAN_FRONTEND=noninteractive

# Install required system utilities and Python packages
RUN apt-get update -y && apt-get install -y \
    wget \
    python3 \
    python3-pip \
    dos2unix && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip install --break-system-packages -U --quiet langchain_community \
    streamlit \
    ollama

# Copy application code and scripts
COPY nautobot_react_agent /nautobot_react_agent/
COPY scripts /scripts/

# Ensure scripts are executable
RUN chmod +x /scripts/startup.sh

# Default entrypoint
CMD ["/bin/bash", "/scripts/startup.sh"]
