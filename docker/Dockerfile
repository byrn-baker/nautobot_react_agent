FROM ubuntu:latest

ARG DEBIAN_FRONTEND=noninteractive

RUN echo "==> Upgrading apk and installing system utilities ...." \
 && apt -y update \
 && apt-get install -y wget \
 && apt-get -y install sudo

RUN echo "==> Installing Python3 and pip ...." \  
 && apt-get install python3 -y \
 && apt install python3-pip -y

RUN echo "==> Install dos2unix..." \
  && sudo apt-get install dos2unix -y 

RUN echo "==> Install requirements.." \
  && pip install --break-system-packages -U --quiet langchain_community \
  && pip install --break-system-packages streamlit --upgrade \
  && pip install --break-system-packages python-dotenv \
  && pip install --break-system-packages openai \
  && pip install --break-system-packages tiktoken \
  && pip install --break-system-packages graphql-core

COPY /nautobot_react_agent /nautobot_react_agent/
COPY .env /nautobot_react_agent/
COPY /scripts /scripts/

RUN echo "==> Convert script..." \
  && dos2unix /scripts/startup.sh

CMD ["/bin/bash", "/scripts/startup.sh"]