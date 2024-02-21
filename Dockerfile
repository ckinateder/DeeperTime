FROM python:3.10.13-slim-bookworm

# copy the requirements file
COPY requirements.txt .

# update and install packages
RUN apt-get update
RUN apt-get install -qq build-essential

# install dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
