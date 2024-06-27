FROM python:3.9-slim

RUN apt-get update && \
    pip3 install --upgrade pip

COPY requirements.txt .
RUN pip3 install -r requirements.txt

ENV NUM_EPISODES=100  # you can set this value in the docker run command

COPY . /app
WORKDIR /app

CMD ["python3", "main.py"]
