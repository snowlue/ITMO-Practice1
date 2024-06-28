FROM python:3.11-slim

RUN apt-get update && \
    python3 -m pip install --upgrade pip

RUN apt-get install libglfw3 -y

COPY ./requirements.txt ./
RUN python3 -m pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html

# you can set this value in the docker run command
ENV NUM_EPISODES=100

COPY . /app
WORKDIR /app

CMD ["python3", "main.py"]
