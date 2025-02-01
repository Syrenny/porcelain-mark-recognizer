FROM python:3.12.3-slim

WORKDIR /home

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /home/

RUN pip3 install -r requirements.txt

COPY ./app /home/app

EXPOSE 8502

CMD ["python3", "-m", "app.main"]