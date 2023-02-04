FROM python:3.10

# for opencv
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

WORKDIR /app

COPY ./requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt

COPY ./* /app/

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]