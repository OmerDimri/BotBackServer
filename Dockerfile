FROM python:3.11

COPY ./requirements.txt /app/
WORKDIR /app/

RUN pip install -r requirements.txt

COPY . /app/
WORKDIR /app/

CMD [ "python", "./manage.py", "runserver", "0.0.0.0:8002", "--noreload" ]