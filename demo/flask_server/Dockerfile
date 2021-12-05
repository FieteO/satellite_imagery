FROM python:3.8-slim-buster
ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0

WORKDIR /usr/src/app

COPY requirements .
RUN pip install --no-cache-dir -r requirements

COPY app .

CMD [ "flask", "run" ]