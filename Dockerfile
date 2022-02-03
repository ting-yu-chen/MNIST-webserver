FROM python:3.9

WORKDIR /usr/src/app

COPY requirements.txt ./
RUN ls 
RUN pip3 install --no-cache-dir -r requirements.txt

COPY ./server .

CMD [ "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]