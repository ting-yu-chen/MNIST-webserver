# MNIST-webserver

Provide endpoints of HTTP request for handwriting digits recongnition

- Pretrain a CNN model from MNIST dataset 
- Save prediction entries in mongoDB
- Create endpoints for POST and GET HTTP request
    - POST: /uploadfile/ for uploading image and get the digit 
    - GET: /all/ for getting all previous predictions 


### Usage 
To build the docker images and start the container :
```
docker-compose up 
```

To send POST request:
```
url_upload = "http://127.0.0.1/uploadfile/"
img = [('file', (img_path, open(img_path, 'rb'), 'image/jpeg'))]
requests.post(url_upload, files=img)
```

