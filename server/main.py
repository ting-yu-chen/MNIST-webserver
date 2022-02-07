from typing import Optional
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import model as model
from PIL import Image
import os.path
import io
from pymongo.errors import PyMongoError
from pymongo import MongoClient


# Connect to mongo db 
try:
    client = MongoClient(
        username='admin', password='1234', host='mongodb', 
        port=27017)

    db = client.database
    predictions = db.predictions
except PyMongoError as e:
    print("database connection error ", e)


app = FastAPI()

# POST endpoint fpr uploading image for mnist prediction
@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile= File(...)):
    img = await file.read()
    pil_img = Image.open(io.BytesIO(img))
    pred = model.predict(pil_img)
    basename = os.path.basename(file.filename)
    # insert the request and prediction results to the database 
    predictions.insert_one(
        {"filename": basename, 
        "img": img, 
        "prediction":pred}
    )
    return {"filename": basename, "prediction":pred}

# GET endpoint fpr retrieve previous predictions based on filename 
@app.get("/prediction/{filename}")
def read_root(filename:str):
    entry = predictions.find_one({"filename": filename})
    if not entry:
        return {"message": "NO prediction found"}
    return {"filename": filename, "prediction": entry['prediction']}

# GET endpoint fpr retrieve all previous predictions
@app.get("/all")
def read_root():
    cursor = predictions.find()
    entries = []
    for entry in cursor:
        entries.append({"filename": entry['filename'], "prediction": entry['prediction']})
    return entries 


# for making fetch from local react possible 
origins = [
    "http://localhost",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
