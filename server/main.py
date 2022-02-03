from typing import Optional
from fastapi import FastAPI, File, UploadFile
import model as model
from torchvision import datasets, transforms as T
import torch.optim as optim
import torch 
import os.path
from PIL import Image
import io

from pymongo import MongoClient
client = MongoClient(port=27017)

# use pytorch code from https://nextjournal.com/gkoehler/pytorch-mnist


random_seed = 1
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)

root = "./"
learning_rate = 0.01
momentum = 0.5
n_epochs = 3
batch_size_train = 64
batch_size_test = 1000

transform = T.Compose([T.ToTensor(), T.Normalize(
    (0.1307,), (0.3081,))])

network = model.Net()

if not os.path.exists('model.pth'):
    train_loader = torch.utils.data.DataLoader(datasets.MNIST(
        root, train=True, download=True, transform=transform), batch_size=batch_size_train, shuffle=True)
    test_loader = torch.utils.data.DataLoader(datasets.MNIST(
        root, train=False, download=True, transform=transform), batch_size=batch_size_test, shuffle=True)

    optimizer = optim.SGD(network.parameters(), lr=learning_rate,
                    momentum=momentum)

    for epoch in range(1, n_epochs + 1):
        model.train(network, optimizer,train_loader, epoch)

    torch.save(network.state_dict(), 'model.pth')

network.load_state_dict(torch.load('model.pth'))

async def predict(pil_img):
    img_tensor = torch.unsqueeze(transform(pil_img), 0)
    output = network(img_tensor)
    _, pred = output.data.max(1, keepdim=True)
    return pred.item()


app = FastAPI()
@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile= File(...)):
    img = await file.read()
    pil_img = Image.open(io.BytesIO(img))
    pred = await predict(pil_img)
    return {"filename": file.filename, "prediction":pred}
