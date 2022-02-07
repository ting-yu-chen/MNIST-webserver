import torch.nn as nn
import torch.nn.functional as F
import torch
from torchvision import datasets, transforms as T
import torch.optim as optim
import os 

# use pytorch code from https://nextjournal.com/gkoehler/pytorch-mnist

log_interval = 10
random_seed = 1
root = "./"
learning_rate = 0.01
momentum = 0.5
n_epochs = 3
batch_size_train = 64
batch_size_test = 1000

torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)

transform = T.Compose([T.ToTensor(), T.Normalize(
    (0.1307,), (0.3081,))])

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)


def train(network, optimizer, dataloader, epoch):
    train_losses = []
    network.train()
    for batch_idx, (data, target) in enumerate(dataloader):
        optimizer.zero_grad()
        output = network(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(dataloader.dataset),
                100. * batch_idx / len(dataloader), loss.item()))
            train_losses.append(loss.item())
            
def test(network, dataloader):
    test_losses = []
    network.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in dataloader:
            output = network(data)
            test_loss += F.nll_loss(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
        test_loss /= len(dataloader.dataset)
        test_losses.append(test_loss)
        print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(dataloader.dataset),
            100. * correct / len(dataloader.dataset)))

# load pretrained model or train the model 
def getModel():
    network = Net()
    if not os.path.exists('model.pth'):
        train_loader = torch.utils.data.DataLoader(datasets.MNIST(
            root, train=True, download=True, transform=transform), batch_size=batch_size_train, shuffle=True)
        test_loader = torch.utils.data.DataLoader(datasets.MNIST(
            root, train=False, download=True, transform=transform), batch_size=batch_size_test, shuffle=True)

        optimizer = optim.SGD(network.parameters(), lr=learning_rate,
                        momentum=momentum)

        for epoch in range(1, n_epochs + 1):
            train(network, optimizer,train_loader, epoch)
        
        test(network, test_loader)
        torch.save(network.state_dict(), 'model.pth')

    network.load_state_dict(torch.load('model.pth'))
    return network

def predict(pil_img):
    network = getModel()  
    img_tensor = torch.unsqueeze(transform(pil_img), 0)
    output = network(img_tensor)
    _, pred = output.data.max(1, keepdim=True)
    return pred.item()
