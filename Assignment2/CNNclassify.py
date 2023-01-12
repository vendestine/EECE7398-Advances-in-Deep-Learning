# load
import torch
import torchvision
from torchvision import transforms
import torch.utils.data as Data

# build model
import torch.nn as nn
import torch.nn.functional as F

# show image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image

# system
import os
import sys
import glob

# ---------------- define Hyper parameters ----------------
BATCH_SIZE = 125
EPOCHS = 50
LEARNING_RATE = 0.01

# -------------------- image augmentation ------------------
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])

# ----------------- prepare training data -----------------------
train_data = torchvision.datasets.CIFAR10(
    root='../data.cifar10',
    train=True,
    transform=transform,
    download=False
)

train_loader = Data.DataLoader(
    dataset=train_data,
    batch_size=BATCH_SIZE,
    shuffle=True
)

# ----------------- prepare testing data -----------------------
test_data = torchvision.datasets.CIFAR10(
    root='../data.cifar10/',
    train=False,
    transform=transform,
    download=False
)

test_loader = Data.DataLoader(
    dataset=test_data,
    batch_size=BATCH_SIZE,
    shuffle=True
)

# ----------------- build the model ------------------------
class My_ass2net(nn.Module):
    def __init__(self):
        super(My_ass2net, self).__init__()
        # layer structure
        self.conv1 = nn.Sequential(       # input shape (3, 32, 32)
            nn.Conv2d(
                in_channels=3,            # input height
                out_channels=32,          # n_filters
                kernel_size=5,            # filter size
                stride=1,                 # filter movement/step
                padding=2,                # if want same width and length of this image after Conv2d, padding=(kernel_size-1)/2 if stride=1
            ),                            # output shape (32, 32, 32)
            nn.MaxPool2d(kernel_size=2),  # choose max value in 2x2 area, output shape (32, 16, 16)
            nn.BatchNorm2d(32),
            nn.ReLU(),  # activation
        )
        self.conv2 = nn.Sequential(       # (32, 16, 16)
            nn.Conv2d(32, 64, 3, 1, 1),   # (64, 16, 16)
            nn.Conv2d(64, 64, 3, 1, 1),   # (64, 16, 16)
            nn.MaxPool2d(2, 2),           # (64, 8, 8)
            nn.BatchNorm2d(64),
            nn.ReLU(),  # activation
        )
        self.conv3 = nn.Sequential(       # (64, 8, 8)
            nn.Conv2d(64, 128, 3, 1, 1),  # (128, 8, 8)
            nn.Conv2d(128, 128, 3, 1, 1), # (128, 8, 8)
            nn.MaxPool2d(2),  # output shape (128, 4, 4)
            nn.BatchNorm2d(128),
            nn.ReLU(),  # activation
        )

        self.fc1 = nn.Linear(128 * 4 * 4, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, 256)
        self.out = nn.Linear(256, 10)


    def forward(self, x):
        x = self.conv1(x)
        y = x
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        outp = self.out(x)
        return outp, y  # return y for visualization

# ----------------- create the model ----------------------------
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# print(device)

model = My_ass2net().to(device)
loss_func = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), momentum=0.7, lr=LEARNING_RATE)


# -------------------------- helper function  ------------------
def save_model(epoch, max_acc):
    # creat model directory if not exist
    if not os.path.isdir('./model'):
        os.mkdir('./model')

    # when better model exists, clean all previous model
    for file in glob.glob("model/*"):
        os.remove(file)

    # save new model
    path = f"./model/epoch_{epoch}_test_acc_{max_acc}.pth"
    torch.save(model.state_dict(), path)
    return path



def load_model(model, path):
    model.load_state_dict(torch.load(path))

# --------------------- test -------------------------------
def test(loader):
    model.eval()
    correct = 0
    total = 0
    total_loss = 0

    with torch.no_grad():
        for data in loader:
            images, labels = data[0].to(device), data[1].to(device)
            # calculate outputs by running images through the network
            outputs, y = model(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total_loss += loss_func(outputs, labels)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = f"{(100 * correct / total):.4f}"
    loss = f"{(total_loss / len(loader.dataset)):.4f}"
    return accuracy, loss


# ------------------- train ----------------------------
def train():
    max_acc = 0
    print("Epoch \t Train Loss \t Train Acc % \t Test Loss \t Test Acc %")
    for epoch in range(EPOCHS):
        for step, (input, target) in enumerate(train_loader):
            model.train()  # set the model in training mode
            input, target = input.to(device), target.to(device)
            # forward
            output, y = model(input)  # Get the predicted value
            loss = loss_func(output, target)  # calculate the loss function
            optimizer.zero_grad()  # Optimization. gradient be zero first
            # backward
            loss.backward()
            optimizer.step()
        train_accuracy, train_loss = test(train_loader)
        test_accuracy, test_loss = test(test_loader)
        print(f"{epoch + 1}/{EPOCHS} \t {train_loss} \t {train_accuracy} \t {test_loss} \t {test_accuracy}")

        # only save better performance model
        test_accuracy = float(test_accuracy)
        if test_accuracy > max_acc:
            max_acc = test_accuracy
            path = save_model(epoch, max_acc)
    print(f"Model saved in file: {path}")


# -------------------------- test ------------------------
def predict(img):
    # load the model
    path = glob.glob("model/*")[0]
    load_model(model, path)
    model.cpu()

    # load one image
    image = Image.open(img)
    image = image.convert('RGB')
    image = transform(image)
    image = image.unsqueeze(0)


    # prediction
    model.eval()
    output, result = model(image)
    _, predicted = torch.max(output.data, 1)
    classes = test_data.classes
    print(f"prediction result: {classes[predicted[0]]}")

    # ouput the first conv layer feature map
    feature_maps = result
    plt.figure(figsize=(6, 6))
    for i in range(32):
        plt.subplot(6, 6, i + 1)
        plt.axis('off')
        plt.imshow(feature_maps[0][i].detach(), cmap='gray', interpolation='spline36')
    plt.savefig('CONV_rslt.png')
    # plt.show()


if __name__ == "__main__":
    option = sys.argv[1]
    if option == "train":
        train()
    else:
        predict(sys.argv[2])
