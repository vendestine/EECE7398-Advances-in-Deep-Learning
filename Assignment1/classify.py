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
EPOCHS = 30
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


# --------------------- plot dataset information -------------------
# print(len(train_data), len(test_data))
# print(train_data[0][0].shape)
# print(train_data.classes)
# classes = train_data.classes
#
# # functions to show an image
# def imshow(img):
#     img = img / 2 + 0.5     # unnormalize
#     npimg = img.numpy()
#     plt.imshow(np.transpose(npimg, (1, 2, 0)))
#     plt.show()
#
#
# # functions to show an image
# dataiter = iter(train_loader)
# images, labels = dataiter.next()
#
# # show images
# imshow(torchvision.utils.make_grid(images))
# # print labels
# print(' '.join(f'{classes[labels[j]]:5s}' for j in range(5)))

# ----------------- build the model ------------------------
class My_ass1net(nn.Module):
    def __init__(self):
        super(My_ass1net, self).__init__()
        # layer structure
        self.hidden1 = nn.Linear(32 * 32 * 3, 1000)
        self.hidden2 = nn.Linear(1000, 100)
        self.out = nn.Linear(100, 10)

    def forward(self, x):
        x = x.view(-1, 32 * 32 * 3)
        # using ReLU function
        x = self.hidden1(x)
        x = F.relu(x)
        x = self.hidden2(x)
        x = F.relu(x)
        outp = self.out(x)
        return outp


# ----------------- create the model ----------------------------
model = My_ass1net()
loss_func = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), momentum=0.7, lr=LEARNING_RATE)


# ------------------- train ----------------------------
def train():
    print("Epoch | step | test_loss | test_accuracy %")
    max_acc = 0
    for epoch in range(EPOCHS):
        for step, (input, target) in enumerate(train_loader):
            model.train()  # set the model in training mode
            # forward
            output = model(input)  # Get the predicted value
            loss = loss_func(output, target)  # calculate the loss function
            optimizer.zero_grad()  # Optimization. gradient be zero first
            # backward
            loss.backward()
            optimizer.step()

            #  Evaluate every 50 batches
            if step % 50 == 0:
                model.eval()  # switch the model to evaluation mode
                # test loss and test correction number
                test_loss = 0
                test_correct = 0

                for data, target in test_loader:
                    output = model(data)
                    criterion = nn.CrossEntropyLoss()
                    test_loss = criterion(output, target)
                    pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
                    test_correct += pred.eq(target.data.view_as(pred)).cpu().sum()
                test_loss /= len(test_loader.dataset)
                test_acc = float(100. * test_correct) / float(len(test_loader.dataset))

                print(f"{epoch + 1}/{EPOCHS} \t {step} \t {test_loss:.4f} \t {test_acc:.3f}")

                # only save better performance model
                if test_acc > max_acc:
                    max_acc = test_acc
                    save_model(epoch, max_acc)


# -------------------------- save model and load model ------------------
def save_model(epoch, max_acc):
    # when better model exists, clean all previous model
    for file in glob.glob("model/*"):
        os.remove(file)
    # save new model
    path = f"./model/epoch_{epoch}_test_acc_{max_acc}.pth"
    print(f"save model to {path}")
    torch.save(model.state_dict(), path)


def load_model(model, path):
    model.load_state_dict(torch.load(path))


# -------------------------- test ------------------------
def test(img):
    # load the model
    path = glob.glob("model/*")[0]
    # print(path)
    load_model(model, path)

    # load one image
    image = Image.open(img)
    image = image.convert('RGB')
    image = transform(image)

    # prediction
    model.eval()
    output = model(image)
    _, predicted = torch.max(output.data, 1)
    classes = test_data.classes
    print(f"prediction result: {classes[predicted[0]]}")


if __name__ == "__main__":
    option = sys.argv[1]
    if option == "train":
        train()
    else:
        test(sys.argv[2])
