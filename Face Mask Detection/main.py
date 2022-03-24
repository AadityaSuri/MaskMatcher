import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import torchvision as tv
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

transform = transforms.Compose(
    [transforms.Resize((64, 64)),
     # transforms.Grayscale(3),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)

batch_size = 8

train_ds = tv.datasets.ImageFolder("./dataset/train", transform)
test_ds = tv.datasets.ImageFolder("./dataset/test", transform)

train_dl = DataLoader(train_ds, batch_size, shuffle=True, num_workers = 0)
test_dl = DataLoader(test_ds, batch_size, shuffle=True, num_workers = 0)

classes = ("with_mask", "without_mask")
#
# def imshow(img):
#   ''' function to show image '''
#   img = img / 2 + 0.5 # unnormalize
#   npimg = img.numpy() # convert to numpy objects
#   plt.imshow(np.transpose(npimg, (1, 2, 0)))
#   plt.show()
#
# # get random training images with iter function
# # dataiter = iter(train_dl)
# # images, labels = dataiter.next()
# images, labels = next(iter(train_dl))
#
# print(' '.join('%s' % classes[labels[j]] for j in range(batch_size)))
# # call function on our images
# imshow(tv.utils.make_grid(images))

class CNN(nn.Module):
    def __init__(self):
        # super().__init__()
        # self.conv1 = nn.Conv2d(3, 6, 5)
        # self.pool = nn.MaxPool2d(2, 2)
        # self.conv2 = nn.Conv2d(6, 16, 5)
        # self.fc1 = nn.Linear(16 * 5 * 5, 120)
        # self.fc2 = nn.Linear(120, 84)
        # self.fc3 = nn.Linear(84, 2)
        super(CNN, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(3*64*64, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 2)
        )

    def forward(self, x):
        # x = self.pool(F.relu(self.conv1(x)))
        # x = self.pool(F.relu(self.conv2(x)))
        # x = torch.flatten(x, 1) # flatten all dimensions except batch
        # x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        # x = self.fc3(x)
        # return x
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


model = CNN()
print(model)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr = 1e-3, momentum=0.9)

# start = torch.cuda.Event(enable_timing=True)
# end = torch.cuda.Event(enable_timing=True)
#
# start.record()

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for i, data in enumerate(dataloader, 0):
        inputs, labels = data

        # Compute prediction error
        pred = model(inputs)
        loss = loss_fn(pred, labels)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            loss, current = loss.item(), i * len(inputs)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


epochs = 5   #starting with 5 epochs --> may need to adjust
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dl, model, loss_fn, optimizer)
    test(test_dl, model, loss_fn)
print("Done!")

torch.save(model.state_dict(), "model.pth")
print("Saved PyTorch Model State to model.pth")
#
# # Using the model for some testing
# model = CNN()
# model.load_state_dict(torch.load("model.pth"))
#
# model.eval()
#
# for i in range(1000):
#
#     x, y = test_ds[i][0], test_ds[i][1]
#     with torch.no_grad():
#         pred = model(x)
#         predicted, actual = classes[pred[0].argmax(0)], classes[y]
#         print(f'Predicted: "{predicted}", Actual: "{actual}"')

