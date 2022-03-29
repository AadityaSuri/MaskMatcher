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
from PIL import Image

transform = transforms.Compose(
    [transforms.Resize((64, 64)),
     transforms.Grayscale(1),
     transforms.ToTensor(),
     transforms.Normalize((0.5), (0.5))]
)

transform_3 = transforms.Compose(
    [transforms.Resize((24, 24)),
     #transforms.Grayscale(),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)
batch_size = 16

ds = tv.datasets.ImageFolder("./dataset/train_1", transform_3)
# test_ds = tv.datasets.ImageFolder("./dataset/test", transform_3)

train_size = int(0.8 * len(ds))
test_size = int(len(ds)) - train_size
train_ds, test_ds = torch.utils.data.random_split(ds, [train_size, test_size])
train_dl = DataLoader(train_ds, batch_size, shuffle=True, num_workers = 0)
test_dl = DataLoader(test_ds, batch_size, shuffle=True, num_workers = 0)

classes = ("with_mask", "without_mask")
# #
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


# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu" #selects device to run NN on, if GPU is available, prints "cuda", otherwise "cpu"
print("Using {} device".format(device)) # print statement for above device variable

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(24*24*3, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 2)
        )


    def forward(self, x):
        # print(x.shape)
        x = torch.flatten(x, start_dim=1)
        # print(x.shape)

        logits = self.linear_relu_stack(x)
        return logits


#
model = CNN().to(device)
# print(model)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr = 1e-3, momentum=0.9)


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for i, data in enumerate(dataloader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

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
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


# epochs = 10   #starting with 5 epochs --> may need to adjust
# for t in range(epochs):
#     print(f"Epoch {t+1}\n-------------------------------")
#     train(train_dl, model, loss_fn, optimizer)
#     test(test_dl, model, loss_fn)
# print("Done!")
# torch.save(model.state_dict(), "model.pth")
# print("Saved PyTorch Model State to model.pth")
# #
# # # Using the model for some testing
# model = CNN()
# model.load_state_dict(torch.load("model.pth"))
#
# model.eval()
# #
# actMask = 0
# actWMask = 0
# predMask = 0
# predWMask = 0
# counter = 0
# n = len(test_ds)
# for i in range(n):
#
#     x, y = test_ds[i][0], test_ds[i][1]
#     # print(x.shape)
#     with torch.no_grad():
#         pred = model(x.reshape((1,24,24,3)))
#         predicted, actual = classes[pred[0].argmax(0)], classes[y]
#
#         #if predicted != actual:
#         print(f'Predicted: "{predicted}", Actual: "{actual}"')
#         if actual == "without_mask":
#             actWMask = actWMask + 1
#
#         if actual == "with_mask":
#             actMask = actMask + 1
#
#         if predicted == actual and predicted == "without_mask":
#             predWMask = predWMask + 1
#
#         if predicted == "with_mask" and predicted == actual:
#             predMask = predMask + 1
#
#         if predicted == actual:
#             counter = counter + 1
#
# print(predMask/actMask)
# print(predWMask/actWMask)
# print(counter/n)

# #

model = CNN()
model.load_state_dict(torch.load("model.pth"))

model.eval()

print("started")
img = Image.open("eranImages/withmask8_.png").convert('RGB')
x = transform_3(img)
x = x.unsqueeze(0)

output = model(x)  # Forward pass
pred = torch.argmax(output, 1)  # Get predicted class if multi-class classification
print('Image predicted as', classes[pred])

