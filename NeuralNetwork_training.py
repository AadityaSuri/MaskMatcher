import torch
import torchvision as tv
import torchvision.transforms as transforms
import torch.nn as nn
from torch.utils.data import DataLoader

# transform applied for testing
transform_3 = transforms.Compose(
    [transforms.Resize((32, 32)), # resizes image to 32 * 32
     transforms.ToTensor(), # converts image to tensor
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))] # Normalizes the tensor
)

# transform applied to training set
transform_3_train = transforms.Compose(
    [transforms.Resize((32, 32)), # resizes image to 32 * 32
     transforms.RandomRotation((-40, 40)), # rotates image by a random angle between -40 and +40 degrees
     transforms.ToTensor(), # converts image to tensor
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))] # Normalizes the tensor
)

batch_size = 16 # batch size is set to 16

ds = tv.datasets.ImageFolder("./dataset/train_1", transform_3_train) # loads dataset and applies the transform_3_train transform to all images

train_size = int(0.8 * len(ds))
test_size = int(len(ds)) - train_size
train_ds, test_ds = torch.utils.data.random_split(ds, [train_size, test_size]) # splits dataset into training set and test set, with 80% of the total dataset going to training set
train_dl = DataLoader(train_ds, batch_size, shuffle=True, num_workers = 0)
test_dl = DataLoader(test_ds, batch_size, shuffle=True, num_workers = 0)

#classification categories
classes = ("with_mask", "without_mask")

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"  # selects device to run NN on, if GPU is available, prints "cuda", otherwise "cpu"
print("Using {} device".format(device)) # print statement for above device variable

# Neural Network class
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(32 * 32 * 3, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 2)
        )

    #feed forwarding
    def forward(self, x):

        x = torch.flatten(x, start_dim=1)
        logits = self.linear_relu_stack(x)
        return logits
        return x



model = CNN().to(device)
print(model)

# using crossEntropyLoss to calculate loss
loss_fn = nn.CrossEntropyLoss()

# optimizing using stochastic gradient descent function
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)


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
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


epochs = 10  #starting with 5 epochs --> may need to adjust
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dl, model, loss_fn, optimizer)
    test(test_dl, model, loss_fn)
print("Done!")
torch.save(model.state_dict(), "model.pth")
print("Saved PyTorch Model State to model.pth")


# Using the model for some testing
model = CNN()
model.load_state_dict(torch.load("model.pth"))
model.eval()

counter = 0
n = len(test_ds)
for i in range(n):

    x, y = test_dl[i][0], test_dl[i][1]
    with torch.no_grad():
        pred = model(x.reshape((1, 32, 32, 3)))
        predicted, actual = classes[pred[0].argmax(0)], classes[y]

        print(f'Predicted: "{predicted}", Actual: "{actual}"')

        if predicted == actual:
            counter = counter + 1

# prints total accuracy rate
print(counter/n)
