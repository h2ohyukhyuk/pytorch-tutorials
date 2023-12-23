# https://pytorch.org/tutorials/intermediate/tensorboard_tutorial.html
import matplotlib.pyplot as plt
import numpy as np

import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import os

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

trainset = torchvision.datasets.FashionMNIST('data', download=True, train=True, transform=transform)
testset = torchvision.datasets.FashionMNIST('data', download=True, train=False, transform=transform)

from torch.utils.data import DataLoader
trainloader = DataLoader(trainset, shuffle=True, batch_size=4)
testloader = DataLoader(testset, shuffle=False, batch_size=4)

classes = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot')

def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)

    img = img / 2 + 0.5
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(img, (1,2,0)))

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(in_features=16*4*4, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.fc3 = nn.Linear(in_features=84, out_features=10)

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = F.relu(x1)
        x2 = self.pool1(x1)
        x3 = self.conv2(x2)
        x3 = F.relu(x3)
        x3 = self.pool2(x3)
        x3 = x3.view(-1, 16*4*4)
        x4 = self.fc1(x3)
        x4 = F.relu(x4)
        x5 = self.fc2(x4)
        x5 = F.relu(x5)
        x6 = self.fc3(x5)
        return x6

net = Net()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


os.makedirs('runs/fashion_mnist', exist_ok=True)
writer = SummaryWriter('runs/fashion_mnist/%s' % datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))

dataiter = iter(trainloader)
images, labels = next(dataiter)

img_grid = torchvision.utils.make_grid(images)
matplotlib_imshow(img_grid)
img_grid = img_grid / 2 + 0.5
writer.add_image('four_fashion_mnist_images', img_grid)
writer.add_graph(net, images)
# helper function
def select_n_random(data, labels, n=100):
    '''
    Selects n random datapoints and their corresponding labels from a dataset
    '''
    assert len(data) == len(labels)

    perm = torch.randperm(len(data))
    return data[perm][:n], labels[perm][:n]

# select random images and their target indices
images, labels = select_n_random(trainset.data, trainset.targets)

# get the class labels for each image
class_labels = [classes[lab] for lab in labels]

# log embeddings
features = images.view(-1, 28 * 28)
writer.add_embedding(features,
                    metadata=class_labels,
                    label_img=images.unsqueeze(1))

def images_to_probs(net, images):
    out = net(images)

    _, preds_tensor = torch.max(out, dim=1)
    preds = np.squeeze(preds_tensor.numpy())
    probs = [F.softmax(el, dim=0)[i].item() for i, el in zip(preds, out)]
    return preds, probs

def plot_classes_preds(net, images, labels):
    '''
    Generates matplotlib Figure using a trained network, along with images
    and labels from a batch, that shows the network's top prediction along
    with its probability, alongside the actual label, coloring this
    information based on whether the prediction was correct or not.
    Uses the "images_to_probs" function.
    '''
    preds, probs = images_to_probs(net, images)
    # plot the images in the batch, along with predicted and true labels
    fig = plt.figure(figsize=(12, 3))
    for idx in np.arange(4):
        ax = fig.add_subplot(1, 4, idx+1, xticks=[], yticks=[])
        imp_tmp = images[idx].clone().detach() / 2 + 0.5
        matplotlib_imshow(images[idx], one_channel=True)
        ax.set_title("{0}, {1:.1f}%\n(label: {2})".format(
            classes[preds[idx]],
            probs[idx] * 100.0,
            classes[labels[idx]]),
                    color=("green" if preds[idx]==labels[idx].item() else "red"))
    return fig

running_loss = 0.0
for epoch in range(1):  # loop over the dataset multiple times

    for i, data in enumerate(trainloader, 0):

        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 1000 == 999:    # every 1000 mini-batches...
            cnt_step = epoch*len(trainloader) + i
            # ...log the running loss
            writer.add_scalar('training loss',
                              running_loss / 1000,
                              global_step=cnt_step)

            # ...log a Matplotlib Figure showing the model's predictions on a
            # random mini-batch
            writer.add_figure('predictions vs. actuals',
                              plot_classes_preds(net, inputs, labels),
                              global_step=cnt_step)
            running_loss = 0.0

print('Finished Training')
writer.close()