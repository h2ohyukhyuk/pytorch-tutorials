import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transform
import torchvision.datasets as dataset
import matplotlib.pyplot as plt
import numpy as np

tf = transform.Compose([transform.ToTensor(), transform.Normalize(0.5, 0.5)])
train_dataset = dataset.MNIST(root='data', train=True, transform=tf, download=True)
val_dataset = dataset.MNIST(root='data', train=False, transform=tf, download=True)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=True)
valid_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)

img, label = next(iter(train_dataloader))

def show_grid(xs):

    fig = plt.figure(figsize=(15,3))

    for i in range(len(xs)):

        img_grid = torchvision.utils.make_grid(xs[i][:4], nrow=2)
        img_grid = np.transpose(img_grid, (1, 2, 0)) * 0.5 + 0.5

        ax = fig.add_subplot(1, len(xs), i+1)
        ax.set_title(f'{i}')
        ax.imshow(img_grid)

    plt.show()


def label_to_grid(ys):
    batch_size = ys.size()[0]

    a = torch.zeros(size=(batch_size, 28, 28))

    for i in range(batch_size):
        y = ys[i].to(torch.int64)
        a[i, (y // 4) * 7:(y // 4) * 7 + 4, (y % 4) * 7:(y % 4) * 7 + 4] = 1

        # print(a[i, ::2, ::2])

    return a.unsqueeze(1)

class ProjectionLayer(nn.Module):
    def __init__(self, in_ch, out_ch, hid_ch):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_ch, hid_ch//2, 5, 1, 2),
            nn.BatchNorm2d(hid_ch//2),
            nn.ReLU(inplace=True))
        self.layer2 = nn.Sequential(
            nn.Conv2d(hid_ch//2, hid_ch, 5, 1, 2),
            nn.BatchNorm2d(hid_ch),
            nn.ReLU(inplace=True))
        self.layer3 = nn.Sequential(
            nn.Conv2d(hid_ch, hid_ch, 5, 1, 2),
            nn.BatchNorm2d(hid_ch),
            nn.ReLU(inplace=True),
        )
        self.layer4 = nn.Conv2d(hid_ch, out_ch, 5, 1, 2)

    def forward(self, x):
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = x3 + x2
        x5 = self.layer4(x4)
        return x5


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = ProjectionLayer(1, 3, 16)
        self.layer2 = ProjectionLayer(3, 3, 16)
        self.layer3 = ProjectionLayer(3, 3, 16)
        self.layer4 = ProjectionLayer(3, 1, 16)



    def forward(self, x):
        x1 = self.layer1(x)
        x1 = F.tanh(x1)
        x2 = self.layer2(x1)
        x2 = F.tanh(x2)
        x3 = self.layer3(x2)
        x3 = F.tanh(x3)
        x4 = self.layer4(x3)

        return x, x1, x2, x3, x4

def train():
    model = Net()
    out = model(img)

    #show_grid(out)

    # grid_label = label_to_grid(label)
    # show_grid(grid_label.unsqueeze(1)[:4])

    model.to('cuda')
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9,0.999), weight_decay=1e-5)
    sch = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[5, 10, 15], gamma=0.5)
    criteria = nn.MSELoss()

    for e in range(20):
        losses = []
        for x, y in train_dataloader:
            x = x.to('cuda')
            y_grid = label_to_grid(y)
            y_grid = y_grid.to('cuda')

            x, x1, x2, x3, out = model(x)
            loss = criteria(out, y_grid)
            losses.append(loss.item())
            opt.zero_grad()
            loss.backward()
            opt.step()
        sch.step()

        print(f'epoch {e:03d} train loss: {np.mean(losses)}')

    torch.save(model.state_dict(), 'model/move-net.pth')

#train()

def test():
    model = Net()
    model.load_state_dict(torch.load('model/move-net.pth'))
    model.eval()

    x, y = next(iter(valid_dataloader))
    x, x1, x2, x3, out = model(x)

    show_grid([x, x1, x2, x3, out])

test()



