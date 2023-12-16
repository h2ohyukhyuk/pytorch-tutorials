import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms

from torch.utils.tensorboard import SummaryWriter

import os
import numpy as np
import io
from datetime import datetime

def print_to_string(*args, **kwargs):
    output = io.StringIO()
    print(*args, file=output, **kwargs)
    contents = output.getvalue()
    output.close()
    return contents

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_ex1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, padding=2, stride=2),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=32, out_channels=48, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=48),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(inplace=True)
        )
        self.feature_ex2 = nn.Sequential(
            nn.Conv2d(in_channels=48, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=64, out_channels=80, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=80),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=80, out_channels=80, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(num_features=80),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=80, out_channels=10, kernel_size=3, padding=1)
        )

    def forward(self, x):
        f1 = self.feature_ex1(x)
        f2 = self.feature_ex2(f1)
        logits = torch.mean(f2, dim=(2,3))
        return logits


os.makedirs('runs/multi-gpu/%s', exist_ok=True)
summary = SummaryWriter('runs/multi-gpu/%s' % datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))

num_epoch = 15
device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

tf_train = transforms.Compose(  [transforms.RandomHorizontalFlip(),
                                transforms.RandomRotation(30),
                                transforms.ToTensor(),
                                 transforms.Normalize(mean=0.5, std=0.5)])

tf_test = transforms.Compose(   [transforms.ToTensor(),
                                transforms.Normalize(mean=0.5, std=0.5)])

train_data = MNIST( root='../data', train=True, transform=tf_train, download=True)
train_data_loader = DataLoader(train_data, batch_size=64, shuffle=True, drop_last=True)
test_data = MNIST( root='../data', train=False, transform=tf_test, download=True)
test_data_loader = DataLoader(test_data, batch_size=64, shuffle=False)

ce_loss = nn.CrossEntropyLoss()

model = Net()
model.to(device)

summary.add_text(tag='log', text_string=print_to_string(model), global_step=None)

opt = torch.optim.AdamW(
    [{'params' : model.feature_ex1.parameters(), 'lr': 1e-2},
     {'params' : model.feature_ex2.parameters()}],
    lr=1e-3, weight_decay=1e-4)

lr_sch_exp = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.99)
lr_sch_ms = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[5, 10], gamma=0.1)

images, gt = next(iter(test_data_loader))
summary.add_graph(model, images.to(device))

for epoch in range(num_epoch):

    summary.add_scalars('lr', {'group1': lr_sch_ms.get_last_lr()[0], 'group2': lr_sch_ms.get_last_lr()[1]}, global_step=epoch)

    model.train()
    train_losses = []
    for i, (input, target) in enumerate(train_data_loader):
        input, target = input.to(device), target.to(device)
        out = model(input)

        loss = ce_loss(out, target)
        train_losses.append(loss.item())

        opt.zero_grad()
        loss.backward()
        opt.step()
        if i > 20:
            break

    lr_sch_exp.step()
    lr_sch_ms.step()

    summary.add_images('train_images', input / 2 + 0.5, global_step=epoch)

    model.eval()
    val_losses = []
    top5s = []
    top1s = []
    for i, (input, target) in enumerate(test_data_loader):
        input, target = input.to(device), target.to(device)
        out = model(input)

        loss = ce_loss(out, target)
        val_losses.append(loss.item())

        top_logits, top_idxs = torch.topk(out, k=5, dim=1, largest=True, sorted=True)
        true_positive = (top_idxs == target.unsqueeze(1)).to(torch.float).sum(dim=0) # batch x 5 -> batch
        top5_cnt = true_positive.sum() # batch -> 1
        top1_cnt = true_positive[0] # batch -> 1
        top5s.append(top5_cnt.item())
        top1s.append(top1_cnt.item())

    summary.add_images('val_images', input / 2 + 0.5, global_step=epoch)
    summary.add_scalars('loss', {'train': np.mean(train_losses), 'valid': np.mean(val_losses)}, global_step=epoch)
    num_test_sam = len(test_data_loader.dataset)
    summary.add_scalars('acc', {'top1': sum(top1s)/num_test_sam, 'top5': sum(top5s)/num_test_sam}, global_step=epoch)
    print(epoch)

summary.close()