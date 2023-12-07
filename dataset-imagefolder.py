import torch
from torchvision import transforms
from torchvision import datasets
import matplotlib.pyplot as plt

dataset_root = 'D:/ImageNet1K/val'
normalize = transforms.Normalize(mean=[0., 0., 0.], std=[1, 1, 1])
transf = transforms.Compose([transforms.RandomResizedCrop(224),
                             transforms.RandomHorizontalFlip(0.5),
                             transforms.ToTensor(),
                             normalize])
target_transf = transforms.Lambda(lambda y: torch.zeros(size=(10)).scatter_(dim=0, index=torch.tensor(y), value=1))
dataset = datasets.ImageFolder(root=dataset_root, transform=transf)

idx = 30000
print(dataset[idx][0].shape)
print(torch.min(dataset[idx][0]), torch.max(dataset[idx][0]))
print(dataset[idx][1])

plt.imshow(dataset[idx][0].permute(1,2,0).to('cpu'))
plt.title(str(dataset[idx][1]))
plt.show()