import torch
import torchvision.models as models

model = models.vgg16(weights='IMAGENET1K_V1')
model_state_dic = model.state_dict()

for n, w in model_state_dic.items():
    print(n, w.shape, w.dtype)

torch.save(model_state_dic, 'data/vgg16.pt')

model_loaded = models.vgg16() # we do not specify ``weights``, i.e. create untrained model
model_loaded.load_state_dict(torch.load('data/vgg16.pt'))
print(model_loaded.eval())