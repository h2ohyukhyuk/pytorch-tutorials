# https://pytorch.org/tutorials/beginner/vt_tutorial.html

from PIL import Image
import torch
import timm
import requests
import torchvision.transforms as transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torch.utils.mobile_optimizer import optimize_for_mobile

print(torch.__version__)
# should be 1.8.0

model = torch.hub.load('facebookresearch/deit:main', 'deit_base_patch16_224', pretrained=True)
model.eval()

transform = transforms.Compose([
    transforms.Resize(256, interpolation=3),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
])

img = Image.open(requests.get("https://raw.githubusercontent.com/pytorch/ios-demo-app/master/HelloWorld/HelloWorld/HelloWorld/image.png", stream=True).raw)
img = transform(img)[None,]
out = model(img)

clsidx = torch.argmax(out)
print('float model predict:', clsidx.item())

#torch.onnx.export(model, torch.zeros((1,3,224,224)), 'model.onnx')

scripted_model = torch.jit.script(model)
scripted_model.save("model/fbdeit_scripted.pt")

backend = 'x86'
model.qconfig = torch.quantization.get_default_qat_qconfig(backend)
torch.backends.quantized.engine = backend

quantized_model = torch.quantization.quantize_dynamic(model, qconfig_spec={torch.nn.Linear}, dtype=torch.qint8)
scripted_quantized_model = torch.jit.script(quantized_model)
scripted_quantized_model.save("model/fbdeit_scripted_quantized.pt")

out = scripted_quantized_model(img)
clsidx = torch.argmax(out)
print('quantized model predict: ', clsidx.item())
# The same output 269 should be printed

optimized_scripted_quantized_model = optimize_for_mobile(scripted_quantized_model)
optimized_scripted_quantized_model.save("model/fbdeit_optimized_scripted_quantized.pt")

out = optimized_scripted_quantized_model(img)
clsidx = torch.argmax(out)
print('quantized optimized model predict: ',clsidx.item())
# Again, the same output 269 should be printed

optimized_scripted_quantized_model._save_for_lite_interpreter("model/fbdeit_optimized_scripted_quantized_lite.ptl")
ptl = torch.jit.load("model/fbdeit_optimized_scripted_quantized_lite.ptl")
out = ptl(img)
clsidx = torch.argmax(out)
print('quantized optimized lite model predict: ',clsidx.item())

with torch.autograd.profiler.profile(use_cuda=False) as prof1:
    out = model(img)
with torch.autograd.profiler.profile(use_cuda=False) as prof2:
    out = scripted_model(img)
with torch.autograd.profiler.profile(use_cuda=False) as prof3:
    out = scripted_quantized_model(img)
with torch.autograd.profiler.profile(use_cuda=False) as prof4:
    out = optimized_scripted_quantized_model(img)
with torch.autograd.profiler.profile(use_cuda=False) as prof5:
    out = ptl(img)

print("original model: {:.2f}ms".format(prof1.self_cpu_time_total/1000))
print("scripted model: {:.2f}ms".format(prof2.self_cpu_time_total/1000))
print("scripted & quantized model: {:.2f}ms".format(prof3.self_cpu_time_total/1000))
print("scripted & quantized & optimized model: {:.2f}ms".format(prof4.self_cpu_time_total/1000))
print("lite model: {:.2f}ms".format(prof5.self_cpu_time_total/1000))


