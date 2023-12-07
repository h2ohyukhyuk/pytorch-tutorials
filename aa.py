
import os
basedir = 'D:/ImageNet1K/train'

for fn in os.listdir(basedir):
    if not os.path.isdir(os.path.join(basedir, fn)):
        continue

    os.rename(os.path.join(basedir, fn),
              os.path.join(basedir,f"{int(fn):04d}"))
