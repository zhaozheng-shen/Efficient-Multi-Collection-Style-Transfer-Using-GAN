import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import random

from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import glob
import random
import os
import torch
from torch.autograd import Variable
from torchvision.transforms import InterpolationMode

def generate_patch(image, patch_size):
  corp_x = random.randint(0,image.shape[0]-patch_size)
  corp_y = random.randint(0,image.shape[1] - patch_size)
  patch = image[corp_x:corp_x+patch_size,corp_y:corp_y + patch_size,:]
  return patch

def generate_permutation(image,patch_size,patch_num):
  output = np.zeros((patch_size*patch_num,patch_size*patch_num,3))
  for i in range(patch_num):
    for j in range(patch_num):
      output[i*patch_size: i*patch_size + patch_size, j*patch_size: j*patch_size + patch_size,:] = generate_patch(image,patch_size)
  return output

num_permu = 20
patch_size = 8
patch_num = 16

imgs = ['Van-Gogh-The-Starry-Night.jpg', 'Mountain-No.2-Jay-DeFeo.jpg', 'Qi-Baishi-FengYeHanChan.jpg', 'Robert-Delaunay-Portrait-de-Metzinger.jpg']
paths = ['vangogh', 'jaydefeo', 'qibaishi', 'robertD']
reshape_size = [250, 300, 260, 340]

for k in range(len(imgs)):
    vangogh = Image.open('./style/' + imgs[k])
    van = vangogh.getdata()
    van = np.array(van)
    van = van.reshape(reshape_size[k],-1,3)
    for i in range(num_permu):
        p = generate_permutation(van,patch_size,patch_num)
        f = Image.fromarray(p.astype(np.uint8))
        if not os.path.exists('./Gated_patch_voc2007/'):
          os.mkdir('./Gated_patch_voc2007/')
        if not os.path.exists('./Gated_patch_voc2007/trainB/'):
          os.mkdir('./Gated_patch_voc2007/trainB/')
        if not os.path.exists('./Gated_patch_voc2007/trainB/' + paths[k]):
          os.mkdir('./Gated_patch_voc2007/trainB/' + paths[k])
        path = "./Gated_patch_voc2007/trainB/" + paths[k] + "/{:04d}.jpg".format(i)
        f.save(path)


import torch
from torchvision.datasets import VOCDetection
from torch.utils.data import Dataset, ConcatDataset

dataset_train = VOCDetection('./', year="2007", image_set="trainval", download=True)
dataset_test = VOCDetection('./', year="2007", image_set="test", download=True)
dataset = torch.utils.data.ConcatDataset([dataset_train, dataset_test])

