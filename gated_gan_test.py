import torch 
import glob
import random
import os
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import sys
import torch.nn as nn
import torch.nn.functional as F

import torchvision
from torch.utils.data import Dataset
from ipywidgets import interact, widgets
import matplotlib.pyplot as plt 
from PIL import Image


from models import *

loadSize = 143
fineSize = 128
ngf = 64
ndf = 64    
in_nc = 3 
out_nc = 3 
cuda=False

gen = Generator(in_nc,out_nc,4,ngf)
gen.load_state_dict(torch.load('./output/netG.pth',map_location='cpu'))

transforms_ = [ transforms.Resize(int(128), Image.BICUBIC), 
        transforms.RandomCrop(128), 
        #transforms.RandomVerticalFlip(p=0),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) 
      ]

transform = transforms.Compose(transforms_)

def tensor2image(tensor):
    image = 127.5*(tensor[0].cpu().float().detach().numpy() + 1.0)
    return image.astype(np.uint8)

@interact(
          vango=widgets.FloatSlider(min=0,max=1,step=.01,value=0,continuous_update=False)
          ,ukyo=widgets.FloatSlider(min=0,max=1,step=.01,value=0,continuous_update=False)
          ,monet=widgets.FloatSlider(min=0,max=1,step=.01,value=0,continuous_update=False)
          ,cezan=widgets.FloatSlider(min=0,max=1,step=.01,value=0,continuous_update=False)
          ,ident=widgets.FloatSlider(min=.01,max=1,step=.01,value=1,continuous_update=False)
)
def generate_image(path='./', filename = '', flip90=False):
    generator=gen
    content = transform(Image.open(path))
    Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor
    input_A = Tensor(1, 3, 128, 128)
    real_A = input_A.copy_(content)
    
    for i in range(4):
        style_label = torch.tensor([i])
        style_OHE = F.one_hot(style_label, 4).long()
        
        generated = gen({
            'content': real_A,
            'style_label': style_OHE
        })
        im=tensor2image(generated.data)
        if flip90:
            im=im.transpose(2,1,0)
        else:
            im=im.transpose(1,2,0)
        fig = plt.figure(figsize=(6,6))
        ax = fig.add_subplot(111)  
        ax.imshow(im)
        f = Image.fromarray(im.astype(np.uint8))
        if not os.path.exists('./test_output/'):
            os.mkdir('./test_output/')
        if not os.path.exists('./test_output/gated_permutaion/'):
            os.mkdir('./test_output/gated_permutaion/')
        f.save('./test_output/gated_permutaion/' + filename.split('.')[0] + "_{:01d}.jpg".format(i))

from os import walk
filenames = next(walk('./test_data/'), (None, None, []))[2]
for f in filenames:
    generate_image('./test_data/' + f, f)

