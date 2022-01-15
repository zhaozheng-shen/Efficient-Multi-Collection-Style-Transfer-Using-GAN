import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import glob
import random
import os
import torch
from torch.autograd import Variable
from torchvision.transforms import InterpolationMode
from models import *


class ImageDataset(Dataset):
    def __init__(self, root_img, root_style, transforms_=None, mode='train'):
        transforms_ = [ transforms.Resize(int(143), InterpolationMode.BICUBIC), 
                transforms.RandomCrop(128), 
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) 
              ]
        #content source
        self.transform = transforms.Compose(transforms_)
        self.X = sorted(glob.glob(os.path.join(root_img, '*')))
        #style image source(s)
        self.Y = []
        style_sources = sorted(glob.glob(os.path.join(root_style, '*')))
        for label, style in enumerate(style_sources):
            temp = [(label, x) for x in sorted(glob.glob(style_sources[label]+"/*"))]
            self.Y.extend(temp)
    def __len__(self):
        return max(len(self.X), len(self.Y))
    def __getitem__(self, index):                                    
        output = {}
        output['content'] = self.transform(Image.open(self.X[index % len(self.X)]))
        #select style
        selection = self.Y[random.randint(0, len(self.Y) - 1)]
        try:
            output['style'] = self.transform(Image.open(selection[1]))
        except:
            selection = self.Y[random.randint(0, len(self.Y) - 1)]
            output['style'] = self.transform(Image.open(selection[1]))
            # print('thisuns grey')
            # print(selection)
        output['style_label'] = selection[0]
        return output

from torch.utils.data import DataLoader
root = ImageDataset('./VOCdevkit/VOC2007/JPEGImages', './Gated_patch_voc2007/trainB')
print(len(root.X), len(root.Y), root.transform)

dataloader = DataLoader(root, batch_size=1, shuffle=True, num_workers=2)
batch = next(iter(dataloader))

class ReplayBuffer():
    def __init__(self, max_size=50):
        assert (max_size > 0), 'Empty buffer or trying to create a black hole. Be careful.'
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0,1) > 0.5:
                    i = random.randint(0, self.max_size-1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return Variable(torch.cat(to_return))

def label2tensor(label,tensor):
    for i in range(label.size(0)):
        tensor[i].fill_(label[i])
    return tensor

# def tensor2image(tensor):
#     image = 127.5*(tensor[0].cpu().float().numpy() + 1.0)
#     if image.shape[0] == 1:
#         image = np.tile(image, (3,1,1))
#     return image.astype(np.uint8)

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant(m.bias.data, 0.0)

class LambdaLR():
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert ((n_epochs - decay_start_epoch) > 0), "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch)/(self.n_epochs - self.decay_start_epoch)

#TRAIN OPTIONS FROM GATED GAN
epoch = 0
n_epochs = 3
decay_epoch=1
batchSize = 1
dataroot = './photo2fourcollection'
loadSize = 143
fineSize = 128
ngf = 64
ndf = 64    
in_nc = 3 
out_nc = 3 
lr = 0.0002 
gpu = 1 
lambda_A = 10.0
pool_size = 50
resize_or_crop = 'resize_and_crop'
autoencoder_constrain = 10 
n_styles = 4
cuda=False
tv_strength=1e-6

generator = Generator(in_nc, out_nc, n_styles, ngf)
discriminator= Discriminator(in_nc, n_styles, ndf)

if cuda:
    generator.cuda()
    discriminator.cuda()

#Losses Init
use_lsgan=True
if use_lsgan:
    criterion_GAN = nn.MSELoss()
else: 
    criterion_GAN = nn.BCELoss()
    
    
criterion_ACGAN = nn.CrossEntropyLoss()
criterion_Rec = nn.L1Loss()
criterion_TV = TVLoss(TVLoss_weight=tv_strength)

#Optimizers & LR schedulers
optimizer_G = torch.optim.Adam(generator.parameters(),
                                lr=lr, betas=(0.5, 0.999))
optimizer_D = torch.optim.Adam(discriminator.parameters(), 
                               lr=lr, betas=(0.5, 0.999))


lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=LambdaLR(n_epochs, epoch,decay_epoch).step)
lr_scheduler_D = torch.optim.lr_scheduler.LambdaLR(optimizer_D, lr_lambda=LambdaLR(n_epochs,epoch, decay_epoch).step)

#Set vars for training
Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor
input_A = Tensor(batchSize, in_nc, fineSize, fineSize)
input_B = Tensor(batchSize, out_nc, fineSize, fineSize)
target_real = Variable(Tensor(batchSize).fill_(1.0), requires_grad=False)
target_fake = Variable(Tensor(batchSize).fill_(0.0), requires_grad=False)

D_A_size = discriminator(input_A.copy_(batch['style']))[0].size()  
D_AC_size = discriminator(input_B.copy_(batch['style']))[1].size()

class_label_B = Tensor(D_AC_size[0],D_AC_size[1],D_AC_size[2]).long()

autoflag_OHE = Tensor(1,n_styles+1).fill_(0).long()
autoflag_OHE[0][-1] = 1

fake_label = Tensor(D_A_size).fill_(0.0)
real_label = Tensor(D_A_size).fill_(0.99) 

rec_A_AE = Tensor(batchSize,in_nc,fineSize,fineSize)

fake_buffer = ReplayBuffer()

##Init Weights
generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)



### TRAIN LOOP
for epoch in range(epoch, n_epochs):
    for i, batch in enumerate(dataloader):
        ## Unpack minibatch
        # source content
        real_content = Variable(input_A.copy_(batch['content']))
        # target style
        real_style = Variable(input_B.copy_(batch['style']))
        # style label
        style_label = batch['style_label']
        # one-hot encoded style
        style_OHE = F.one_hot(style_label,n_styles).long()
        # style Label mapped over 1x19x19 tensor for patch discriminator 
        class_label = class_label_B.copy_(label2tensor(style_label,class_label_B)).long()
        
        #### Update Discriminator
        optimizer_D.zero_grad()
        
        # Generate style-transfered image
        genfake = generator({
            'content':real_content,
            'style_label': style_OHE})
        
        # Add generated image to image pool and randomly sample pool 
        fake = fake_buffer.push_and_pop(genfake)
        # Discriminator forward pass with sampled fake 
        out_gan, out_class = discriminator(fake)
        # Discriminator Fake loss (correctly identify generated images)
        errD_fake = criterion_GAN(out_gan, fake_label)
        # Backward pass and parameter optimization
        errD_fake.backward()
        optimizer_D.step()
        
        optimizer_D.zero_grad()
        # Discriminator forward pass with target style
        out_gan, out_class = discriminator(real_style)
        # Discriminator Style Classification loss
        errD_real_class = criterion_ACGAN(out_class.transpose(1,3),class_label)*lambda_A
        # Discriminator Real loss (correctly identify real style images)
        errD_real = criterion_GAN(out_gan, real_label)        
        errD_real_total = errD_real + errD_real_class
        # Backward pass and parameter optimization
        errD_real_total.backward()
        optimizer_D.step()
        
        
        errD = (errD_real+errD_fake)/2.0
        
                
        #### Generator Update
        ## Style Transfer Loss
        optimizer_G.zero_grad()
        
        # Discriminator forward pass with generated style transfer
        out_gan, out_class = discriminator(genfake)
        
        # Generator gan (real/fake) loss
        err_gan = criterion_GAN(out_gan, real_label)
        # Generator style class loss
        err_class = criterion_ACGAN(out_class.transpose(1,3), class_label)*lambda_A
        # Total Variation loss
        err_TV = criterion_TV(genfake)
        
        errG_tot = err_gan + err_class + err_TV
        errG_tot.backward()
        optimizer_G.step()
        
        ## Auto-Encoder (Recreation) Loss
        optimizer_G.zero_grad()
        identity = generator({
            'content': real_content,
            'style_label': autoflag_OHE,
        })
        err_ae = criterion_Rec(identity,real_content)*autoencoder_constrain
        err_ae.backward()
        optimizer_G.step()
        if i % 20 == 0:
            print('Batch:', i)
            print("Discriminator loss:", errD_real_total.item(), "Generator loss:", errG_tot.item())

        
    
    ##update learning rates
    lr_scheduler_G.step()
    lr_scheduler_D.step()
    
    #Save model
    torch.save(generator.state_dict(), 'output/netG.pth')
    torch.save(discriminator.state_dict(), 'output/netD.pth')