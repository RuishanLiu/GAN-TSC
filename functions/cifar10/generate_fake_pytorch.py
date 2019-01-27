"""
Code modified from: https://github.com/gitlimlab/ACGAN-PyTorch
"""
import argparse
import os
import numpy as np
import random
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from net_GAN_pytorch import _netG
cudnn.benchmark = True

def color_preprocessing(data):
    '''Process GAN output to image data. (Note: Particular to different GAN model.)'''
    data = data.astype('float32')
    data = data.transpose(0, 2, 3, 1)      
    mean = [0.5, 0.5, 0.5]
    std  = [0.5, 0.5, 0.5]        
    for i in range(3):
        data[:,:,:,i] = mean[i] +  std[i] * data[:,:,:,i]
    data = data.copy()*255.
    return data

def generate_fake(n, model_path):
    # Parameters
    num_classes = 10
    batch_size = 1
    ngpu = 1 
    nz = 110 

    # Define the generator and initialize the weights
    netG = _netG(ngpu, nz)
    netG.load_state_dict(torch.load(model_path))
    # Tensor placeholders
    noise = torch.FloatTensor(batch_size, nz, 1, 1)
    aux_label = torch.LongTensor(batch_size)
    # Define variables
    noise = Variable(noise)
    aux_label = Variable(aux_label)
    if torch.cuda.is_available():
        netG.cuda()
        aux_label = aux_label.cuda()
        noise = noise.cuda()
        
    data_fake = []
    for i in xrange(n):
        noise.data.resize_(batch_size, nz, 1, 1).normal_(0, 1)
        label = np.random.randint(0, num_classes, batch_size)
        noise_ = np.random.normal(0, 1, (batch_size, nz))
        class_onehot = np.zeros((batch_size, num_classes))
        class_onehot[np.arange(batch_size), label] = 1
        noise_[np.arange(batch_size), :num_classes] = class_onehot[np.arange(batch_size)]
        noise_ = (torch.from_numpy(noise_))
        noise.data.copy_(noise_.view(batch_size, nz, 1, 1))
        aux_label.data.resize_(batch_size).copy_(torch.from_numpy(label))
        fake = netG(noise)
        fake_data = fake.data.cpu().numpy()
        fake_image = color_preprocessing(fake_data)
        data_fake.append(fake_image)
    data_fake = np.concatenate(data_fake, axis=0)
    data_fake = np.asarray(data_fake.astype(np.uint8))
    return data_fake   