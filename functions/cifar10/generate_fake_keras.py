"""
Code modified from: https://github.com/King-Of-Knights/Keras-ACGAN-CIFAR10
"""
from keras.models import load_model
import numpy as np
import tensorflow as tf

def generate_fake(n, model_path):
    '''
    Generate balanced synthetic data from trained Keras GAN model
    Parameters:
        - n: number of total GAN data generated
        - model_path: path to trained Keras model
    '''
    class_num = 10
    latent_size = 110
    batch_size = 256   
    def gen_acgan_ind(ind):
        '''Generate ACGAN Images for class ind from batch_size'''
        sampled_labels = ind*np.ones(batch_size)
        noise = np.random.normal(0, 0.5, (batch_size, latent_size))
        generated_images = netG.predict([noise, sampled_labels.reshape((-1, 1))], verbose=0).transpose(0, 2, 3, 1)
        fake = np.asarray((generated_images * 127.5 + 127.5).astype(np.uint8)) 
        return fake
    def gen_acgan_n_ind(ind, n_ind):
        '''Generate n_ind ACGAN Images for class ind'''
        data_ind = gen_acgan_ind(ind)
        while data_ind.shape[0] < n_ind:
            data_ind = np.concatenate([data_ind, gen_acgan_ind(ind)], axis=0)
        return data_ind[:n_ind, :, :, :]   
    netG = load_model(model_path)
    data_acgan = np.concatenate([gen_acgan_n_ind(ind, int(n/class_num)) for ind in range(class_num)], axis=0) 
    return data_acgan
                