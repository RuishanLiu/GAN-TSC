"""
Code modified from: https://github.com/King-Of-Knights/Keras-ACGAN-CIFAR10
"""
from keras.models import load_model
import numpy as np
import tensorflow as tf

def generate_gan(n, model_path):
    '''
    Generate balanced synthetic data from trained Keras GAN model
    Parameters:
        - n: number of total GAN data generated
        - model_path: path to trained Keras model
    '''
    class_num = 2
    latent_size = 100
    batch_size = 256   
    def gen_acgan_ind(ind):
        '''Generate ACGAN Images for class ind from batch_size'''
        sampled_labels = ind*np.ones(batch_size)
        noise = np.random.normal(0, 0.5, (batch_size, latent_size))
        generated_images = netG.predict([noise, sampled_labels.reshape((-1, 1))], verbose=0)
        fake = np.float32(generated_images)
        return fake
    def gen_acgan_n_ind(ind, n_ind):
        '''Generate n_ind ACGAN Images for class ind'''
        data_ind = gen_acgan_ind(ind)
        while data_ind.shape[0] < n_ind:
            data_ind = np.concatenate([data_ind, gen_acgan_ind(ind)], axis=0)
        return data_ind[:n_ind]  
    netG = load_model(model_path)
    data_acgan = np.concatenate([gen_acgan_n_ind(ind, int(n/class_num)) for ind in range(class_num)], axis=0) 
    return data_acgan
                
def generate_munge(X, y, times, s=1, p=0.5):
    label_fake = np.concatenate([y for _ in range(times)], axis=0)
    X_fake = []
    for i_times in range(times):
        X_t = X.copy()
        for i in range(X_t.shape[0]):
            ei = X_t[i, :]
            j = np.argsort(np.sum((X_t - ei)**2, axis=1))[1] # closest
            ej = X_t[j, :]
            sd = abs(ei - ej)/s
            ind_p = np.random.rand(X_t.shape[1]) < p
            X_t[i, ind_p] = np.random.normal(ej, sd)[ind_p]
            X_t[j, ind_p] = np.random.normal(ei, sd)[ind_p]
        X_fake.append(X_t)
    X_fake = np.concatenate(X_fake, axis=0)
    return X_fake, label_fake
