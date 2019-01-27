"""
Code modified from: https://github.com/chengshengchan/model_compression
"""
import tensorflow as tf
import numpy as np
import os, sys
import argparse
from functions.cifar10.net import lenet, nin

def parse_args():
    parser = argparse.ArgumentParser(description='teacher-student model')    
    parser.add_argument('--p_fake', dest='p_fake', default=0.8, help='probability of training on GAN data', type=float)
    parser.add_argument('--model_path', dest='model_path', default='models/cifar10/netG_keras.h5', help="path to trained GAN model.", type=str)
    parser.add_argument('--lr', dest='lr', default=1e-4, help='learning rate', type=float)
    parser.add_argument('--epoch', dest='epoch', default=200, help='total epoch', type=int)
    parser.add_argument('--dropout', dest='dropout', default=0.5, help="dropout ratio", type=float)
    parser.add_argument('--batch_size', dest='batch_size', default=256, help="batch size", type=int)
    parser.add_argument('--gpu', dest='gpu', default=0, help="which gpu to use", type=int)    
    args = parser.parse_args()
    return args, parser

def main():
    # Parameters
    lr = args.lr
    model_path = args.model_path
    total_epoch = args.epoch
    batch_size = args.batch_size
    dropout_rate = args.dropout
    p_fake = args.p_fake
    
    # Placeholders
    x = tf.placeholder(tf.float32, [batch_size, dim, dim, 3])
    keep_prob = tf.placeholder(tf.float32) 
    
    # Load Data
    (data, label), (data_test, label_test) = tf.keras.datasets.cifar10.load_data()
    mean = np.mean(data, axis=0)
    index = np.array(range(len(data))) 
    iterations = len(data)/batch_size    
    
    # Load Model and Basic Settings
    teacher=nin(x, keep_prob)
    student=lenet(x, keep_prob)
    tf_loss = tf.nn.l2_loss(teacher - student)/batch_size
    optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(tf_loss)
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
    sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True))
    tf.global_variables_initializer().run()
    with tf.device('/cpu:0'):
        saver = tf.train.Saver(max_to_keep=100)

    # Train
    print('Start Training')
    for i in xrange(total_epoch):
        np.random.shuffle(index)
        cost_sum = 0
        total = 0
        
        # Generate GAN data
        if p_fake > 0:
            data_acgan = generate_fake(int(p_fake*len(data)), model_path)            
            j_acgan = 0
            index_acgan = np.array(range(len(data_acgan))) 
            np.random.shuffle(index_acgan)           
                  
        for j in xrange(iterations):
            if np.random.rand() > p_fake: # Train on real training data 
                batch_x = data[index[j*batch_size:(j+1)*batch_size]]
            else: # Train on GAN data
                if (j_acgan+1)*batch_size < len(data_acgan):
                    batch_x = data_acgan[index_acgan[j_acgan*batch_size:(j_acgan+1)*batch_size]]
                    j_acgan += 1
                else:
                    j_rand = np.random.randint(j_acgan)
                    batch_x = data_acgan[index_acgan[j_rand*batch_size:(j_rand+1)*batch_size]]         
            batch_x = np.float32(batch_x) - mean
            _, cost = sess.run([optimizer, tf_loss],
                               feed_dict={x : batch_x, keep_prob : 1-dropout_rate})      
            total += batch_size                     
            cost_sum += cost
        print ("Epoch %d || Training cost = %.2f"%(i, cost_sum/iterations/n_classes))

        
    # Test
    pred = tf.nn.softmax(student)
    total = 0
    correct = 0
    cost_test = 0
    iterations_test = len(data_test)/batch_size
    for j in xrange(iterations_test):
        batch_x = data_test[j*batch_size:(j+1)*batch_size] - mean
        prob, cost = sess.run([pred, tf_loss],
                feed_dict={x : batch_x, keep_prob : 1.0})
        label_batch = label_test[j*batch_size:(j+1)*batch_size].reshape(-1)
        pred_batch = np.array( [np.argmax(prob[i]) for i in range(prob.shape[0])])
        correct += sum(label_batch == pred_batch)
        total += batch_size
        cost_test += cost
    print ("\nEnd of Training\nTest acc = %.4f || Test cost = %.2f\n"%(float(correct)/total, cost_test/iterations_test/n_classes))


if __name__ == '__main__':
    args, parser = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    
    # Parameters for CIFAR-10
    dim = 32
    n_classes = 10
    
    # Functions to read Keras or Pytorch Models
    if 'keras' in args.model_path:
        from functions.cifar10.generate_fake_keras import generate_fake
    elif 'pytorch' in args.model_path:
        from functions.cifar10.generate_fake_pytorch import generate_fake
    else:
        sys.exit('ERROR: model_path is not valid. Default must include - keras / pytorch - in the model name. New model please refer to the code and make corresponding modification.')
    
    main()
