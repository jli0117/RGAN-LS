### https://github.com/aymericdamien/TensorFlow-Examples/blob/master/tensorflow_v1/examples/3_NeuralNetworks/gan.py
### https://github.com/jsyoon0823/TimeGAN

from __future__ import division, print_function, absolute_import

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from utils import random_generator, batch_generator_with_time


def rnn_cell(module_name, hidden_dim):

    assert module_name in ['gru','lstm']
    
    # GRU
    if (module_name == 'gru'):
        rnn_cell = tf.nn.rnn_cell.GRUCell(num_units=hidden_dim, activation=tf.nn.tanh)

    # LSTM
    elif (module_name == 'lstm'):
        rnn_cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_dim, activation=tf.nn.tanh)

    return rnn_cell


def rganls(train_x, epochs, batch_size, hidden_dim=16, num_layers=3, \
           module_name='gru', z_dim=10, learning_rate=0.0002, dname='bzm', expname=''):

    tf.reset_default_graph() 

    # Network Parameters
    seq_len = train_x.shape[1]
    dim = train_x.shape[2]
    time= [seq_len]*(train_x.shape[0])

    # Input place holders
    X = tf.placeholder(tf.float32, [None, seq_len, dim], name = "myinput_x")
    Z = tf.placeholder(tf.float32, [None, seq_len, z_dim], name = "myinput_z")
    T = tf.placeholder(tf.int32, [None], name = "myinput_t")

    def generator (Z, T):     
        with tf.variable_scope("generator", reuse = tf.AUTO_REUSE):
            e_cell = tf.nn.rnn_cell.MultiRNNCell([rnn_cell(module_name, hidden_dim) for _ in range(num_layers)])
            e_outputs, e_last_states = tf.nn.dynamic_rnn(e_cell, Z, dtype=tf.float32, sequence_length = T)
            E = tf.contrib.layers.fully_connected(e_outputs, dim, activation_fn=tf.nn.sigmoid)     
        return E

    def discriminator (X, T):   
        with tf.variable_scope("discriminator", reuse = tf.AUTO_REUSE):
            d_cell = tf.nn.rnn_cell.MultiRNNCell([rnn_cell(module_name, hidden_dim) for _ in range(num_layers)])
            d_outputs, d_last_states = tf.nn.dynamic_rnn(d_cell, X, dtype=tf.float32, sequence_length = T)
            Y_hat = tf.contrib.layers.fully_connected(d_outputs, 1, activation_fn=None) 
        return Y_hat  


    # Generator
    X_gen = generator(Z, T)

    # Discriminator
    Y_fake = discriminator(X_gen, T)
    Y_real = discriminator(X, T)     

    # Variables        
    g_vars = [v for v in tf.trainable_variables() if v.name.startswith('generator')]
    d_vars = [v for v in tf.trainable_variables() if v.name.startswith('discriminator')]
    
    # Discriminator loss
    D_loss_real = tf.losses.sigmoid_cross_entropy(tf.ones_like(Y_real), Y_real)
    D_loss_fake = tf.losses.sigmoid_cross_entropy(tf.zeros_like(Y_fake), Y_fake)
    D_loss = D_loss_real + D_loss_fake
                
    # Generator loss
    # 1. Adversarial loss
    G_loss_adv = tf.losses.sigmoid_cross_entropy(tf.ones_like(Y_fake), Y_fake)

    # 2. Momments matching loss
    G_loss_V1 = tf.reduce_mean(tf.abs(tf.sqrt(tf.nn.moments(X_gen,[0])[1] + 1e-6) - tf.sqrt(tf.nn.moments(X,[0])[1] + 1e-6)))
    G_loss_V2 = tf.reduce_mean(tf.abs((tf.nn.moments(X_gen,[0])[0]) - (tf.nn.moments(X,[0])[0])))
    G_loss_mm = 100*(G_loss_V1 + G_loss_V2)

        
    # 3. Summation
    G_loss = G_loss_adv + G_loss_mm 

    # optimizer
    D_solver = tf.train.AdamOptimizer(learning_rate).minimize(D_loss, var_list = d_vars)
    G_solver = tf.train.AdamOptimizer(learning_rate).minimize(G_loss, var_list = g_vars)      

    # Initialization
    init = tf.global_variables_initializer()  

    # Start training
    with tf.Session() as sess:

        # Run the initializer
        sess.run(init)

        d_loss_plt = []
        g_loss_plt = []

        for i in range(1, epochs+1):
            # train generator
            for kk in range(2):
                X_mb, T_mb = batch_generator_with_time(train_x, time, batch_size)
                Z_mb = random_generator(batch_size, z_dim, T_mb, seq_len)
                _, step_g_loss, step_g_loss_adv, step_g_loss_mm = \
                    sess.run([G_solver, G_loss, G_loss_adv, G_loss_mm], feed_dict={Z: Z_mb, X: X_mb, T: T_mb})

            # train discriminator
            X_mb, T_mb = batch_generator_with_time(train_x, time, batch_size)           
            Z_mb = random_generator(batch_size, z_dim, T_mb, seq_len)
            check_d_loss = sess.run(D_loss, feed_dict={X: X_mb, T: T_mb, Z: Z_mb})
            # Train discriminator (only when the discriminator does not work well)
            if (check_d_loss > 0.15):        
                _, step_d_loss = sess.run([D_solver, D_loss], feed_dict={X: X_mb, T: T_mb, Z: Z_mb})
            
            d_loss_plt.append(step_d_loss)
            g_loss_plt.append(step_g_loss)

            # If at saving epoches => save generated samples
            if ((i % 1000 == 0)|(i == (epochs-1))):
                print("\nrunning epoch: ", i)
                gen_num = 2000
                Z_gen = random_generator(gen_num, z_dim, [seq_len]*gen_num, seq_len)
                syn_data = sess.run(X_gen, feed_dict={Z: Z_gen, T: [seq_len]*gen_num})

                savepath='gan_ckpt/'+dname+'/'+expname
                if not os.path.exists(savepath):
                    os.makedirs(savepath)
                np.savez(savepath+'/syn_epoch_%d.npz'%i, syn_data=syn_data)

        # display loss of discriminator and generator
        plt.plot(np.arange(0,epochs), d_loss_plt,'r--',label='loss for discriminator')
        plt.plot(np.arange(0,epochs), g_loss_plt,'g--',label='loss for generator')
        plt.title('Loss for training gan')
        plt.xlabel('train epoches')
        plt.ylabel('loss')
        plt.legend(['loss for discriminator', 'loss for generator'])
        plt.savefig(savepath+'/GANs training losses.png')
        plt.show()
        plt.close()

