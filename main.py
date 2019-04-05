import os
import shutil
import random
import numpy as np
import math
from keras.engine.input_layer import Input
from keras import backend as K
import pandas as pd
import root_pandas
from sklearn.preprocessing import StandardScaler

from models import set_trainability, make_generator, make_discriminator, make_gan
from plotting import plot_losses, plot_distribution

###Recreate plots folders
folders_=['plots']
for dir in folders_:
        if os.path.exists(dir):
            shutil.rmtree(dir)
        os.makedirs(dir)


###Only on
input_columns = ['trk_pt', 'trk_eta', 'trk_phi']
output_columns = ['trk_dxyClosestPV', 'trk_dzClosestPV', 'trk_ptErr', 'trk_etaErr',
                  'trk_dxyErr', 'trk_dzErr', 'trk_nChi2']
temp = ['trk_isTrue', 'trk_algo']
input_file = "data/trackingNtuple_TTBarLeptons.root"
dataframe = root_pandas.read_root(input_file,columns=input_columns+output_columns+temp, flatten=True)[input_columns+output_columns+temp]
dataframe = dataframe[(dataframe.trk_isTrue==1) & (dataframe.trk_algo==4) & (dataframe.trk_pt<=10.0)]
dataframe = dataframe.drop(temp, axis=1)

scaler = StandardScaler()
scaler.fit(dataframe[output_columns])

def generate_input(n_samples):
    pT = np.random.uniform(0.0, 10.0, n_samples)
    eta = np.random.uniform(-2.1, 2.1, n_samples)
    phi = np.random.uniform(-math.pi, math.pi, n_samples)

    result = np.column_stack((pT, eta, phi))
    return result

def sample_data_and_gen(G, n_samples):
    inp = generate_input(n_samples/2)
    generated = G.predict(inp)
    generated = scaler.inverse_transform(generated)
    generated = np.concatenate((inp, generated), axis=1)
    real = dataframe.sample(n=n_samples/2)
    output = pd.DataFrame(np.concatenate((generated, real), axis=0), columns=input_columns+output_columns)
    truth = np.concatenate((np.ones(n_samples/2),-1*np.ones(n_samples/2)))
    output['trk_generated'] = truth

    output = output.sample(frac=1.0)

    return output[input_columns+output_columns], output['trk_generated']

def pretrain(G, D, n_samples, batch_size=32, epochs=10):
    X, y = sample_data_and_gen(G, n_samples)
    set_trainability(D, True)
#    D.fit(X[output_columns], y, epochs=epochs, batch_size=batch_size, verbose=False)
    D.fit(scaler.transform(X[output_columns]), y, epochs=epochs, batch_size=batch_size, verbose=False)

def train(GAN, G, D, epochs=100, n_samples=50000, batch_size=64, verbose=False, v_freq=10):
    d_iters=10
    D_loss = []
    G_loss = []
    e_range = range(epochs)
    for epoch in e_range:
        d_loss = []
        g_loss = []
        pretrain(G, D, n_samples, batch_size, 20)
        for batch in range(n_samples/batch_size):
            X, y = sample_data_and_gen(G, batch_size)
            set_trainability(D, True)
#            d_loss.append(D.train_on_batch(X[output_columns], y))
            d_loss.append(D.train_on_batch(scaler.transform(X[output_columns]), y))

            set_trainability(D, False)
            X = generate_input(batch_size)
            y = -1*np.ones(batch_size) #Claim these are true tracks, see if discriminator believes
            g_loss.append(GAN.train_on_batch(X, y))

        G_loss.append(np.mean(g_loss))
        D_loss.append(np.mean(d_loss))
        if verbose and (epoch + 1) % v_freq == 0:
            print("Epoch #{}: Generative Loss: {}, Discriminative Loss: {}".format(epoch + 1, G_loss[-1], D_loss[-1]))
            plot_losses(G_loss, D_loss)
            X, y = sample_data_and_gen(G, 2000)
            binning = np.linspace(-0.5,0.5,100)
            for distr in output_columns:
                plot_distribution(X[y==1][distr], binning, epoch=epoch+1, title="Generated_"+distr)
                plot_distribution(X[y==-1][distr], binning, epoch=epoch+1, title="Real_"+distr)

        # if (epoch + 1) % 200 == 0:
        #     print "Old lr: "+ str(K.eval(D.optimizer.lr))
        #     K.set_value(D.optimizer.lr, 0.5*K.eval(D.optimizer.lr))
        #     K.set_value(G.optimizer.lr, 0.5*K.eval(G.optimizer.lr))
        #     print "New lr: "+ str(K.eval(D.optimizer.lr))

    return D_loss, G_loss

if __name__ == '__main__':
    G_in = Input(shape=[len(input_columns)])
    G, G_out = make_generator(G_in, len(output_columns), lr=1e-4)
    D_in = Input(shape=[len(output_columns)])
    D, D_out = make_discriminator(D_in, lr=1e-4)

    GAN_in = Input(shape=[len(input_columns)])
    GAN, GAN_out = make_gan(GAN_in, G, D)

    pretrain(G, D, 10000)
    train(GAN, G, D, 1000, verbose=True)

