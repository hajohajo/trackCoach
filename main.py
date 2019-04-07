import argparse
import os
import numpy as np
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Reshape, Flatten
from keras.layers.merge import _Merge
from keras.optimizers import Adam
from keras import backend as K
from functools import partial
from keras.regularizers import l2
import pandas as pd
import root_pandas
import keras
from sklearn.preprocessing import StandardScaler
import shutil
import math
from plotting import plot_losses, plot_distribution

BATCH_SIZE = 64
TRAINING_RATIO = 10
GRADIENT_PENALTY_WEIGHT = 10
REGRATE = 1e-7
EPOCHS=10000

input_columns = ['trk_pt', 'trk_eta', 'trk_phi']
output_columns = ['trk_dxyClosestPV', 'trk_dzClosestPV', 'trk_ptErr', 'trk_etaErr',
                  'trk_dxyErr', 'trk_dzErr', 'trk_nChi2']

def swish(x):
    beta = 1.5
    return beta * x * keras.backend.sigmoid(x)

def wasserstein_loss(y_true, y_pred):
    return K.mean(y_true * y_pred)

def gradient_penalty_loss(y_true, y_pred, averaged_samples, gradient_penalty_weight):
    gradients = K.gradients(y_pred, averaged_samples)[0]
    gradients_sqr = K.square(gradients)
    gradients_sqr_sum = K.sum(gradients_sqr, axis=np.arange(1, len(gradients_sqr.shape)))
    gradient_l2_norm = K.sqrt(gradients_sqr_sum)
    gradient_penalty = gradient_penalty_weight * K.square(1 - gradient_l2_norm)
    return K.mean(gradient_penalty)

def make_generator(inp_dim):
    model = Sequential()
    model.add(Dense(100, activation=swish, kernel_initializer='he_uniform', kernel_regularizer=l2(REGRATE), input_dim=inp_dim))
    model.add(Dense(100, activation=swish, kernel_initializer='he_uniform', kernel_regularizer=l2(REGRATE)))
    model.add(Dense(100, activation=swish, kernel_initializer='he_uniform', kernel_regularizer=l2(REGRATE)))
    model.add(Dense(100, activation=swish, kernel_initializer='he_uniform', kernel_regularizer=l2(REGRATE)))
    model.add(Dense(len(output_columns), activation='linear', kernel_initializer='he_uniform', kernel_regularizer=l2(REGRATE)))
    return model

def make_discriminator(inp_dim):
    model = Sequential()
    model.add(Dense(10, activation=swish, kernel_initializer='he_uniform', kernel_regularizer=l2(REGRATE), input_dim=inp_dim))
    model.add(Dense(10, activation=swish, kernel_initializer='he_uniform', kernel_regularizer=l2(REGRATE)))
    model.add(Dense(10, activation=swish, kernel_initializer='he_uniform', kernel_regularizer=l2(REGRATE)))
    model.add(Dense(10, activation=swish, kernel_initializer='he_uniform', kernel_regularizer=l2(REGRATE)))
    model.add(Dense(10, activation=swish, kernel_initializer='he_uniform', kernel_regularizer=l2(REGRATE)))
    model.add(Dense(10, activation=swish, kernel_initializer='he_uniform', kernel_regularizer=l2(REGRATE)))
    model.add(Dense(10, activation=swish, kernel_initializer='he_uniform', kernel_regularizer=l2(REGRATE)))
    model.add(Dense(10, activation=swish, kernel_initializer='he_uniform', kernel_regularizer=l2(REGRATE)))
    model.add(Dense(1, activation='linear', kernel_initializer='he_uniform', kernel_regularizer=l2(REGRATE)))
    return model


class RandomWeightedAverage(_Merge):
    def _merge_function(self, inputs):
        weights = K.random_uniform((BATCH_SIZE, 1, 1, 1))
        return (weights * inputs[0]) + ((1 - weights) * inputs[1])

def set_trainability(model, trainable=False):
    model.trainable = trainable
    for layer in model.layers:
        layer.trainable = trainable

def generate_input(n_samples):
    pT = np.random.uniform(0.0, 10.0, n_samples)
    eta = np.random.uniform(-2.1, 2.1, n_samples)
    phi = np.random.uniform(-math.pi, math.pi, n_samples)

    result = np.column_stack((pT, eta, phi))
    return result

def sample_data_and_gen(G, n_samples):
    inp = scaler_inp.transform(generate_input(n_samples/2))
    generated = G.predict(inp)
    generated = scaler_out.inverse_transform(generated)
    generated = np.concatenate((inp, generated), axis=1)
    real = dataframe.sample(n=n_samples/2)[input_columns+output_columns]
    output = pd.DataFrame(np.concatenate((generated, real), axis=0), columns=input_columns+output_columns)
    truth = np.concatenate((np.ones(n_samples/2),-1*np.ones(n_samples/2)))
    output['trk_generated'] = truth

    output = output.sample(frac=1.0)

    return output[input_columns+output_columns], output['trk_generated']

def pretrain(D, nepochs):
    real_input = scaler_out.transform(dataframe[output_columns].sample(frac=1.0))
    noisy_input = scaler_inp.transform(generate_input(real_input.shape[0]))
    positive_y = np.ones((real_input.shape[0], 1), dtype=np.float32)
    negative_y = -positive_y
    dummy_y = np.zeros((real_input.shape[0], 1), dtype=np.float32)

    D.fit([real_input, noisy_input], [positive_y, negative_y, dummy_y], epochs=nepochs)

if __name__ == '__main__':
    ###Recreate plots folders
    folders_ = ['plots']
    for dir in folders_:
        if os.path.exists(dir):
            shutil.rmtree(dir)
        os.makedirs(dir)

    ###Only on

    temp = ['trk_isTrue', 'trk_algo']
    input_file = "data/trackingNtuple_TTBarLeptons.root"
    dataframe = root_pandas.read_root(input_file, columns=input_columns + output_columns + temp, flatten=True)[
        input_columns + output_columns + temp]
    dataframe = dataframe.astype("float64")
    dataframe = dataframe[(dataframe.trk_isTrue == 1) & (dataframe.trk_algo == 4) & (dataframe.trk_pt <= 10.0)]
    dataframe = dataframe.drop(temp, axis=1)

    scaler_out = StandardScaler()
    scaler_out.fit(dataframe[output_columns])
    scaler_inp = StandardScaler()
    scaler_inp.fit(dataframe[input_columns])

    generator = make_generator(len(input_columns))
    discriminator = make_discriminator(len(output_columns))

    set_trainability(discriminator, False)

    generator_input = Input(shape=[len(input_columns)])
    generator_layers = generator(generator_input)
    discriminator_layers_for_generator = discriminator(generator_layers)
    generator_model = Model(inputs=[generator_input],
                            outputs=[discriminator_layers_for_generator])
    generator_model.compile(optimizer=Adam(1e-4, beta_1=0.5, beta_2=0.9), loss=wasserstein_loss)

    set_trainability(discriminator, True)
    set_trainability(generator, False)

    real_samples = Input(shape=[len(output_columns)])
    generator_input_for_discriminator = Input(shape=[len(input_columns)])
    generated_samples_for_discriminator = generator(generator_input_for_discriminator)
    discriminator_output_from_generator = discriminator(generated_samples_for_discriminator)
    discriminator_output_from_real_samples = discriminator(real_samples)

    averaged_samples = RandomWeightedAverage()([real_samples, generated_samples_for_discriminator])
    averaged_samples_out = discriminator(averaged_samples)

    partial_gp_loss = partial(gradient_penalty_loss, averaged_samples=averaged_samples,
                              gradient_penalty_weight=GRADIENT_PENALTY_WEIGHT)
    partial_gp_loss.__name__ = 'gradient_penalty'

    discriminator_model = Model(inputs=[real_samples, generator_input_for_discriminator],
                                outputs=[discriminator_output_from_real_samples,
                                         discriminator_output_from_generator,
                                         averaged_samples_out])
    discriminator_model.compile(optimizer=Adam(1e-4, beta_1=0.5, beta_2=0.9),
                                loss=[wasserstein_loss,
                                      wasserstein_loss,
                                      partial_gp_loss])
    positive_y = np.ones((BATCH_SIZE, 1), dtype=np.float32)
    negative_y = -positive_y
    dummy_y = np.zeros((BATCH_SIZE, 1), dtype=np.float32)

    pretrain(discriminator_model, 5)

    G_loss = []
    D_loss = []
    v_freq = 10
    dataframe_ = dataframe.sample(frac=1.0)[output_columns]
    for epoch in range(EPOCHS):
        dataframe_ = dataframe_.sample(frac=1.0)
        discriminator_loss = []
        generator_loss = []
        minibatches_size = BATCH_SIZE * TRAINING_RATIO
        for i in range(int(dataframe_.shape[0] // (BATCH_SIZE * TRAINING_RATIO))):
            discriminator_minibatches = scaler_out.transform(dataframe_[i * minibatches_size:(i+1) * minibatches_size])
            for j in range(TRAINING_RATIO):
                input_batch = discriminator_minibatches[j * BATCH_SIZE:(j+1) * BATCH_SIZE]
                noise_batch = scaler_inp.transform(generate_input(BATCH_SIZE))
                discriminator_loss.append(discriminator_model.train_on_batch(
                    [input_batch, noise_batch],
                    [positive_y, negative_y, dummy_y]
                ))
                # discriminator_loss.append(0)
            noise_batch = scaler_inp.transform(generate_input(BATCH_SIZE))
#            generator_loss.append(0)
            generator_loss.append(generator_model.train_on_batch(noise_batch, positive_y))

        G_loss.append(np.mean(generator_loss))
        D_loss.append(np.mean(discriminator_loss))
#        print("Epoch #{}: Generative Loss: {}, Discriminative Loss: {}".format(epoch + 1, G_loss[-1], D_loss[-1]))

        if (epoch + 1) % v_freq == 0:
            print("Epoch #{}: Generative Loss: {}, Discriminative Loss: {}".format(epoch + 1, G_loss[-1], D_loss[-1]))
            plot_losses(G_loss, D_loss)
            X, y = sample_data_and_gen(generator, 2000)
            binning = np.linspace(-0.5,0.5,100)
            for distr in output_columns:
                plot_distribution(X[y==1][distr], binning, epoch=epoch+1, title="Generated_"+distr)
                plot_distribution(X[y==-1][distr], binning, epoch=epoch+1, title="Real_"+distr)