import keras
from keras import backend as K
from keras.engine.topology import  Layer
from keras.models import Model
from keras.layers import Input
from keras.layers.core import Dense
from keras.optimizers import Adam, RMSprop
from keras.callbacks import TensorBoard
from keras.regularizers import l2
from keras.initializers import RandomNormal
from keras.layers.merge import _Merge
import numpy as np

BATCH_SIZE = 64
TRAINING_RATIO = 5
GRADIENT_PENALTY_WEIGHT = 10

regrate = 1e-7

###Popular activation
def swish(x):
    beta = 1.5
    return beta * x * keras.backend.sigmoid(x)

###Some losses
def multiply_loss(y_true, y_pred):
    return K.mean(y_true*y_pred)

def mean_loss(y_true, y_pred):
    return K.mean(y_pred)

def gradPenaltyLoss(y_true, y_pred, averaged_samples, gradient_penalty_weight):
    gradients = K.gradients(y_pred, averaged_samples)[0]
    gradients_sqr = K.square(gradients)
    gradients_sqr_sum = K.sum(gradients_sqr, axis=np.arange(1, len(gradients_sqr.shape)))
    gradients_l2_norm = K.sqrt(gradients_sqr_sum)
    gradient_penalty = gradient_penalty_weight * K.square(1 - gradients_l2_norm)
    return K.mean(gradient_penalty)

class randomWeightedAverage(_Merge):
    def _merge_function(self, inputs):
        weights = K.random_uniform((BATCH_SIZE, 1, 1, 1))
        return (weights * inputs[0]) + ((1 - weights) * inputs[1])


###GradNorm layer
class GradNorm(Layer):
    def __init__(self, **kwards):
        super(GradNorm, self).__init__(**kwards)

    def build(self, input_shapes):
        super(GradNorm, self).build(input_shapes)

    def call(self, inputs):
        target, wrt = inputs
        grads = K.gradients(target, wrt)
        assert len(grads) == 1
        grad = grads[0]
        return K.sqrt(K.sum(K.batch_flatten(K.square(grad)), axis=1, keepdims=True))

    def compute_output_shape(self, input_shapes):
        return (input_shapes[1][0],1)



###Freezing/Unfreezing model weights
def set_trainability(model, trainable=False):
    model.trainable = trainable
    for layer in model.layers:
        layer.trainable = trainable

###Generator model
def make_generator(G_in, out_dim, lr=1e-3):
    x = Dense(50, activation=swish, kernel_initializer='he_uniform', kernel_regularizer=l2(regrate))(G_in)
    x = Dense(50, activation=swish, kernel_initializer='he_uniform', kernel_regularizer=l2(regrate))(x)
    x = Dense(50, activation=swish, kernel_initializer='he_uniform', kernel_regularizer=l2(regrate))(x)
    x = Dense(50, activation=swish, kernel_initializer='he_uniform', kernel_regularizer=l2(regrate))(x)
    x = Dense(50, activation=swish, kernel_initializer='he_uniform', kernel_regularizer=l2(regrate))(x)
    x = Dense(50, activation=swish, kernel_initializer='he_uniform', kernel_regularizer=l2(regrate))(x)
    x = Dense(50, activation=swish, kernel_initializer='he_uniform', kernel_regularizer=l2(regrate))(x)
    x = Dense(50, activation=swish, kernel_initializer='he_uniform', kernel_regularizer=l2(regrate))(x)
    x = Dense(50, activation=swish, kernel_initializer='he_uniform', kernel_regularizer=l2(regrate))(x)
    G_out = Dense(out_dim, activation='linear', kernel_initializer='he_uniform')(x)
    G = Model(G_in, G_out)
#    G.compile(loss='mae', optimizer=RMSprop(lr))
    return G, G_out


weight_init = RandomNormal(mean=0., stddev=0.02)

from keras.constraints import Constraint
class WeightClip(Constraint):
    '''Clips the weights incident to each hidden unit to be inside a range
    '''
    def __init__(self, c=2):
        self.c = c

    def __call__(self, p):
        return K.clip(p, -self.c, self.c)

    def get_config(self):
        return {'name': self.__class__.__name__,
                'c': self.c}

clipvalue=0.01
###Discriminator model
def make_discriminator(D_in, lr=1e-3):
    x = Dense(20, activation=swish, kernel_initializer=weight_init, kernel_regularizer=l2(regrate), W_constraint = WeightClip(clipvalue))(D_in)
    x = Dense(20, activation=swish, kernel_initializer=weight_init, kernel_regularizer=l2(regrate), W_constraint = WeightClip(clipvalue))(x)
    x = Dense(20, activation=swish, kernel_initializer=weight_init, kernel_regularizer=l2(regrate), W_constraint = WeightClip(clipvalue))(x)
    D_out = Dense(1, activation='linear', kernel_initializer=weight_init, W_constraint = WeightClip(clipvalue))(x)
    D = Model(D_in, D_out)
#    D.compile(loss=multiply_loss, optimizer=RMSprop(lr))
    return D, D_out

###The chained model i.e. GAN
def make_gan(GAN_in, G, D):
    set_trainability(D, False)
    x = G(GAN_in)
    GAN_out = D(x)
    GAN = Model(GAN_in, GAN_out)
    GAN.compile(loss=multiply_loss, optimizer=G.optimizer)
    return GAN, GAN_out


