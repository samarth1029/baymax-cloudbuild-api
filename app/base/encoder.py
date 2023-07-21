import tensorflow as tf
from tensorflow.keras.layers import Dense, LSTM, Input, Embedding, Conv2D,Concatenate,Flatten,Add,Dropout,GRU
import warnings
warnings.filterwarnings('ignore')


class Encoder(tf.keras.layers.Layer):
    def __init__(self, units):
        super(Encoder, self).__init__()
        self.units = units
    # self.bs = batch_size

    def build(self, input_shape):
        self.maxpool = tf.keras.layers.MaxPool1D()
        self.dense = Dense(self.units, kernel_initializer=tf.keras.initializers.glorot_uniform(seed = 56), name='dense_encoder')

    def call(self, input_, training=True):

        x = self.maxpool(input_)
        x = self.dense(x)

        return x

    def get_states(self, bs):

        return tf.zeros((bs, self.units))