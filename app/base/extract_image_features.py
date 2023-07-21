from tensorflow.keras.applications import densenet
from tensorflow.keras.applications.densenet import preprocess_input
from tensorflow.keras.layers import Dense, Dropout, Input, Conv2D
from tensorflow.keras.models import Model
import numpy as np
import pandas as pd
from tqdm import tqdm
import tensorflow as tf
import pickle
from PIL import Image
from skimage.transform import resize
import warnings

warnings.filterwarnings('ignore')


class ImageFeatures:
    def __init__(self):
        self.train_dataset = pd.read_csv('app/data/Train_Data.csv')
        self.test_dataset = pd.read_csv('app/data/Test_Data.csv')
        self.cv_dataset = pd.read_csv('app/data/CV_Data.csv')

    def load_image(self, img_name):
        image = Image.open(img_name)
        X = np.asarray(image.convert("RGB"))
        X = np.asarray(X)
        X = preprocess_input(X)
        X = resize(X, (224, 224, 3))
        X = np.expand_dims(X, axis=0)
        X = np.asarray(X)
        return X

    def create_chexnet_model(self):
        chex = densenet.DenseNet121(include_top=False, weights=None, input_shape=(224, 224, 3))
        X = chex.output
        X = Dense(14, activation="sigmoid", name="predictions")(X)
        model = Model(inputs=chex.input, outputs=X)
        model.load_weights(r"app/data/brucechou1983_CheXNet_Keras_0.3.0_weights.h5")
        return Model(inputs=model.input, outputs=model.layers[-2].output)

    def image_features(self, train, test, cv):
        model = self.create_chexnet_model()
        Xnet_features_attention = {}
        for key, img1, img2, finding in tqdm(train.values):
            i1 = self.load_image(img1)
            img1_features = model.predict(i1)

            i2 = self.load_image(img2)
            img2_features = model.predict(i2)

            input_ = np.concatenate((img1_features, img2_features), axis=2)
            input_ = tf.reshape(input_, (input_.shape[0], -1, input_.shape[-1]))

            Xnet_features_attention[key] = input_

        for key, img1, img2, finding in tqdm(test.values):
            i1 = self.load_image(img1)
            img1_features = model.predict(i1)

            i2 = self.load_image(img2)
            img2_features = model.predict(i2)

            input_ = np.concatenate((img1_features, img2_features), axis=2)
            input_ = tf.reshape(input_, (input_.shape[0], -1, input_.shape[-1]))

            Xnet_features_attention[key] = input_

        for key, img1, img2, finding in tqdm(cv.values):
            i1 = self.load_image(img1)
            img1_features = model.predict(i1)

            i2 = self.load_image(img2)
            img2_features = model.predict(i2)

            input_ = np.concatenate((img1_features, img2_features), axis=2)
            input_ = tf.reshape(input_, (input_.shape[0], -1, input_.shape[-1]))

            Xnet_features_attention[key] = input_

        return Xnet_features_attention

    def dump_features_to_pickle(self):
        Xnet_features_attention = self.image_features(self.train_dataset, self.test_dataset, self.cv_dataset)
        with open(r'app/data/Image_features_attention.pickle', 'wb') as f:
            pickle.dump(Xnet_features_attention, f)
