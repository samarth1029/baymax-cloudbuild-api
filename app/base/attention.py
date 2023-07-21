import numpy as np
import pandas as pd
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import time
from tensorflow.keras.layers import Dense, LSTM, Input, Embedding, Conv2D, Concatenate, Flatten, Add, Dropout, GRU
import datetime
import warnings
from app.base.encoder import Encoder

warnings.filterwarnings('ignore')


class AttentionModel:
    def __init__(self):
        self.train_dataset = pd.read_csv(r'app/data/Train_Data_colab.csv')
        self.cv_dataset = pd.read_csv(r'app/data/CV_Data_colab.csv')
        self.test_dataset = pd.read_csv(r'app/data/Test_Data_colab.csv')
        self.X_train, self.X_test, self.X_cv = self.train_dataset['Person_id'], self.test_dataset['Person_id'], \
                                               self.cv_dataset['Person_id'][:546]
        self.y_train, self.y_test, self.y_cv = self.train_dataset['Report'], self.test_dataset['Report'], \
                                               self.cv_dataset['Report'][:546]
        self.max_capt_len = 153
        self.pad_size = self.max_capt_len
        self.BATCH_SIZE = 14
        self.BUFFER_SIZE = 500
        self.units = 256
        self.att_units = 10
        self.vocab_size = 1428
        with open('Image_features_attention.pickle', 'rb') as f:
            self.Xnet_Features = pickle.load(f)
        with open('glove.840B.300d.pkl', 'rb') as f:
            self.glove_vectors = pickle.load(f)

    def _tokenize(self):
        tokenizer = Tokenizer(filters='!"#$%&()*+,-/:;<=>?@[\\]^_`{|}~\t\n')
        tokenizer.fit_on_texts(self.y_train.values)

        train_rep_tok = tokenizer.texts_to_sequences(self.y_train)
        cv_rep_tok = tokenizer.texts_to_sequences(self.y_cv)
        test_rep_tok = tokenizer.texts_to_sequences(self.y_test)

        train_rep_padded = pad_sequences(train_rep_tok, maxlen=153, padding='post')
        cv_rep_padded = pad_sequences(cv_rep_tok, maxlen=153, padding='post')
        test_rep_padded = pad_sequences(test_rep_tok, maxlen=153, padding='post')

        tokenizer.word_index['<pad>'] = 0
        tokenizer.index_word[0] = '<pad>'
        return train_rep_padded, cv_rep_padded, tokenizer

    def load_image(self, id_, report):
        """Loads the Image Features with their corresponding Ids"""
        img_feature = self.Xnet_Features[id_.decode('utf-8')][0]
        return img_feature, report

    def create_dataset(self, img_name_train, reps):
        dataset = tf.data.Dataset.from_tensor_slices((img_name_train, reps))
        dataset = dataset.map(
            lambda item1, item2: tf.numpy_function(self.load_image, [item1, item2], [tf.float32, tf.int32]),
            num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.shuffle(500).batch(self.BATCH_SIZE).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        return dataset

    def data_generator(self):
        train_rep_padded, cv_rep_padded, _ = self._tokenize()
        train_generator = self.create_dataset(self.X_train.values, train_rep_padded)
        cv_generator = self.create_dataset(self.X_cv.values, cv_rep_padded)
        return train_generator, cv_generator

    def create_embeddings(self):
        _, _, tokenizer = self._tokenize()
        vocab_size = len(tokenizer.word_index.keys()) + 1
        embedding_matrix = np.zeros((vocab_size, 300))
        for word, i in tokenizer.word_index.items():
            if word in self.glove_vectors.keys():
                vec = self.glove_vectors[word]
                embedding_matrix[i] = vec
            else:
                continue
        return embedding_matrix

    def create_model(self):
        model1 = Attention_Model(self.vocab_size, self.units, self.max_capt_len, self.att_units, self.BATCH_SIZE)
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        model1.compile(optimizer=optimizer, loss=self.maskedLoss)
        return model1

    def maskedLoss(self, y_true, y_pred):
        loss_function = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='auto')
        mask = tf.math.logical_not(tf.math.equal(y_true, 0))
        loss_ = loss_function(y_true, y_pred)
        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ = loss_ * mask
        loss_ = tf.reduce_mean(loss_)
        return loss_

    def train_model(self):
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = 'app/data/Tensorboard/attention_OneStep/fit2/' + current_time + '/train'
        val_log_dir = 'app/data/Tensorboard/attention_OneStep/fit2/' + current_time + '/test'
        train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        val_summary_writer = tf.summary.create_file_writer(val_log_dir)
        return self.run_epochs_and_save_model(EPOCHS=10, train_summary_writer=train_summary_writer,
                                              val_summary_writer=val_summary_writer)

    def run_epochs_and_save_model(self, EPOCHS, train_summary_writer, val_summary_writer):
        epoch_train_loss = []
        epoch_val_loss = []
        train_generator, cv_generator = self.data_generator()
        model1 = self.create_model()
        for epoch in range(EPOCHS):
            start = time.time()
            print("EPOCH: ", epoch + 1)
            batch_loss_tr = 0
            batch_loss_val = 0
            for img, rep in train_generator:
                res = model1.train_on_batch([img, rep[:, :-1]], rep[:, 1:])
                batch_loss_tr += res
            train_loss = batch_loss_tr / (self.X_train.shape[0] / self.BATCH_SIZE)

            with train_summary_writer.as_default():
                tf.summary.scalar('loss', train_loss, step=epoch)

            for img, rep in cv_generator:
                res = model1.test_on_batch([img, rep[:, :-1]], rep[:, 1:])
                batch_loss_val += res
            val_loss = batch_loss_val / (self.X_cv.shape[0] / self.BATCH_SIZE)

            with val_summary_writer.as_default():
                tf.summary.scalar('loss', val_loss, step=epoch)

            epoch_train_loss.append(train_loss)

            epoch_val_loss.append(val_loss)
            print(f'Training Loss: {train_loss},  Validation Loss: {val_loss}')
            print(f'Time Taken for this Epoch : {time.time() - start} sec')
            model1.save_weights(r'app/data/trained_model.h5')
        return model1

    def inference_concat(self, inputs):
        in_ = len(inputs.split()) - 1
        inputs = self.Xnet_Features[inputs]
        enc_state = tf.zeros((1, 256))
        model1 = self.train_model()
        enc_output = model1.layers[0](inputs)
        input_state = enc_state
        pred = []
        _, _, tokenizer = self._tokenize()
        cur_vec = np.array([tokenizer.word_index['startseq']]).reshape(-1, 1)
        for _ in range(153):
            inf_output, input_state, attention_weights = model1.layers[1].onestep_decoder(cur_vec, input_state,
                                                                                          enc_output)

            cur_vec = np.reshape(np.argmax(inf_output), (1, 1))
            if cur_vec[0][0] != 0:
                pred.append(cur_vec)
            else:
                break
        return ' '.join([tokenizer.index_word[e[0][0]] for e in pred if e[0][0] not in [0, 7]])


class OneStepDecoder(tf.keras.layers.Layer):
    def __init__(self, vocab_size, att_units, dec_units):
        super(OneStepDecoder, self).__init__()
        self.vocab_size = vocab_size
        # self.emb_dim = emb_dim
        self.att_units = att_units
        self.dec_units = dec_units

    def build(self, input_shape):
        self.embedding = Embedding(self.vocab_size, output_dim=300, input_length=153, mask_zero=True,
                                   weights=[AttentionModel.create_embeddings()],
                                   name="embedding_layer_decoder")
        self.gru = GRU(self.dec_units, return_sequences=True, return_state=True, name="Decoder_GRU")
        self.fc = Dense(self.vocab_size)

        self.V = Dense(1)
        self.W = Dense(self.att_units)
        self.U = Dense(self.att_units)

    def call(self, dec_input, hidden_state, enc_output):
        hidden_with_time = tf.expand_dims(hidden_state, 1)
        attention_weights = self.V(tf.nn.tanh(self.U(enc_output) + self.W(hidden_with_time)))
        attention_weights = tf.nn.softmax(attention_weights, 1)
        context_vector = attention_weights * enc_output
        context_vector = tf.reduce_sum(context_vector, axis=1)

        x = self.embedding(dec_input)
        x = tf.concat([tf.expand_dims(context_vector, axis=1), x], axis=-1)
        output, h_state = self.gru(x, initial_state=hidden_state)
        output = tf.reshape(output, (-1, output.shape[2]))
        x = self.fc(output)

        return x, h_state, attention_weights


class Decoder(tf.keras.layers.Layer):

    def __init__(self, vocab_size, input_length, dec_units, att_units):
        super(Decoder, self).__init__()
        self.vocab_size = vocab_size
        #    self.embedding_dim = embedding_dim
        self.input_length = input_length
        self.dec_units = dec_units
        self.att_units = att_units
        self.onestep_decoder = OneStepDecoder(self.vocab_size, self.att_units, self.dec_units)

    @tf.function
    def call(self, dec_input, hidden_state, enc_output):
        all_outputs = tf.TensorArray(tf.float32, dec_input.shape[1], name='output_arrays')

        for timestep in range(dec_input.shape[1]):
            output, hidden_state, attention_weights = self.onestep_decoder(dec_input[:, timestep:timestep + 1],
                                                                           hidden_state, enc_output)

            all_outputs = all_outputs.write(timestep, output)

        all_outputs = tf.transpose(all_outputs.stack(), [1, 0, 2])
        return all_outputs


class Attention_Model(tf.keras.Model):
    def __init__(self, vocab, units, max_capt_len, att_units, batch_size):
        super(Attention_Model, self).__init__()
        self.batch_size = batch_size
        self.encoder = Encoder(units)
        self.decoder = Decoder(vocab,max_capt_len, units, att_units)

    def call(self, data):
        enc_input, dec_input = data[0], data[1]
        enc_output = self.encoder(enc_input)
        enc_state = self.encoder.get_states(self.batch_size)
        return self.decoder(dec_input, enc_state, enc_output)


