from keras.models import Model
from keras.layers import Embedding, Dense, LSTM, Input
from keras import optimizers
from keras.callbacks import ModelCheckpoint
import keras
import numpy as np
from data import Vocab
from batcher import Batcher


class Seq2SeqSummarizer(object):

    def __init__(self, hps):
        self._hps = hps
        vocab = Vocab(hps['vocab_path'], hps['vocab_size'])
        self._vocab = vocab
        self._version = 0
        self._model_name = 'Seq2Seq'

        embedding_init = keras.initializers.TruncatedNormal(mean=0.0, stddev=0.05, seed=None)
        embedding_layer = Embedding(hps['vocab_size'], hps['hidden_dim'], embeddings_initializer=embedding_init)

        encoder_inputs = Input(shape=(None,), name='encoder_inputs')
        encoder_lstm = LSTM(hps['hidden_dim'], return_state=True, name='encoder_lstm')
        encoder_outputs, state_h, state_c = encoder_lstm(embedding_layer(encoder_inputs))
        encoder_states = [state_h, state_c]
        decoder_inputs = Input(shape=(None,), name='decoder_inputs')
        decoder_lstm = LSTM(hps['hidden_dim'], return_state=True, return_sequences=True, name='decoder_lstm')
        decoder_outputs, _, _ = decoder_lstm(embedding_layer(decoder_inputs), initial_state=encoder_states)
        decoder_dense = Dense(hps['vocab_size'], activation='softmax', name='decoder_dense')
        dense_outputs = decoder_dense(decoder_outputs)

        model = Model([encoder_inputs, decoder_inputs], dense_outputs)
        adam = optimizers.Adam()
        model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

        self.model = model

    
    def make_feed_dict(self, batch, just_enc=False):
        feed_dict = dict()
        feed_dict['enc_batch'] = batch.enc_batch
        feed_dict['dec_batch'] = batch.dec_batch
        feed_dict['target_batch'] = np.eye(hps['vocab_size'])[batch.target_batch]
        return feed_dict

    def generate_batch(self, mode): #mode: train/test/val
        hps = self._hps
        hps['mode'] = mode
        batcher = Batcher(hps['data_path'] + '/{}.bin'.format(mode), self._vocab, hps, single_pass=True)
        while True:
            batch = batcher.next_batch()
            feed_dict = self.make_feed_dict(batch)
            yield [feed_dict['enc_batch'], feed_dict['dec_batch']], feed_dict['target_batch']
        
    def load_weights(self, weight_file_path):
        if os.path.exists(weight_file_path):
            self.model.load_weights(weight_file_path)
        
    def fit(self, load_weights=True, epochs=100, batch_size=None, model_dir_path=None):
        hps = self._hps
        weight_file_path = hps['model_path'] + 'weights.txt'
        if load_weights:
            self.model.load_weights(weight_file_path)
        if model_dir_path is None:
            model_dir_path = hps['model_path']
        if batch_size is None:
            batch_size = hps['batch_size']

        self._version += 1
        checkpoint = ModelCheckpoint(weight_file_path)
        architecture_file_path = hps['model_path'] + 'archit.txt'
        open(architecture_file_path, 'w').write(self.model.to_json())
        
        history = self.model.fit_generator(generator=self.generate_batch('train'), steps_per_epoch=hps['batch_size'],
                                           epochs=epochs,
                                           verbose=1, validation_data=self.generate_batch('val'), validation_steps=hps['batch_size'],
                                           callbacks=[checkpoint])
        self.model.save_weights(weight_file_path)
        return history

    def load_weights(self, weight_file_path):
        if os.path.exists(weight_file_path):
            self.model.load_weights(weight_file_path)

