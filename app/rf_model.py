#coding:utf8
import os
import glob
import pickle
#import pandas as pd

import numpy as np
import tensorflow as tf


#vocab_size = 50_000

#tokenizer = Tokenizer(num_words=vocab_size, oov_token='<OOV>')
#tokenizer.fit_on_texts(df['diagnostico'])

# alternativa para cargar
class Modelo_LSTM:

    # modelo LSTM
    path = './seed_data'
    l_files = glob.glob(os.path.join(path, "*.pkl"))
    lstm_files = {'encoder': l_files[0], 
                'seedcausa': l_files[1],
                'tokenizer': l_files[3]}
    
    def __init__(self):
        self.modelo    = tf.keras.models.load_model('lstm_diagnostic_model.h5')
        self.tokenizer = pickle.load(open(self.lstm_files['tokenizer'],'rb'))
        self.encoder   = pickle.load(open(self.lstm_files['encoder'],'rb'))


    def prediction(self, texto):
        tokens = self.tokenizer.texts_to_sequences([texto])
        vector = self.modelo.predict(tokens)
        return np.argmax(vector[0])


    class Modelo_BiGru:

        path = './bigru'

        def __init__(self):
            self.modelo =  tf.keras.models.load_model(path)

        def prediction(self, texto):
            new_sequences = tokenizer.texts_to_sequences(texto)
            new_padded_sequences = pad_sequences(new_sequences, maxlen=max_length, padding='post')
            predictions = best_model.predict(new_padded_sequences)
            predicted_classes = np.argmax(predictions, axis=1)
        
            return predicted_classes
        


if __name__=='__main__':
    modelo = Modelo_LSTM()

    # ejemplos
    modelo.prediction('infarto al miocardio')
    modelo.prediction('infarto')
    modelo.prediction('tuvo un infarto')
    modelo.prediction('diabetes')


    tk = modelo.tokenizer.texts_to_sequences(['infarto'])
    vc = modelo.modelo.predict(tk)
    np.argmax(vc[0]), tk

    causas = pd.read_pickle('./seed_data/seedcausa.pkl')

