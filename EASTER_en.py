import os
import random

import numpy as np
import pandas as pd
import tensorflow as tf
from abc import ABC
from typing import List
from keras.api._v2.keras import Model
from keras.api._v2.keras import layers
from keras.api._v2.keras import optimizers
from keras.api._v2.keras.utils import Sequence
from keras.api._v2.keras.callbacks import ModelCheckpoint
from transformers import RobertaTokenizerFast, TFRobertaForSequenceClassification
from sklearn.preprocessing import OneHotEncoder
from config import *


class DataGenerator(ABC, Sequence):
    tokenizer = RobertaTokenizerFast.from_pretrained(ROBERTA_PATH)

    def __init__(self, texts, labels):
        max_len = 0
        input_ids = []
        attention_mask = []
        self.encoder = OneHotEncoder().fit([[-1], [0], [1]])
        for text in texts:
            t = DataGenerator.tokenizer(text, return_tensors='tf')
            input_ids.append(t['input_ids'].numpy())
            attention_mask.append(t['attention_mask'].numpy())
            max_len = max(len(input_ids[-1][0]), max_len)
        self.max_len = max_len
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = self.transform_label(labels)

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, batch_ids):
        return {'input_ids': self.input_ids[batch_ids], 'attention_mask': self.attention_mask[batch_ids]}, \
               np.array([self.labels[batch_ids]])

    def _pad_sequences(seq: List[np.ndarray], padding=0, padding_method='same'):
        padded_seq = []
        for index in range(len(seq)):
            t = seq[index]
            pad_l = padding - t.shape[0]
            if pad_l > 0:
                pad_t = np.array([t[-1]])
                if padding_method == 'one':
                    pad_t = np.ones(pad_t.shape, pad_t.dtype)
                elif padding_method == 'zero':
                    pad_t = np.zeros(pad_t.shape, pad_t.dtype)
                pad_t = np.concatenate([pad_t for _ in range(pad_l)])
                t = np.concatenate((t, pad_t))
            padded_seq.append(t)
        return padded_seq

    def pad_to(self, length):
        self.input_ids = [i.reshape((1, -1)) for i in
                          self._pad_sequences([i[0] for i in self.input_ids], length, 'zero')]
        self.attention_mask = [i.reshape((1, -1)) for i in
                               self._pad_sequences([i[0] for i in self.attention_mask], length, 'same')]
        self.max_len = max(self.max_len, length)
        return self

    def transform_label(self, labels):
        return self.encoder.transform(np.asarray(labels).reshape((-1, 1))).toarray()


def build_model(kernel_size, filters, strides, units, include_text_cnn=True):
    # load roberta model
    roberta = TFRobertaForSequenceClassification.from_pretrained(ROBERTA_PATH)

    # define model inputs
    inputs = {'input_ids': layers.Input((None,), dtype=tf.int32),
              'attention_mask': layers.Input((None,), dtype=tf.int32)}

    # Using the first layer of the Roberta model to process input data.
    roberta_main = roberta.layers[0](inputs)[0]

    # one road
    if include_text_cnn:
        conv_outputs = []
        for size, filter_, stride in zip(kernel_size, filters, strides):
            x = layers.Conv1D(filter_, size, stride, activation='relu')(roberta_main)
            x = layers.GlobalMaxPool1D()(x)
            conv_outputs.append(x)
        x = layers.concatenate(conv_outputs)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(units, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        output_0 = layers.Dense(3, activation='relu')(x)


    # another road
    output_1 = roberta.layers[1](roberta_main)

    if include_text_cnn:
        x = layers.concatenate([output_0, output_1])
    else:
        x = output_1

    outputs = layers.Dense(3, activation='softmax')(x)
    model_ = Model(inputs, outputs)
    model_.compile(optimizers.Adam(5e-6), 'categorical_crossentropy', ['accuracy'])
    return model_


def train_model(target: str, model_name: str, model: Model, epochs):
    train_data = pd.read_csv(f'{BASE_DIR}/data/sentiment_acos/{target}_train.csv')
    train_data = DataGenerator(train_data['text'], train_data['sentiment'])
    valid_data = pd.read_csv(f'{BASE_DIR}/data/sentiment_acos/{target}_dev.csv')
    valid_data = DataGenerator(valid_data['text'], valid_data['sentiment'])
    callbacks = [
        ModelCheckpoint(f'{BASE_DIR}/data/pretrained/{model_name}.h5', 'val_accuracy', 1, True, True)
    ]
    model.fit(train_data, epochs=epochs, validation_data=valid_data, callbacks=callbacks, verbose=1)
    return model


def load_model(model_name: str, kernel_size, filters, strides, units, include_text_cnn):
    model = build_model(kernel_size, filters, strides, units, include_text_cnn)
    model.load_weights(f'{BASE_DIR}/data/pretrained/{model_name}.h5')
    return model


def evaluate_model(target: str, model: Model):
    test_data = pd.read_csv(f'{BASE_DIR}/data/sentiment_acos/{target}_test.csv')
    real_data = list(test_data['sentiment'])
    test_data = DataGenerator(test_data['text'], test_data['sentiment'])
    pred_data = model.predict(test_data)
    pred_data = list(map(lambda x: int(x) - 1, np.where(pred_data == np.max(pred_data, axis=1).reshape(-1, 1))[1]))
    result = dict()
    for sentiment in (-1, 0, 1):
        pred = set(map(lambda y: y[0], filter(lambda x: x[1] == sentiment, enumerate(pred_data))))
        real = set(map(lambda y: y[0], filter(lambda x: x[1] == sentiment, enumerate(real_data))))
        TP = len(list(filter(lambda x: x in pred, real)))
        FP = len(pred) - TP
        FN = len(real) - TP
        result[sentiment] = (TP, FP, FN)
    tag = {-1: 'negative', 0: 'neutral', 1: 'positive', 2: 'overall'}
    result[2] = tuple(map(lambda y: sum(map(lambda x: x[y], result.values())), range(3)))
    print('P,R,F')
    for k in (-1, 0, 1, 2):
        print(tag[k], end=':')
        try:
            P, R = result[k][0] / (result[k][0] + result[k][1]), result[k][0] / (result[k][0] + result[k][2])
            F = (2 * P * R) / (P + R)
        except ZeroDivisionError:
            P,R,F = None,None,None
        print(f'{P},{R},{F}')
    return result

model_params = {
    'laptop': ((2, 2, 3), (300, 200, 200), (2, 1, 1), 150),
    'laptop_split': ((2, 2, 3), (300, 200, 200), (2, 1, 1), 150),
    'rest16': ((2, 2, 3), (200, 200, 150), (1, 1, 1), 150),
    'rest16_split': ((2, 2, 3), (200, 200, 150), (1, 1, 1), 150),
}

epoch = 30

def analysis_sentiment(target_name,model_name,test_sentiment_fname):
    # Train the model and evaluate its performance on the sentiment test data：
    # Default include_text_cnn = True
    include_text_cnn = True
    model = build_model(*model_params[target_name], include_text_cnn)
    train_model(f'{target_name}', model_name, model, epoch)
    best_model = load_model(model_name, *model_params[target_name], include_text_cnn)
    evaluate_model(f'{target_name}', best_model)

    # Use the best model to predict all the AOS test data
    test_text_fname = f'{BASE_DIR}/data/acos/{target_name}_quad_test.tsv'
    with open(test_text_fname, 'r', encoding='utf-8') as f:
        text_list = list(map(lambda x: x.split('\t')[0], f.read().splitlines()))
    sentiment_list = best_model.predict(DataGenerator(text_list, [0 for _ in text_list]))
    sentiment_list = list( map(lambda x: int(x) - 1, np.where(sentiment_list == np.max(sentiment_list, axis=1).reshape(-1, 1))[1]))
    sentiment_df = pd.DataFrame({ 'text': text_list,'sentiment': sentiment_list })
    sentiment_df.to_csv(test_sentiment_fname, index=False)
    return best_model, text_list, sentiment_list


if __name__ == "__main__":
    """
    first step work，train model，predict sentiment
    """
    target_name = "laptop"
    model_name = f'{target_name}_model'
    test_sentiment_fname = f'{BASE_DIR}/data/pred_senti/{target_name}_test_sentiment.csv'
    analysis_sentiment(target_name,model_name,test_sentiment_fname);

