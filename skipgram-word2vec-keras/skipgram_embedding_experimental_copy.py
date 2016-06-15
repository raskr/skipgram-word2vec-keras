# -*- coding: utf-8 -*-
from keras.layers import Input, merge, Activation, Dense
from keras.models import Model
import utils
import numpy as np
import theano.tensor as T


def batch_generator(couples, labels):
    while 1:
        for i in range(nb_batch):
            begin, end = batch_size*i, batch_size*(i+1)
            pvt = couples[:, 0][begin: end]
            ctx = couples[:, 1][begin: end]
            lbl = labels[begin: end]

            # create large 2d array for Dense
            pvt_dst = np.zeros((batch_size, vocab_size), dtype=np.float32)
            ctx_dst = np.zeros((batch_size, vocab_size), dtype=np.float32)
            for j, (p, c) in enumerate(zip(pvt, ctx)):
                pvt_dst[j][p] = 1
                ctx_dst[j][c] = 1

            yield ([pvt_dst, ctx_dst], lbl)

# load data
sentences, index2word = utils.load_sentences_brown(5000)

# params
nb_epoch = 10
batch_size = 1000
vec_dim = 128
window_size = 7
vocab_size = len(index2word)

# create input
couples, labels = utils.skip_grams(sentences, window_size, vocab_size)
# data_pivot, data_context, data_label = utils.create_input(couples, labels, batch_size)
# assert data_pivot.shape == data_context.shape == data_label.shape

# metrics
nb_batch = len(labels) // batch_size
samples_per_epoch = batch_size * nb_batch

# graph definition (so slow model)
input_pvt = Input(batch_shape=(batch_size, vocab_size,))
input_ctx = Input(batch_shape=(batch_size, vocab_size,))

embedded_pvt = Dense(input_dim=vocab_size,
                     output_dim=vec_dim)(input_pvt)

embedded_ctx = Dense(input_dim=vocab_size,
                     output_dim=vec_dim)(input_ctx)

merged = merge(inputs=[embedded_pvt, embedded_ctx],
               mode=lambda a: (a[0]*a[1]).sum(-1).reshape((batch_size, 1)),
               # mode=lambda a: T.tensordot(a[0], a[1]),
               output_shape=(batch_size, 1))

predictions = Activation('sigmoid')(merged)


# build and train the model
model = Model(input=[input_pvt, input_ctx], output=predictions)
model.compile(optimizer='rmsprop', loss='mse', metrics=['accuracy'])
gen = batch_generator(couples, labels)
model.fit_generator(generator=gen,
                    samples_per_epoch=samples_per_epoch,
                    nb_epoch=nb_epoch, verbose=1)

# save_weight
utils.save_weights(model, index2word, vec_dim)

# eval using gensim
utils.most_similar(positive=['she', 'him'], negative=['he'])