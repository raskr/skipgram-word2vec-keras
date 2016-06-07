# -*- coding: utf-8 -*-
from keras.layers import Input, merge, Activation, Dense
from keras.models import Model
import utils
import numpy as np
import theano.tensor as T

# load data
sentences, index2word, word2index = utils.load_sentences_brown(3)

# params
batch_size = 5
sent_len = 5
vec_dim = 100
window_size = 5
vocab_size = len(index2word)


def create_input(x, y):
    """
    :param x: couples from output of skip_grams(). i.e. [[1, 2], [1, 3], [2, 1], ...]
    :return: two numpy arrays. (centers, others) pair. i.e. [[1, 1, 2], [2, 3, 1]]
    """
    x_ = np.array(x).swapaxes(0, 1)
    y_ = np.array(y)
    garbage = len(y_) % batch_size
    return x_[0][:-garbage], x_[1][:-garbage], y_[:-garbage]


def batch_generator():
    while 1:
        for i in range(nb_batch):
            pvt = data_pivot[batch_size*i: batch_size*(i+1)]
            ctx = data_ctx[batch_size*i: batch_size*(i+1)]
            y = labels[batch_size*i: batch_size*(i+1)].reshape(5, 1)

            # create large 2d array for pivot and context
            pivot_dst = np.zeros((batch_size, vocab_size), dtype=np.float32)
            ctx_dst = np.zeros((batch_size, vocab_size), dtype=np.float32)

            for j, (p, c) in enumerate(zip(pvt, ctx)):
                pivot_dst[j][p] = 1.0
                ctx_dst[j][c] = 1.0

            yield ([pivot_dst, ctx_dst], y)


# create input
couples, labels = utils.skip_grams(sentences, window_size, vocab_size)
data_pivot, data_ctx, labels = create_input(couples, labels)

# metrics
nb_batch = len(data_pivot) // batch_size
samples_per_epoch = batch_size * nb_batch

# graph definition
input_pivot = Input(shape=(vocab_size,))
input_ctx = Input(shape=(vocab_size,))

embedded_pivot = Dense(input_dim=vocab_size,
                       output_dim=vec_dim)(input_pivot)

embedded_ctx = Dense(input_dim=vocab_size,
                     output_dim=vec_dim)(input_ctx)

merged = merge(inputs=[embedded_pivot, embedded_ctx],
               mode=lambda a: (T.tensordot(a[0], a[1])),
               output_shape=(batch_size, 1))

predictions = Activation('sigmoid')(merged)


# build and train the model
model = Model(input=[input_pivot, input_ctx], output=predictions)
model.compile(optimizer='rmsprop', loss='mse', metrics=['accuracy'])
model.fit_generator(generator=batch_generator(), samples_per_epoch=samples_per_epoch,
                    nb_epoch=10, verbose=1)

# save_weight
utils.save_weights(model, index2word, vocab_size, vec_dim)

# eval
utils.similar_words_of('great')
