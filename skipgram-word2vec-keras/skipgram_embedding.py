from keras.layers import Input, merge, Activation
from keras.models import Model
from keras.layers.embeddings import Embedding
import utils
import numpy as np


def create_input(x, y, seq_len):
    """
    :param x: couples from output of skip_grams(). i.e. [[1, 2], [1, 3], [2, 1], ...]
    :return: two numpy array. (centers, others) pair. i.e. [[1, 1, 2], [2, 3, 1]]
    """
    x_ = np.array(x).swapaxes(0, 1)
    y_ = np.array(y)
    garbage = len(x_[0]) % seq_len
    final_shape = (len(x_[0][:-garbage]) / seq_len, seq_len)
    return x_[0][:-garbage].reshape(final_shape), \
           x_[1][:-garbage].reshape(final_shape), \
           y_[:-garbage].reshape(final_shape)


def batch_generator():
    while 1:
        for i in range(nb_batch):
            yield (
                [data_pivot[batch_size*i: batch_size*(i+1), :],
                 data_ctx[batch_size*i: batch_size*(i+1), :]],
                labels[batch_size*i: batch_size*(i+1), :],
            )

# load data
sentences, index2word, word2index = utils.load_sentences_brown(3)

# params
batch_size = 2
seq_len = 5
vec_dim = 100
window_size = 5
vocab_size = len(index2word)

# create input
couples, labels = utils.skip_grams(sentences, window_size, vocab_size)
data_pivot, data_ctx, labels = create_input(couples, labels, seq_len)

# metrics
nb_batch = len(data_pivot) // batch_size
samples_per_epoch = batch_size * nb_batch

# graph definition
input_pivot = Input(shape=(seq_len,), dtype='int32')
input_ctx = Input(shape=(seq_len,), dtype='int32')

embedded_pivot = Embedding(input_dim=vocab_size,
                           output_dim=vec_dim,
                           input_length=seq_len)(input_pivot)

embedded_ctx = Embedding(input_dim=vocab_size,
                         output_dim=vec_dim,
                         input_length=seq_len)(input_ctx)

merged = merge(inputs=[embedded_pivot, embedded_ctx],
               mode=lambda a: (a[0] * a[1]).sum(2),
               output_shape=(batch_size, seq_len,))

predictions = Activation('sigmoid')(merged)

# build and train the model
model = Model(input=[input_ctx, input_pivot], output=predictions)
model.compile(optimizer='rmsprop', loss='mse', metrics=['accuracy'])
model.fit_generator(generator=batch_generator(), samples_per_epoch=samples_per_epoch,
                    nb_epoch=10, verbose=1)

# save weights
utils.save_weights(model, index2word, vocab_size, vec_dim)

# eval
utils.similar_words_of('great')
