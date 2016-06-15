from keras.layers import Input, merge, Activation, Dense
from keras.models import Model
import utils
import numpy as np


def batch_generator(couples, labels):
    import random

    garbage = len(labels) % batch_size
    data_pvt = couples[:, 0][:-garbage]
    data_ctx = couples[:, 1][:-garbage]
    data_lbl = labels[:-garbage]
    assert data_pvt.shape == data_ctx.shape == data_lbl.shape

    while 1:
        # shuffle data at beginning of every epoch
        seed = random.randint(0, 10e6)
        random.seed(seed)
        random.shuffle(data_pvt)
        random.seed(seed)
        random.shuffle(data_ctx)
        random.seed(seed)
        random.shuffle(data_lbl)

        # feed batches actually
        for i in range(nb_batch):
            # slice mini-batch data
            begin, end = batch_size*i, batch_size*(i+1)
            pvt = data_pvt[begin: end]
            ctx = data_ctx[begin: end]
            lbl = data_lbl[begin: end]

            # convert those into large 2d array for `Dense`
            pvt_dst = np.zeros((batch_size, vocab_size), dtype=np.float32)
            ctx_dst = np.zeros((batch_size, vocab_size), dtype=np.float32)
            for j, (p, c) in enumerate(zip(pvt, ctx)):
                pvt_dst[j][p] = 1
                ctx_dst[j][c] = 1

            yield ([pvt_dst, ctx_dst], lbl)

# load data
sentences, index2word = utils.load_sentences_brown(5000)

# params
nb_epoch = 3
batch_size = 1000
vec_dim = 128
window_size = 7
vocab_size = len(index2word)

# create input
couples, labels = utils.skip_grams(sentences, window_size, vocab_size)

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
               output_shape=(batch_size, 1))

predictions = Activation('sigmoid')(merged)


# build and train the model
model = Model(input=[input_pvt, input_ctx], output=predictions)
model.compile(optimizer='rmsprop', loss='mse', metrics=['accuracy'])
gen = batch_generator(couples, labels)
model.fit_generator(generator=gen,
                    samples_per_epoch=samples_per_epoch,
                    nb_epoch=nb_epoch, verbose=1)

# save weights
utils.save_weights(model, index2word, vec_dim)

# eval using gensim
utils.most_similar(positive=['she', 'him'], negative=['he']) #=> her