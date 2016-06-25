from keras.layers import Input, merge, Activation, Dense
from keras.models import Model
import utils
import numpy as np


def batch_generator(cpl, lbl):
    import random

    # trim the tail
    garbage = len(labels) % batch_size

    data_pvt = cpl[:, 0][:-garbage]
    data_ctx = cpl[:, 1][:-garbage]
    data_lbl = lbl[:-garbage]

    assert data_pvt.shape == data_ctx.shape == data_lbl.shape

    while 1:
        # shuffle data at beginning of every epoch (takes few minutes)
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
# - sentences: list of (list of id)
# - index2word: list of string
sentences, index2word = utils.load_sentences_brown()

# params
nb_epoch = 3
# learn `batch_size words` at a time
batch_size = 6000
vec_dim = 128
# half of window
window_size = 8
vocab_size = len(index2word)

# create input
couples, labels = utils.skip_grams(sentences, window_size, vocab_size)
print 'shape of couples: ', couples.shape
print 'shape of labels: ', labels.shape

# metrics
nb_batch = len(labels) // batch_size
samples_per_epoch = batch_size * nb_batch

# graph definition (pvt: center of window, ctx: context)
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
model.fit_generator(generator=batch_generator(couples, labels),
                    samples_per_epoch=samples_per_epoch,
                    nb_epoch=nb_epoch, verbose=1)

# save weights
utils.save_weights(model, index2word, vec_dim)

# eval using gensim
print 'the....'
utils.most_similar(positive=['the'])
print 'she - he + him....'
utils.most_similar(positive=['she', 'him'], negative=['he'])
