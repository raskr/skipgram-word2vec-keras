from keras.layers import Input, merge, Activation
from keras.models import Model
from keras.layers.embeddings import Embedding
import utils


def batch_generator(cpl, lbl):
    import random

    garbage = len(labels) % batch_size

    pvt = cpl[:, 0][:-garbage]
    ctx = cpl[:, 1][:-garbage]
    lbl = lbl[:-garbage]
    assert pvt.shape == ctx.shape == lbl.shape

    while 1:
        # shuffle data at beginning of every epoch
        seed = random.randint(0, 10e6)
        random.seed(seed)
        random.shuffle(pvt)
        random.seed(seed)
        random.shuffle(ctx)
        random.seed(seed)
        random.shuffle(lbl)

        # feed batches actually
        for i in range(nb_batch):
            begin, end = batch_size*i, batch_size*(i+1)
            yield ([pvt[begin: end], ctx[begin: end]], lbl[begin: end])


# load data
# sentences: list of (list of id)
# index2word: list of string
sentences, index2word = utils.load_sentences_brown(10000)

# params
nb_epoch = 4
# learn `batch_size words` at a time
batch_size = 3000
vec_dim = 128
window_size = 5
vocab_size = len(index2word)

# create input
couples, labels = utils.skip_grams(sentences, window_size, vocab_size)

# metrics
nb_batch = len(labels) // batch_size
samples_per_epoch = batch_size * nb_batch

# graph definition
input_pvt = Input(batch_shape=(batch_size, 1), dtype='int32')
input_ctx = Input(batch_shape=(batch_size, 1), dtype='int32')

embedded_pvt = Embedding(input_dim=vocab_size,
                         output_dim=vec_dim,
                         input_length=1)(input_pvt)

embedded_ctx = Embedding(input_dim=vocab_size,
                         output_dim=vec_dim,
                         input_length=1)(input_ctx)

merged = merge(inputs=[embedded_pvt, embedded_ctx],
               mode=lambda a: (a[0]*a[1]).sum(-1),
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
utils.most_similar(positive=['she', 'him'], negative=['he']) #=> 'her'
