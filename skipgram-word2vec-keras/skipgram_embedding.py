from keras.layers import Input, merge, Activation
from keras.models import Model
from keras.layers.embeddings import Embedding
import utils


def batch_generator(cpl, lbl):
    import random

    # trim the tail
    garbage = len(labels) % batch_size

    pvt = cpl[:, 0][:-garbage]
    ctx = cpl[:, 1][:-garbage]
    lbl = lbl[:-garbage]

    assert pvt.shape == ctx.shape == lbl.shape

    # epoch loop
    while 1:
        # shuffle data at beginning of every epoch (takes few minutes)
        seed = random.randint(0, 10e6)
        random.seed(seed)
        random.shuffle(pvt)
        random.seed(seed)
        random.shuffle(ctx)
        random.seed(seed)
        random.shuffle(lbl)

        for i in range(nb_batch):
            begin, end = batch_size*i, batch_size*(i+1)
            # feed i th batch
            yield ([pvt[begin: end], ctx[begin: end]], lbl[begin: end])


# load data
# - sentences: list of (list of word-id)
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
input_pvt = Input(batch_shape=(batch_size, 1), dtype='int32')
input_ctx = Input(batch_shape=(batch_size, 1), dtype='int32')

embedded_pvt = Embedding(input_dim=vocab_size,
                         output_dim=vec_dim,
                         input_length=1)(input_pvt)

embedded_ctx = Embedding(input_dim=vocab_size,
                         output_dim=vec_dim,
                         input_length=1)(input_ctx)

merged = merge(inputs=[embedded_pvt, embedded_ctx],
               mode=lambda x: (x[0] * x[1]).sum(-1),
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
