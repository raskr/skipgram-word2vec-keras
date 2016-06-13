from keras.layers import Input, merge, Activation
from keras.models import Model
from keras.layers.embeddings import Embedding
import utils


def batch_generator(n_batch, b_size, data_pvt, data_ctx, data_lbl):
    while 1:
        for i in range(n_batch):
            begin, end = b_size*i, b_size*(i+1)
            yield (
                [data_pvt[begin: end],
                 data_ctx[begin: end]],
                data_lbl[begin: end],
            )


# load data
sentences, index2word, word2index = utils.load_sentences_brown()

# params
nb_epoch = 10
batch_size = 10000 # learn 10000 `words` at a time
vec_dim = 128
window_size = 7
vocab_size = len(index2word)

# create input
couples, labels = utils.skip_grams(sentences, window_size, vocab_size)
data_pivot, data_context, data_label = utils.create_input(couples, labels, batch_size)
assert data_pivot.shape == data_context.shape == data_label.shape

# metrics
nb_batch = len(data_pivot) // batch_size
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
gen = batch_generator(nb_batch, batch_size, data_pivot, data_context, data_label)
model.fit_generator(generator=gen,
                    samples_per_epoch=samples_per_epoch,
                    nb_epoch=nb_epoch, verbose=1)

# save weights
utils.save_weights(model, index2word, vocab_size, vec_dim)

# eval using gensim
utils.most_similar(positive=['she', 'him'], negative=['he']) #=> 'her'