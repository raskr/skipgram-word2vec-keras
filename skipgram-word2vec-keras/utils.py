filename = 'vectors_embedding.txt'


def load_sentences_brown(min_count=5, nb_sentences=None):
    """
    :param min_count: I remove less-frequent words than this value
    :param nb_sentences: Use if all brown sentences are too many
    :return: index2word (list of string)
    """
    from nltk.corpus import brown
    import gensim

    print 'building vocab ...'
    sents = brown.sents() if nb_sentences is None else brown.sents()[:nb_sentences]

    # I use gensim model only for building vocab
    model = gensim.models.Word2Vec(min_count=min_count)
    model.build_vocab(sents)
    vocab = model.vocab

    ids = [[vocab[w].index for w in sent
            if w in vocab and vocab[w].sample_int > model.random.rand() * 2**32]
           for sent in sents]
    return ids, model.index2word


def skip_grams(sentences, window, vocab_size, nb_negative_samples=6.):
    """
    calc `keras.preprocessing.sequence.skipgrams` for each sentence
    and concatenate those.

    :param sentences: i.e. [[1, 2, 3, 4, 5], [6, 7], [2, 7], ...]
    :return: concatenated skip-grams
    """
    import keras.preprocessing.sequence as seq
    import numpy as np

    print 'building skip-grams ...'
    # table = seq.make_sampling_table(vocab_size)
    table = None

    def sg(sentence):
        return seq.skipgrams(sentence, vocab_size,
                             window_size=np.random.randint(window - 1) + 1,
                             sampling_table=table,
                             negative_samples=nb_negative_samples)

    couples = []
    labels = []

    # concat all skipgrams
    for cpl, lbl in [sg(sent) for sent in sentences]:
        couples.extend(cpl)
        labels.extend(lbl)

    return couples, labels


def create_input(x, y, b_size):
    """
    :param x: Output from skip_grams(). i.e. [[1, 2], [1, 3], [2, 1], ...]
    :param y: Output from skip_grams(). i.e. [ 1, 0, 1, ...]
    :param b_size: size of one batch
    :return: Three numpy arrays. Each of them has same shape like (n,)
    """
    import numpy as np

    garbage = len(x) % b_size

    pivot = np.array(x)[:, 0][:-garbage]
    ctx = np.array(x)[:, 1][:-garbage]
    label = np.array(y)[:-garbage]

    return pivot, ctx, label


def save_weights(model, index2word, vec_dim):
    """
    :param model: keras model
    :param index2word: list of string (treat list index as id)
    :param vec_dim: dim of embedding vector
    :return:
    """
    vec = model.get_weights()[0]
    f = open(filename, 'w')
    f.write(" ".join([str(len(index2word)), str(vec_dim)]))
    f.write("\n")
    for i, word in enumerate(index2word):
        f.write(word)
        f.write(" ")
        f.write(" ".join(map(str, list(vec[i, :]))))
        f.write("\n")
    f.close()


def most_similar(positive=[], negative=[]):
    """
    :param positive: list of string
    :param negative: list of string
    :return:
    """
    from gensim import models
    vec = models.word2vec.Word2Vec.load_word2vec_format(filename, binary=False)
    for v in vec.most_similar_cosmul(positive=positive, negative=negative, topn=20):
        print(v)
