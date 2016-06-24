# output file
filename = 'vectors.txt'


def load_sentences_brown(nb_sentences=None):
    """
    :param nb_sentences: Use if all brown sentences are too many
    :return: index2word (list of string)
    """
    from nltk.corpus import brown
    import gensim

    print 'building vocab ...'

    if nb_sentences is None:
        sents = brown.sents()
    else:
        sents = brown.sents()[:nb_sentences]

    # I use gensim model only for building vocab
    model = gensim.models.Word2Vec()
    model.build_vocab(sents)
    vocab = model.vocab

    # ids: list of (list of word-id)
    ids = [[vocab[w].index for w in sent
            if w in vocab and vocab[w].sample_int > model.random.rand() * 2**32]
           for sent in sents]

    return ids, model.index2word


def skip_grams(sentences, window, vocab_size, nb_negative_samples=5.):
    """
    calc `keras.preprocessing.sequence.skipgrams` for each sentence
    and concatenate those.

    :param sentences: list of (list of word-id)
    :return: concatenated skip-grams
    """
    import keras.preprocessing.sequence as seq
    import numpy as np

    print 'building skip-grams ...'

    def sg(sentence):
        return seq.skipgrams(sentence, vocab_size,
                             window_size=np.random.randint(window - 1) + 1,
                             negative_samples=nb_negative_samples)

    couples = []
    labels = []

    # concat all skipgrams
    for cpl, lbl in map(sg, sentences):
        couples.extend(cpl)
        labels.extend(lbl)

    return np.asarray(couples), np.asarray(labels)


def save_weights(model, index2word, vec_dim):
    """
    :param model: keras model
    :param index2word: list of string
    :param vec_dim: dim of embedding vector
    :return:
    """
    vec = model.get_weights()[0]
    f = open(filename, 'w')
    # first row in this file is vector information
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
