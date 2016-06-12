filename = 'vectors_embedding.txt'


def load_sentences_brown(nb_sentences=None):
    """
    :param nb_sentences: Use if all brown sentences are too many
    :return: list of (list of word-id), index2word, word2index
    """
    from nltk.corpus import brown

    index2word = {}
    word2index = {}
    sentences = []
    stop_words = ['the', 'in', 'to', 'for', 'of', ',', '.', '"']

    sents = brown.sents()[:nb_sentences] if nb_sentences else brown.sents()
    for sent in sents:
        words = []
        for word in sent:
            word = str(word).lower()
            if word in stop_words:
                break
            if word not in word2index:
                ind = len(word2index)
                word2index[word] = ind
                index2word[ind] = word
            words.append(word2index[word])
        sentences.append(words)
        # print map(lambda x: index2word[x], words)
    return sentences, index2word, word2index


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


def skip_grams(sentences, window_size, vocab_size, nb_negative_samples=6.):
    """
    calc `keras.preprocessing.sequence.skipgrams` for each sentence
    and concatenate those.

    :param sentences: i.e. [[1, 2, 3, 4, 5], [6, 7], [2, 7], ...]
    :return: concatenated skip-grams
    """
    import keras.preprocessing.sequence as seq
    import numpy as np

    def sg(sentence):
        w = np.random.randint(window_size - 1) + 1
        return seq.skipgrams(sentence, vocab_size, window_size=w,
                             negative_samples=nb_negative_samples)

    # concat and flatten
    couples = []
    labels = []
    for c, l in map(sg, sentences):
        couples.extend(c)
        labels.extend(l)
    return couples, labels


def save_weights(model, index2word, vocab_size, vec_dim):
    # save weights
    vec = model.get_weights()[0]
    f = open(filename, 'w')
    f.write(" ".join([str(vocab_size), str(vec_dim)]))
    f.write("\n")
    for i, word in index2word.items():
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
