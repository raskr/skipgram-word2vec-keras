def load_sentences_brown(nb_sentences=None):
    """
    :param nb_sentences: Use if all brown sentences are too many
    :return: list of (list of word-id)
    """
    from nltk.corpus import brown

    index2word = {}
    word2index = {}
    sentences = []
    stop_words = ['the', 'in', 'to', 'or', 'and', 'for', 'of']

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


# def load_sentences_ptb(nb_sentences=2000):
#     """
#     :param nb_sentences: Use if all brown sentences are too many
#     :return: list of (list of word-id)
#     """
#
#     index2word = {}
#     word2index = {}
#     sentences = []
#
#     n = 0
#     with open('ptb.train.txt') as f:
#         for sent in f:
#             if n > nb_sentences:
#                 break
#             n += 1
#             words = []
#             for word in sent.split():
#                 if word == '.' or word == ',': break
#                 if word not in word2index :
#                     ind = len(word2index)
#                     word2index[word] = ind
#                     index2word[ind] = word
#                 words.append(word2index[word])
#             sentences.append(words)
#     return sentences, index2word, word2index


def create_input(x, y, b_size):
    import numpy as np
    """
    :param x: Output from skip_grams(). i.e. [[1, 2], [1, 3], [2, 1], ...]
    :return: Three numpy arrays. Each of them has same shape like (n,)
    """
    garbage = len(x) % b_size

    p = np.array(x)[:, 0][:-garbage]
    c = np.array(x)[:, 1][:-garbage]
    l = np.array(y)[:-garbage]

    return p, c, l


def skip_grams(sentences, window_size, vocab_size, nb_negative_samples=6.):
    """
    calc `keras.preprocessing.sequence.skipgrams` for each sentence
    and concatenate those.

    :param sentences: i.e. [[1, 2, 3, 4, 5], [6, 7], [2, 7], ...]
    :return: concatenated skip-grams of each sentence
    """
    import keras.preprocessing.sequence as seq
    import numpy as np

    # table = seq.make_sampling_table(vocab_size)
    table = None

    def sg(sentence):
        w = np.random.randint(window_size - 1) + 1
        return seq.skipgrams(sentence, vocab_size, window_size=w,
                             negative_samples=nb_negative_samples,
                             shuffle=True, sampling_table=table)

    # concat and flatten
    couples = []
    labels = []
    for c, l in map(sg, sentences):
        couples.extend(c)
        labels.extend(l)
    return couples, labels


def save_weights(model, index2word, vocab_size, vec_dim):
    # save weights
    # filename = datetime.now().strftime('vectors_%Y-%m-%d_%H:%M:%S.txt')
    vec = model.get_weights()[0]
    f = open('vectors_embedding.txt', 'w')
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
    vec = models.word2vec.Word2Vec.load_word2vec_format('vectors_embedding.txt', binary=False)
    for v in vec.most_similar_cosmul(positive=positive, negative=negative, topn=20):
        print(v)
