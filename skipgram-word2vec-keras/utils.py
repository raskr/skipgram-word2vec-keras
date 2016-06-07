def load_sentences_brown(nb_sentences=2000):
    """
    :param nb_sentences: Use if all brown sentences are too many
    :return: list of (list of word-id)
    """
    from nltk.corpus import brown

    index2word = {}
    word2index = {}
    sentences = []

    for sent in brown.sents()[:nb_sentences]:
        words = []
        for word in sent[:-1]: # exclude period from a sent
            if word not in word2index:
                ind = len(word2index)
                word2index[word] = ind
                index2word[ind] = word
            words.append(word2index[word])
        sentences.append(words)
    return sentences, index2word, word2index


def skip_grams(sentences, window_size, vocab_size, nb_negative_samples=1):
    """
    calc `keras.preprocessing.sequence#skipgrams` for each sentence
    and concatenate those.

    :param sentences: i.e. [[1, 2, 3, 4, 5], [6, 7], [2, 7], ...]
    :return: concatenated skip-grams of each sentence
    """
    import keras.preprocessing.sequence as seq
    import numpy as np

    def call_skip_grams(sentence):
        w = np.random.randint(window_size - 1) + 1
        return seq.skipgrams(sentence, vocab_size, window_size=w,
                             negative_samples=nb_negative_samples)

    # map(sentence to skip-grams)
    tmp = map(call_skip_grams, sentences)
    # concat and flatten
    return reduce(lambda acc, x: (acc[0]+x[0], acc[1]+x[1]), tmp)


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


def similar_words_of(word):
    from gensim import models
    vec = models.word2vec.Word2Vec.load_word2vec_format('vectors_embedding.txt', binary=False)
    for v in vec.most_similar(positive=[word], topn=20):
        print(v)
