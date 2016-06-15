import gensim

from nltk.corpus import brown
brown_sents = list(brown.sents())[:20000]

filename = 'vectors_embedding_gensim.txt'
model = gensim.models.word2vec.Word2Vec(brown_sents, hs=-1, negative=5, sg=1)
model.train(brown_sents)
model.save(filename)

for v in model.most_similar_cosmul(positive=['she', 'him'], negative=['he'], topn=20):
    print(v)
