import string
import bz2
import nltk
from collections import Counter
from gensim.models import Phrases
from gensim.models import Word2Vec
from nltk.corpus import stopwords

sentences = []
bigram = Phrases()

with bz2.BZ2File('./2009.csv.bz2') as file_:
    for i, line in enumerate(file_):
        sentence = [word
                    for word in nltk.word_tokenize(line.decode("utf-8").lower())
                    if word not in string.punctuation]
        sentences.append(sentence)
        bigram.add_vocab([sentence])

bigram_model = Word2Vec(bigram[sentences])
bigram_model_counter = Counter()

bigram_model.save('ok.w2v')

for key in bigram_model.vocab.keys():
    if key not in stopwords.words("english"):
        if len(key.split("_")) > 1:
            bigram_model_counter[key] += bigram_model.vocab[key].count

for key, counts in bigram_model_counter.most_common(50):
    print('{0: <20} {1}'.format(key.encode("utf-8"), counts))
