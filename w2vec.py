from gensim import  models
import pandas as pd
from nltk.corpus import stopwords
import string
import pprint

""" DEMO """
def depunct(tokens):
    """Remove punctuations from the text"""
    return [token.translate(None, string.punctuation) for token in tokens]
        
data = pd.read_csv('data/calvin.csv')
documents = data['quote'].tolist()
stoplist = stopwords.words('english')

sent = [d.lower().split() for d in documents]
texts = [depunct([word for word in document.lower().split() if word
                  not in stoplist]) for document in documents]
bigrams_model = models.Phrases(texts)
bigrams = list(bigrams_model[texts])
trigrams_model = models.Phrases(bigrams)
trigrams = list(trigrams_model[bigrams])

sent.extend(trigrams)
sent.extend(bigrams)

model = models.Word2Vec()
model.build_vocab(sent)
model.train(sent)

chain = ['calvin', 'tiger', 'hobbes', 'mom']
pprint.pprint([k for k,v in model.most_similar(positive=chain, negative=[], topn=50) if '_' in k])
