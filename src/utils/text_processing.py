import gensim
import spacy
from gensim.utils import simple_preprocess
import nltk
from nltk.corpus import stopwords

"""
    This function returns words without any punctuation
"""
def sentences_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))

def remove_stopwords(texts):
    stop_words = list(stopwords.words('english')) + ['from', 'subject', 're', 'edu', 'use']
    return [[word for word in gensim.utils.simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

def do_bigrams(bimodels, texts):
    return [bimodels[doc] for doc in texts]

def do_trigrams(trimodels, bimodels, texts):
    return [trimodels[bimodels[doc]] for doc in texts]

def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    nlp = spacy.load('en_core_web_lg', disable=['parser', 'ner'])
    lems = []
    for text in texts:
        doc = nlp(" ".join(text))
        lems.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    
    return lems
