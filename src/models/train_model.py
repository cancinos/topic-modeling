import pandas as pd
import gensim
import pickle
from gensim.models import CoherenceModel
import gensim.corpora as corpora
from pprint import pprint
from src.utils import text_processing

def train(): 
    """
        0. Load words generate by raw data
    """
    with open('data/processed/words.pkl', 'rb') as file:
        words = pickle.load(file)
    #words = pd.DataFrame(pd.read_json('data/processed/words.json', encoding='utf8')).values.tolist()
    """
        1. Let's generate our bigrams and trigrams from words
    """
    bigrams = gensim.models.Phrases(words, min_count=5, threshold=100)
    trigrams = gensim.models.Phrases(bigrams[words], threshold=100)

    bimodel = gensim.models.phrases.Phraser(bigrams)
    trimodel = gensim.models.phrases.Phraser(trigrams)
    
    """
        2. Now we eliminate stopwords
    """
    words = text_processing.remove_stopwords(words)
    """
        3. And use the resultant words to create bigrams, and trigrams
    """
    wbigrams = text_processing.do_bigrams(bimodel, words)
    """
        4. Let's create our lems
    """
    lems = text_processing.lemmatization(wbigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
    """
        5. Now we need to build a dictionary
    """
    dic = corpora.Dictionary(lems)
    """
        6. Let's build a term-document frequency structure
    """
    corpus = [dic.doc2bow(text) for text in lems]
    
    """
        7. Finally, let's generate a LDA model and see what's on our docs
    """
    lda = gensim.models.ldamodel.LdaModel(
        corpus=corpus,
        id2word=dic,
        num_topics=20,
        random_state=100,
        update_every=1,
        chunksize=100,
        passes=10,
        alpha='auto',
        per_word_topics=True
    )

    pprint(lda.print_topics())

train()