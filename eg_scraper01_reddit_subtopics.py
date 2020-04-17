# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 19:19:37 2019

@author: Marco
"""
#=====Import and Declare=========================================
# Import libraries
import requests, re

url = "https://www.reddit.com/r/politics.json"
subtopics_dict = {} # Export this with metadata to aid in building a corpus

payload = {}
headers = {
  'User-agent': 'headlessghost'
}

response = requests.request("GET", url, headers=headers, data = payload)
response_string = str(response.text.encode('utf-8'))

title_list = re.findall(r'"title": "(.*?)"', response_string)
link_list = re.findall(r'"permalink": "(.*?)"', response_string)

if len(title_list) == len(link_list):
    for i in range(0, len(title_list)):
        #add to dic
        subtopics_dict[str(i)] = {}
        subtopics_dict[str(i)]['title'] = title_list[i]
        subtopics_dict[str(i)]['link'] = link_list[i]
        
else:
    if len(title_list) < len(link_list):
        for i in range(0, len(title_list)):
            #add to dic
            subtopics_dict[str(i)] = {}
            subtopics_dict[str(i)]['title'] = title_list[i]
            subtopics_dict[str(i)]['link'] = link_list[i]
    else:
        for i in range(0, len(link_list)):      #link_list is shorter
            #add to dic
            subtopics_dict[str(i)] = {}
            subtopics_dict[str(i)]['title'] = title_list[i]
            subtopics_dict[str(i)]['link'] = link_list[i]

#Get URL for subtopic comments
for i in range(0, len(subtopics_dict)):                #len(subtopics_dict)
    url_prefix = r'http://reddit.com'
    url_suffix = '.json'
    url_combined = url_prefix + subtopics_dict[str(i)]['link'] + url_suffix
    
    payload = {}
    headers = {
      'User-agent': 'headlessghost'
    }
    
    try:
        response = requests.request("GET", url_combined, headers=headers, data = payload)
        response_string = str(response.text.encode('utf-8'))
        
        comment_list = re.findall(r'"body": "(.*?)"', response_string)
        subtopics_dict[str(i)]['comments'] = comment_list  
        
    except:
        print("An exception occurred")
        
#==================================================================
#/////////Data Cleaning and Processing Steps
#==================================================================

#======================================================
#/// Topic Modeling (LDA) << [SubTopic Titles]

'''
1) Process training text (import saved model?) 
    https://radimrehurek.com/gensim/models/callbacks.html
    https://radimrehurek.com/gensim/models/ldamodel.html
    https://radimrehurek.com/gensim/models/ldamulticore.html
    
    from gensim.test.utils import datapath
    
     # Save model to disk.
    temp_file = datapath("model")
    lda.save(temp_file)
    
    # Load a potentially pretrained model from disk.
    lda = LdaModel.load(temp_file)
    
    
'''
#Write a function to perform the pre processing steps on the entire dataset
import pandas as pd
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
import numpy as np
import nltk
nltk.download('wordnet')
stemmer = SnowballStemmer("english")

# Keyed storage of all output values. This will shared with visualizer 
output_dict = {}

# Import corpus data
data = pd.read_csv('abcnews-date-text.csv');
# We only need the Headlines text column from the data
data_text = data[:300000][['headline_text']];
data_text['index'] = data_text.index

documents = data_text

def lemmatize_stemming(text):
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))

# Tokenize and lemmatize
def preprocess(text):
    result=[]
    for token in gensim.utils.simple_preprocess(text) :
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            # TODO: Apply lemmatize_stemming on the token, then add to the results list
            result.append(lemmatize_stemming(token))
    return result

processed_docs = documents['headline_text'].map(preprocess)

dictionary = gensim.corpora.Dictionary(processed_docs)

# TODO: apply dictionary.filter_extremes() with the parameters mentioned above
dictionary.filter_extremes(no_below=15, no_above=0.1, keep_n=100000)

from gensim.test.utils import datapath
    
# Load a potentially pretrained model from disk.
temp_file = datapath(r'C:\Users\Marco\Desktop\Gits\eg-texttools\lda-models\model_save')
lda_model_tfidf = gensim.models.LdaMulticore.load(temp_file)

# LOOP: Create a loop and storage for returned subtopic LDA values
unseen_document = subtopics_dict['2']['title']

    # Data preprocessing step for the unseen document
bow_vector = dictionary.doc2bow(preprocess(unseen_document))

    # LDA Topic Modeling && and store result in OUTPUT dictionary
    # The first loop entry pulls the OUTPUT dictionary
topic_dict = {}
for index, score in sorted(lda_model_tfidf[bow_vector], key=lambda tup: -1*tup[1]):
    #print("Score: {}\t Topic: {}".format(score, lda_model_tfidf.print_topic(index, 5)))
    topic_dict[str(index)] = lda_model_tfidf.print_topic(index, 5)
for t in range(0, len(topic_dict)):
    topic_dict[str(t)] = re.sub(r"[^a-z\+]", "", topic_dict[str(t)])
    topic_dict[str(t)] = topic_dict[str(t)].split("+")    

# Get highest topic from score
score_dict = {}
for index, score in sorted(lda_model_tfidf[bow_vector], key=lambda tup: -1*tup[1]):
    #print("Score: {}\t Topic: {}".format(score, lda_model_tfidf.print_topic(index, 5)))
    score_dict[str(index)] = {}
    score_dict[str(index)]['topic'] = lda_model_tfidf.print_topic(index, 1)
    score_dict[str(index)]['score'] = score
for t in range(0, len(score_dict)):
    score_dict[str(t)]['topic'] = re.sub(r"[^a-z\+]", "", score_dict[str(t)]['topic'])
print(score_dict)

highscore = 0.0
topic_select = ""
for t in range(0, len(score_dict)):
    score = score_dict[str(t)]['score']
    topic = score_dict[str(t)]['topic']
    if score > highscore:
        highscore = score
        topic_select = topic
print(str(highscore) + "  " + topic_select)  


# Check for topic_dict for matching word in 
for t in range(0, len(topic_dict)):
    #print(topic_dict[str(t)])
    for w in topic_dict[str(t)]:
        #print(w)
        if topic_select == w:
            print(t)
#print(topic_dict)
        
#======================================================
#/// POS Tagging (HMM Tagger) << [Comments]

'''
1) Process training text 
2) Train and save model in jupyter file
3) (import saved model?) 
    https://pomegranate.readthedocs.io/en/latest/callbacks.html
    https://pomegranate.readthedocs.io/en/latest/HiddenMarkovModel.html#pomegranate.hmm.HiddenMarkovModel.from_json
    https://pomegranate.readthedocs.io/en/latest/parallelism.html#faq  *example vvv
    
    # Save json model example (?):
        model = HiddenMarkovModel.from_samples(X_train)
        with open("model.json", "w") as outfile:
            outfile.write(model.to_json())              
    
    # Load json model example:
        model = HiddenMarkovModel.from_json(name)    
'''
import numpy as np

from IPython.core.display import HTML
from itertools import chain
from collections import Counter, defaultdict
from helpers import show_model, Dataset
from pomegranate import State, HiddenMarkovModel, DiscreteDistribution
from bs4 import BeautifulSoup as soup 
import html5lib
import nltk
nltk.download("stopwords")   # download list of stopwords (only once; need not run it again)
nltk.download("sentiwordnet")
from nltk.corpus import stopwords # import stopwords
from nltk.corpus import sentiwordnet as swn
from nltk.stem.porter import *
stemmer = PorterStemmer()

# Import data
data = Dataset("tags-universal.txt", "brown-universal.txt", train_test_split=0.8)

def replace_unknown(sequence):
    """Return a copy of the input sequence where each unknown word is replaced
    by the literal string value 'nan'. Pomegranate will ignore these values
    during computation.
    """
    return [w if w in data.training_set.vocab else 'nan' for w in sequence]

def simplify_decoding(X, model):
    """X should be a 1-D sequence of observations for the model to predict"""
    _, state_path = model.viterbi(replace_unknown(X))
    return [state[1].name for state in state_path[1:-1]] 


# Load json model
model_new = HiddenMarkovModel.from_json(r'C:\Users\Marco\Desktop\Gits\eg-texttools\hmm-models\model.json')

def review_to_words(review):
    """Convert a raw review string into a sequence of words."""
    
    # TODO: Remove HTML tags and non-letters,
    #       convert to lowercase, tokenize,
    #       remove stopwords and stem
    text = soup(review, "html.parser").get_text()
    text = re.sub(r"[^a-zA-Z\.{1}]", " ", text.lower())
    words = text.split()
    words = [w for w in words if w not in stopwords.words("english")]
    words = [PorterStemmer().stem(w) for w in words]

    # Return final list of words
    return words

tag_dict = {}
sentiment_dict = {}
# LOOP through all subtopic comments:
for c in range(1, len(subtopics_dict['1']['comments'])): # Skip first comment to avoid boilerplate
    raw_comment = subtopics_dict['1']['comments'][c]

    #1) Clean all raw comment text
    input_text = review_to_words(raw_comment)
        
        #2) TAG words predictions <<< Comment by comment
    tagged_text = simplify_decoding(input_text, model_new)   
        #3) Count TAGs for each comment && store in OUTPUT dictionary
    tag_dict[str(c)] = {'.':0, 'ADJ':0, 'ADP':0, 'ADV':0, 'CONJ':0, 'DET':0, 'NOUN':0, 'NUM':0, 'PRON':0, 'PRT':0, 'VERB':0, 'X':0}
        
        #4) Sentiment analysis && store in OUTPUT dictionary
    sentiment_dict[str(c)] = {'pos': 0.0, 'neg': 0.0, 'obj': 0.0}
    if len(tagged_text) == len(input_text):
        for i in range(0, len(tagged_text)):
            tag = tagged_text[i]
            word = input_text[i]
            if tag == 'NOUN':
                tag_dict[str(c)]['NOUN'] += 1
                try:
                    sentiment = swn.senti_synset(word + ".n.01")
                    sentiment_dict[str(c)]['pos'] += sentiment.pos_score()
                    sentiment_dict[str(c)]['neg'] += sentiment.neg_score()
                    sentiment_dict[str(c)]['obj'] += sentiment.obj_score()
                except:
                    print("An exception occurred")
            if tag == 'VERB':
                tag_dict[str(c)]['VERB'] += 1
                try:
                    sentiment = swn.senti_synset(word + ".v.01")
                    sentiment_dict[str(c)]['pos'] += sentiment.pos_score()
                    sentiment_dict[str(c)]['neg'] += sentiment.neg_score()
                    sentiment_dict[str(c)]['obj'] += sentiment.obj_score()
                except:
                    print("An exception occurred")
            if tag == 'ADJ':
                tag_dict[str(c)]['ADJ'] += 1
                try:
                    sentiment = swn.senti_synset(word + ".a.01")
                    sentiment_dict[str(c)]['pos'] += sentiment.pos_score()
                    sentiment_dict[str(c)]['neg'] += sentiment.neg_score()
                    sentiment_dict[str(c)]['obj'] += sentiment.obj_score()
                except:
                    print("An exception occurred")
            if tag == 'ADV':
                tag_dict[str(c)]['ADV'] += 1
                try:
                    sentiment = swn.senti_synset(word + ".r.01")
                    sentiment_dict[str(c)]['pos'] += sentiment.pos_score()
                    sentiment_dict[str(c)]['neg'] += sentiment.neg_score()
                    sentiment_dict[str(c)]['obj'] += sentiment.obj_score()
                except:
                    print("An exception occurred")
            if tag == '.':
                tag_dict[str(c)]['.'] += 1
            if tag == 'ADP':
                tag_dict[str(c)]['ADP'] += 1
            if tag == 'CONJ':
                tag_dict[str(c)]['CONJ'] += 1
            if tag == 'DET':
                tag_dict[str(c)]['DET'] += 1
            if tag == 'NUM':
                tag_dict[str(c)]['NUM'] += 1
            if tag == 'PRON':
                tag_dict[str(c)]['PRON'] += 1
            if tag == 'PRT':
                tag_dict[str(c)]['PRT'] += 1
            if tag == 'X':
                tag_dict[str(c)]['X'] += 1
print(len(tag_dict))

    #5) Store <<< Comment by comment





