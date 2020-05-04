# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 19:19:37 2019

@author: Marco Flagg
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
#LDA Topic Modeling && POS TAGs && Sentiment Analysis
#==================================================================
import numpy as np
import pandas as pd

import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from gensim.test.utils import datapath
from pomegranate import State, HiddenMarkovModel, DiscreteDistribution

from IPython.core.display import HTML
from itertools import chain
from collections import Counter, defaultdict
from helpers import show_model, Dataset
from bs4 import BeautifulSoup as soup 

import nltk
nltk.download("stopwords")   # download list of stopwords (only once; need not run it again)
nltk.download("sentiwordnet")
nltk.download('wordnet')
from nltk.corpus import stopwords # import stopwords
from nltk.corpus import sentiwordnet as swn
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer, SnowballStemmer
snowball_stemmer = SnowballStemmer("english")
porter_stemmer = PorterStemmer()

# Import LDA corpus data
lda_data = pd.read_csv('abcnews-date-text.csv');
lda_data_text = lda_data[:300000][['headline_text']]; # We only need the Headlines text column from the data
lda_data_text['index'] = lda_data_text.index
documents = lda_data_text

# Import HMM Tagger corpus data and model
hmm_data = Dataset("tags-universal.txt", "brown-universal.txt", train_test_split=0.8)
hmm_model = HiddenMarkovModel.from_json(r'C:\Users\Marco\Desktop\Gits\eg-texttools\hmm-models\model.json')

# Output dictionary. To export data to visualizer. 
output_dict = {}

def lemmatize_stemming(text):
    return snowball_stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))

# Tokenize and lemmatize
def preprocess(text):
    result=[]
    for token in gensim.utils.simple_preprocess(text) :
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            # TODO: Apply lemmatize_stemming on the token, then add to the results list
            result.append(lemmatize_stemming(token))
    return result

def replace_unknown(sequence):
    """Return a copy of the input sequence where each unknown word is replaced
    by the literal string value 'nan'. Pomegranate will ignore these values
    during computation.
    """
    return [w if w in hmm_data.training_set.vocab else 'nan' for w in sequence]

def simplify_decoding(X, model):
    """X should be a 1-D sequence of observations for the model to predict"""
    _, state_path = model.viterbi(replace_unknown(X))
    return [state[1].name for state in state_path[1:-1]] 

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

#======================================================
#/// Topic Modeling (LDA) << [SubTopic Titles]
# Preprocess raw corpus data && Create dictionary
processed_docs = documents['headline_text'].map(preprocess)
dictionary = gensim.corpora.Dictionary(processed_docs)

# Apply dictionary.filter_extremes() with the parameters mentioned above
dictionary.filter_extremes(no_below=15, no_above=0.1, keep_n=100000)

# Load a potentially pretrained model from disk.
temp_file = datapath(r'C:\Users\Marco\Desktop\Gits\eg-texttools\lda-models\model_save')
lda_model_tfidf = gensim.models.LdaMulticore.load(temp_file)

topic_dict = {}

# LOOP: Create a loop and storage for returned subtopic LDA values
for s in range (1, len(subtopics_dict)):
    
    output_dict[str(s)] = {'topic':''}
    subtopic_title = subtopics_dict[str(s)]['title']

        # Data preprocessing step for the unseen document
    bow_vector = dictionary.doc2bow(preprocess(subtopic_title))
    
    # LDA Topic Modeling && and store result in OUTPUT dictionary
    # The first loop entry creates the Topic dictionary
    if s == 1:
        for index, score in sorted(lda_model_tfidf[bow_vector], key=lambda tup: -1*tup[1]):
            topic_dict[str(index)] = lda_model_tfidf.print_topic(index, 5)
        for t in range(0, len(topic_dict)):
            topic_dict[str(t)] = re.sub(r"[^a-z\+]", "", topic_dict[str(t)])
            topic_dict[str(t)] = topic_dict[str(t)].split("+")    
    
    # Build comment score dictionary
    topic_list = []
    score_list = []
    for index, score in sorted(lda_model_tfidf[bow_vector], key=lambda tup: -1*tup[1]):
        topic_list.append(lda_model_tfidf.print_topic(index, 1))
        score_list.append(score)
    for t in range(0, len(topic_list)):
        topic_list[t] = re.sub(r"[^a-z\+]", "", topic_list[t])
    
    # Get highest topic from score
    highscore = 0.0
    topic_select = ""
    if len(topic_list) == len(score_list):
        for t in range(0, len(score_list)):
            score = score_list[t]
            topic = topic_list[t]
            if score > highscore:
                highscore = score
                topic_select = topic
        
        # Check topic_dict for matching word in topic dictionary
        for t in range(0, len(topic_dict)):
            #print(topic_dict[str(t)])
            for w in topic_dict[str(t)]:
                #print(w)
                if topic_select == w:
                    # Save matching topic to output_dictionary
                    output_dict[str(s)]['topic'] = t
        
    #======================================================
    #/// POS Tagging (HMM Tagger) << [Comments]
    
    # Create 'tag' and 'sentiments' subdictionaries for each subtopic
    output_dict[str(s)]['tags'] = {}
    output_dict[str(s)]['sentiments'] = {}
    #output_dict = {0:{'topic':'?', 'tags': {1:{N':0, 'PRT':0, 'VERB':0, 'X':0}, 2:{}...}, 'sentiments': {1: {'pos': 0.0, 'neg': 0.0, 'obj': 0.0}, 2:{}...}}}

    # LOOP through all subtopic comments:
    for c in range(1, len(subtopics_dict[str(s)]['comments'])): # Skip first comment to avoid boilerplate
        raw_comment = subtopics_dict[str(s)]['comments'][c]
    
        #1) Clean all raw comment text
        input_text = review_to_words(raw_comment)
        if len(input_text) == 0:
            input_text.append("unknown")
            
        #2) TAG words predictions <<< Comment by comment
        tagged_text = simplify_decoding(input_text, hmm_model)   
        
        #3) Initialize Comment TAG && SENTIMENT dictionaries for each comment
        output_dict[str(s)]['tags'][str(c)] = {'.':0, 'ADJ':0, 'ADP':0, 'ADV':0, 'CONJ':0, 'DET':0, 'NOUN':0, 'NUM':0, 'PRON':0, 'PRT':0, 'VERB':0, 'X':0}
        output_dict[str(s)]['sentiments'][str(c)] = {'pos': 0.0, 'neg': 0.0, 'obj': 0.0}
        if len(tagged_text) == len(input_text):
            
            # LOOP through Comments to count TAGS and SENTIMENT
            for i in range(0, len(tagged_text)):
                tag = tagged_text[i]
                word = input_text[i]
                if tag == 'NOUN':
                    output_dict[str(s)]['tags'][str(c)]['NOUN'] += 1
                    try:
                        sentiment = swn.senti_synset(word + ".n.01")
                        output_dict[str(s)]['sentiments'][str(c)]['pos'] += sentiment.pos_score()
                        output_dict[str(s)]['sentiments'][str(c)]['neg'] += sentiment.neg_score()
                        output_dict[str(s)]['sentiments'][str(c)]['obj'] += sentiment.obj_score()
                    except:
                        print("An exception occurred")
                if tag == 'VERB':
                    output_dict[str(s)]['tags'][str(c)]['VERB'] += 1
                    try:
                        sentiment = swn.senti_synset(word + ".v.01")
                        output_dict[str(s)]['sentiments'][str(c)]['pos'] += sentiment.pos_score()
                        output_dict[str(s)]['sentiments'][str(c)]['neg'] += sentiment.neg_score()
                        output_dict[str(s)]['sentiments'][str(c)]['obj'] += sentiment.obj_score()
                    except:
                        print("An exception occurred")
                if tag == 'ADJ':
                    output_dict[str(s)]['tags'][str(c)]['ADJ'] += 1
                    try:
                        sentiment = swn.senti_synset(word + ".a.01")
                        output_dict[str(s)]['sentiments'][str(c)]['pos'] += sentiment.pos_score()
                        output_dict[str(s)]['sentiments'][str(c)]['neg'] += sentiment.neg_score()
                        output_dict[str(s)]['sentiments'][str(c)]['obj'] += sentiment.obj_score()
                    except:
                        print("An exception occurred")
                if tag == 'ADV':
                    output_dict[str(s)]['tags'][str(c)]['ADV'] += 1
                    try:
                        sentiment = swn.senti_synset(word + ".r.01")
                        output_dict[str(s)]['sentiments'][str(c)]['pos'] += sentiment.pos_score()
                        output_dict[str(s)]['sentiments'][str(c)]['neg'] += sentiment.neg_score()
                        output_dict[str(s)]['sentiments'][str(c)]['obj'] += sentiment.obj_score()
                    except:
                        print("An exception occurred")
                if tag == '.':
                    output_dict[str(s)]['tags'][str(c)]['.'] += 1
                if tag == 'ADP':
                    output_dict[str(s)]['tags'][str(c)]['ADP'] += 1
                if tag == 'CONJ':
                    output_dict[str(s)]['tags'][str(c)]['CONJ'] += 1
                if tag == 'DET':
                    output_dict[str(s)]['tags'][str(c)]['DET'] += 1
                if tag == 'NUM':
                    output_dict[str(s)]['tags'][str(c)]['NUM'] += 1
                if tag == 'PRON':
                    output_dict[str(s)]['tags'][str(c)]['PRON'] += 1
                if tag == 'PRT':
                    output_dict[str(s)]['tags'][str(c)]['PRT'] += 1
                if tag == 'X':
                    output_dict[str(s)]['tags'][str(c)]['X'] += 1

# Export output_dict
import json
with open(r'C:\Users\Marco\Desktop\Gits\eg-texttools\output_dicts\output_dict.json', 'w') as fp:
    json.dump(output_dict, fp)
    
text_file = open(r'C:\Users\Marco\Desktop\Gits\eg-texttools\output_dicts\output_dict.txt', "w")
text_file.write(str(output_dict))
text_file.close()

