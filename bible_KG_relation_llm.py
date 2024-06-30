import pandas as pd
import numpy as np
import networkx as nx
from sklearn.metrics import accuracy_score, precision_score, recall_score

import nltk
from nltk.tokenize import sent_tokenize

import spacy
from spacy.matcher import Matcher

import json, pickle
from datasets import Dataset, DatasetDict

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, Trainer, TrainingArguments

# Load pre-trained NLP model
print("Load pre-trained NLP model")
nlp = spacy.load('en_core_web_trf')

print("Load Bible texts")
kjv_content = pd.read_csv('Bible_dataset/t_kjv.csv')

print("Processing texts")
verse_list = kjv_content['t'].tolist()

sentences = []
for verse in verse_list:
    sentence_list = sent_tokenize(verse)
    sentences.extend(sentence_list)

sentence_docs = [doc for doc in nlp.pipe(sentences)]

def get_entity_pairs(sent):
    ## chunk 1
    ent1 = ""
    ent2 = ""

    prv_tok_dep = ""    # dependency tag of previous token in the sentence
    prv_tok_text = ""   # previous token in the sentence

    prefix = ""
    modifier = ""

    #############################################################
    for tok in nlp(sent):
        ## chunk 2
        # if token is a punctuation mark then move on to the next token
        if tok.dep_ != "punct":
            # check: token is a compound word or not
            if tok.dep_ == "compound":
                prefix = tok.text
                # if the previous word was also a 'compound' then add the current word to it
                if prv_tok_dep == "compound":
                  prefix = prv_tok_text + " "+ tok.text

            # check: token is a modifier or not
            if tok.dep_.endswith("mod") == True:
                modifier = tok.text
                # if the previous word was also a 'compound' then add the current word to it
                if prv_tok_dep == "compound":
                  modifier = prv_tok_text + " "+ tok.text

            ## chunk 3
            if tok.dep_.find("subj") == True:
                ent1 = modifier +" "+ prefix + " "+ tok.text
                prefix = ""
                modifier = ""
                prv_tok_dep = ""
                prv_tok_text = ""      

            ## chunk 4
            if tok.dep_.find("obj") == True:
                ent2 = modifier +" "+ prefix +" "+ tok.text

            ## chunk 5  
            # update variables
            prv_tok_dep = tok.dep_
            prv_tok_text = tok.text
    #############################################################

    return [ent1.strip(), ent2.strip()]

def get_relation(sent):
    doc = nlp(sent)

    # Matcher class object 
    matcher = Matcher(nlp.vocab)

    #define the pattern 
    pattern = [{'DEP':'ROOT'}, 
            {'DEP':'prep','OP':"?"},
            {'DEP':'agent','OP':"?"},  
            {'POS':'ADJ','OP':"?"}] 

    matcher.add("matching_1",[pattern]) 

    matches = matcher(doc)
    k = len(matches) - 1

    span = doc[matches[k][1]:matches[k][2]] 

    return(span.text)

entity_pairs = []
relations = []
llm_input_samples = []
for sent in sentences:
    ents = get_entity_pairs(sent)
    entity_pairs.append(ents)
    rel = get_relation(sent)
    relations.append(rel)
    sample = {
                'instruction': 'Please find the entity pair and extract the relation in the input sentence.', 
                'input': sent, 'output': (ents, rel)
             }
    llm_input_samples.append(sample)

print("Converting to Hugging Face format")
llm_input_dataset = Dataset.from_list(llm_input_samples)

print("Splitting training/validation data")
llm_input_dataset_split = llm_input_dataset.train_test_split(test_size=0.2)

with open('bible_relation_extraction.json','w') as f:
    json.dump(llm_input_samples,f)