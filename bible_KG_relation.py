import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score

import nltk
from nltk.tokenize import word_tokenize

import spacy
from spacy.matcher import Matcher

import json, pickle
from datasets import Dataset, DatasetDict

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline, Trainer, TrainingArguments

# Load pre-trained NLP model
print("Load pre-trained NLP model")
nlp = spacy.load('en_core_web_trf')

print("Load Bible texts")
kjv_content = pd.read_csv('Bible_dataset/t_kjv.csv')

print("Processing texts")
verse_list = kjv_content['t'].tolist()
processed_verses = []
for verse in verse_list:
    cleaned_verse = []
    for token in nlp(verse):
        if (not token.is_stop) and (not token.is_punct):
            cleaned_verse.append(token.text)
    clean_verse = ' '.join(cleaned_verse)
    processed_verses.append(clean_verse)

processed_verse_docs = [doc for doc in nlp.pipe(processed_verses)]

# relation extraction rule
def extract_relations(doc):
    relations = []
    for token in doc:
        if token.dep_ == "ROOT" and token.pos_ == "VERB":
            subject = None
            obj = None
            for child in token.children:
                if child.dep_ in ("nsubj", "nsubjpass") and child.ent_type_:
                    subject = child
                if child.dep_ in ("dobj", "pobj") and child.ent_type_:
                    obj = child
            if subject and obj:
                relation = {
                    "head": subject.i,
                    "tail": obj.i,
                    "relation": token.lemma_
                }
                relations.append(relation)
    return relations

print("Finding relations")
data_relations = []
for doc in processed_verse_docs:
    relations = extract_relations(doc)
    tokens = [token.text for token in doc]
    entities = [{"start": ent.start_char, "end": ent.end_char, "label": ent.label_} for ent in doc.ents]
    if len(entities) != 0 and len(relations) != 0:
        data_relations.append({"tokens": tokens, "entities": entities, "relations": relations})

print("Assign IDs to relations")
label_list = []
for rel in data_relations:
    relation_list = rel["relations"]
    for r in relation_list:
        label_list.append(r["relation"])

unique_label_list = list(set(label_list))  # 添加所有可能的关系标签
unique_label_list.append("no_relation")
label_to_id = {label: i for i, label in enumerate(unique_label_list)}
id_to_label = {i: label for i, label in enumerate(unique_label_list)}

# 转换为Dataset格式
print("Converting to Hugging Face format")
dataset = Dataset.from_list(data_relations)

print("Splitting training/validation data")
dataset_split = dataset.train_test_split(test_size=0.2)

# with open(f'rel_dataset.pkl', 'wb') as f:
    # pickle.dump(dataset, f)

def preprocess_function(examples):
    tokenized_inputs = relation_tokenizer(examples['tokens'], is_split_into_words=True, truncation=True, padding=True)
    
    entity_spans = []
    relation_labels = []

    for i in range(len(examples["entities"])):
        entities = examples["entities"][i]
        spans = [(entity["start"], entity["end"]) for entity in entities]
        entity_spans.append(spans)
    for i in range(len(examples["relations"])):    
        relations = examples["relations"][i]
        rels = [label_to_id[relation["relation"]] for relation in relations]
        relation_labels.append(rels)

    tokenized_inputs["entity_spans"] = entity_spans
    tokenized_inputs["labels"] = relation_labels
    
    return tokenized_inputs

AVAILABLE_PRETRAINED_MODELS = [
    "bert-base-cased",
    "bert-large-cased",
    "distilbert-base-cased",
    "roberta-base",
    "albert-base-v2",
    # "meta-llama/Meta-Llama-3-8B-Instruct"
]

# 微調BERT模型
# model_name = "meta-llama/Meta-Llama-3-8B-Instruct"

for model_name in AVAILABLE_PRETRAINED_MODELS:
    relation_model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(unique_label_list))
    relation_tokenizer = AutoTokenizer.from_pretrained(model_name)    
    tokenized_dataset_split = dataset_split.map(preprocess_function, batched=True)
    training_args = TrainingArguments(
        output_dir='./results',
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
    )

    trainer = Trainer(
        model=relation_model,
        args=training_args,
        train_dataset=tokenized_dataset_split['train'],
        eval_dataset=tokenized_dataset_split['test']
    )

    trainer.train()

    # 保存模型
    relation_model.save_pretrained(f'./rel_model_{model_name}')
    relation_tokenizer.save_pretrained(f'./rel_tokenizer_{model_name}')

    # 評估模型
    metrics = trainer.evaluate()
    print(metrics)

    # 获取验证集
    eval_dataset = tokenized_dataset_split['test']

    # 进行预测
    predictions, labels, _ = trainer.predict(eval_dataset)

    # 获取预测标签
    preds = np.argmax(predictions, axis=1)

    # 扁平化列表
    true_labels = labels.flatten()
    pred_labels = preds.flatten()

    # 过滤掉填充的标签（通常是-100）
    mask = true_labels != -100
    true_labels = true_labels[mask]
    pred_labels = pred_labels[mask]

    # 计算指标
    accuracy = accuracy_score(true_labels, pred_labels)
    precision = precision_score(true_labels, pred_labels, average='weighted')
    recall = recall_score(true_labels, pred_labels, average='weighted')

    print(f'Results of fine-tuning {model_name}')
    print(f'Accuracy: {accuracy}')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')