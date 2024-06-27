import pandas as pd
import numpy as np
import torch
import spacy
from spacy.matcher import Matcher
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from datasets import Dataset, DatasetDict, load_dataset, load_metric, load_from_disk
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline, Trainer, TrainingArguments

nlp = spacy.load('en_core_web_trf')
kjv_content = pd.read_csv('Bible_dataset/t_kjv.csv')

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

df_entities = pd.read_csv('processed_bible_entity.csv')
unique_labels = df_entities['type'].unique().tolist()
id2label = {i: unique_labels[i] for i in range(len(unique_labels))}
label2id = {unique_labels[i]: i for i in range(len(unique_labels))}

def spacy_to_hf_format(doc):
    tokens = [token.text for token in doc]
    labels = ["O"] * len(tokens)
    
    for ent in doc.ents:
        ent_label = ent.label_
        ent_start = ent.start
        ent_end = ent.end
        
        labels[ent_start] = f"B-{ent_label}"
        for i in range(ent_start + 1, ent_end):
            labels[i] = f"I-{ent_label}"
    
    return tokens, labels

def map_labels_to_int(example):
    ner_tags_int = [label2id[label] for label in example['annotations']]
    example['ner_tags'] = ner_tags_int
    return example

verse_ner_dataset = []
for verse_doc in processed_verse_docs:
    tokens, labels = spacy_to_hf_format(verse_doc)
    data = {"tokens": tokens, "annotations": labels}
    verse_ner_dataset.append(data)
    

verse_dataset = Dataset.from_list(verse_ner_dataset)
verse_dataset_split = verse_dataset.train_test_split(test_size=0.2)

verse_dataset_split['train'] = verse_dataset_split['train'].map(map_labels_to_int)
verse_dataset_split['test'] = verse_dataset_split['test'].map(map_labels_to_int)

def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, padding=True, is_split_into_words=True)

    labels = []
    for i, label in enumerate(examples[f"ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their respective word.
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:  # Set the special tokens to -100.
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:  # Only label the first token of a given word.
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

# Finetune models
AVAILABLE_PRETRAINED_MODELS = [
    "bert-base-cased",
    "bert-large-cased",
    "distilbert-base-cased",
    "roberta-base",
    "albert-base-v2",
    "meta-llama/Meta-Llama-3-8B-Instruct"
]

model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=len(unique_labels)) 

if model_name == "roberta-base":
    tokenizer = AutoTokenizer.from_pretrained(model_name, add_prefix_space=True)
else:
    tokenizer = AutoTokenizer.from_pretrained(model_name)

tokenized_verse_dataset_split = verse_dataset_split.map(tokenize_and_align_labels, batched=True)
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

training_args = TrainingArguments(
output_dir='./ner_results',
evaluation_strategy="epoch",
learning_rate=2e-5,
per_device_train_batch_size=32,
per_device_eval_batch_size=32,
num_train_epochs=3,
weight_decay=0.01,
)
trainer = Trainer(
model=model,
args=training_args,
data_collator=data_collator,
train_dataset=tokenized_verse_dataset_split["train"],
eval_dataset=tokenized_verse_dataset_split["test"],
tokenizer=tokenizer,
)

trainer.train()

# 保存模型
model.save_pretrained(f'./ner_model_{model_name}')
tokenizer.save_pretrained(f'./ner_tokenizer_{model_name}')

# 評估模型
eval_metrics = trainer.evaluate()
print(eval_metrics)

# 获取验证集
eval_dataset = tokenized_verse_dataset_split["test"]

# 进行预测
predictions, labels, _ = trainer.predict(eval_dataset)

# 获取预测标签
preds = np.argmax(predictions, axis=2)

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