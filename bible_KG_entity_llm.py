import pandas as pd
import numpy as np
import torch
import spacy
from spacy.matcher import Matcher
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from datasets import Dataset, DatasetDict, load_dataset, load_metric, load_from_disk
from transformers import LlamaForCausalLM, LlamaTokenizer, AutoTokenizer, AutoModelForCausalLM, pipeline, Trainer, TrainingArguments, DataCollatorForSeq2Seq

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

verse_ner_dataset = []
for verse_doc in processed_verse_docs:
    tokens, labels = spacy_to_hf_format(verse_doc)
    data = {"tokens": tokens, "annotations": labels}
    verse_ner_dataset.append(data)

llm_input_samples = []
for sent in verse_list:
    doc = nlp(sent)
    tokens, labels = spacy_to_hf_format(doc)
    sample = {
                'instruction': 'Please find the entities in the input sentence.', 
                'input': sent, 'output': f'{[(tokens[j], labels[j]) for j in range(len(tokens))]}'
             }
    llm_input_samples.append(sample)

with open('bible_entity_recognition.json','w') as f:
    json.dump(llm_input_samples,f)

print("Converting to Hugging Face format")
llm_input_dataset = Dataset.from_list(llm_input_samples)

print("Splitting training/validation data")
llm_input_dataset_split = llm_input_dataset.train_test_split(test_size=0.2)


# system_message = """You are a helpful assistant to perform the following task.
# "TASK: the task is to extract entities from a Bible verse."
# "INPUT: the input is a Bible verse."
# "OUTPUT: the output is the annotation of tokens in the input verse."
# """


# def get_ner_from_bible(verse, num_shots=3):
    # messages = [{"role": "system", "content": f"{system_message}"}]
    provide examples
    # for i in range(shot):
        # messages.append({"role": "user", "content": f'{verse_list[i]}'})
        # messages.append({"role": "assistant", "content": f'{[(verse_ner_dataset[i]['tokens'][j], verse_ner_dataset[i]['annotations'][j]) for j in range(len(verse_ner_dataset[i]['tokens']))]}'})
    
    # messages.append({"role": "user", "content": verse})
    # prompt = pipeline.tokenizer.apply_chat_template(
            # messages, 
            # tokenize=False, 
            # add_generation_prompt=True
    # )
    # terminators = [
        # pipeline.tokenizer.eos_token_id,
        # pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    # ]

    # time_start = time.time()
    # outputs = pipeline(
        # prompt,
        # max_new_tokens=2048,
        # eos_token_id=terminators,
        # do_sample=False,
        # pad_token_id=pipeline.tokenizer.eos_token_id
    # )
    # time_end = time.time()

    # return outputs[0]["generated_text"][len(prompt):], (time_end - time_start)

# model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

# tokenizer = AutoTokenizer.from_pretrained(model_id)
# model = LlamaForCausalLM.from_pretrained(
        # base_model,
        # load_in_8bit=True,
        # torch_dtype=torch.float16,
        # device_map="auto",
    # )

# pipeline = transformers.pipeline(
    # "text-generation",
    # model=model_id,
# )

# data_collator = DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True)

# training_args = TrainingArguments(
    # output_dir='./ner_results_llm',
    # evaluation_strategy="epoch",
    # learning_rate=2e-5,
    # per_device_train_batch_size=8,
    # per_device_eval_batch_size=8,
    # num_train_epochs=3,
    # weight_decay=0.01,
# )
# trainer = Trainer(
    # model=model,
    # args=training_args,
    # data_collator=data_collator,
    # train_dataset=llm_input_dataset_split["train"],
    # eval_dataset=llm_input_dataset_split["test"],
    # tokenizer=tokenizer,
# )

# trainer.train()