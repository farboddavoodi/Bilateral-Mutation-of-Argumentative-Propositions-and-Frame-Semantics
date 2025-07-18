import json
import sqlite3
import csv
from tqdm import tqdm
import pandas as pd
import torch
# from google.colab import drive
# import matplotlib.pyplot as plt
import os
import numpy as np
# from sklearn.model_selection import train_test_split
from tokenizers import Encoding
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForTokenClassification, AdamW, get_linear_schedule_with_warmup, Trainer, TrainingArguments,  BatchEncoding, AutoTokenizer,AutoModelForCausalLM, TextStreamer
import datasets
from datasets import Dataset
import xml.etree.ElementTree as ET
import zipfile
import tarfile
import glob
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from unsloth import FastLanguageModel
from unsloth import is_bfloat16_supported
from trl import SFTTrainer
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
from sklearn.model_selection import train_test_split

# from google.colab import drive
# drive.mount('/content/drive')



# def read_json_files(root_dir):
#   dataframes = []
#   for root, dirs, files in os.walk(root_dir):
#       for filename in files:
#           if filename.endswith(".json"):
#               filepath = os.path.join(root, filename)
#               df = pd.read_json(filepath, lines=True)
#               dataframes.append(df)
#   return pd.concat(dataframes)


# def prepare_token_classification_data(json_data):
#     all_roles = []
#     for key, value in json_data.items():
#         predicate = value['predicate']
#         roles = value['roles']
#         verb_index = int(key)
#         roles[verb_index] = "B-V"
#         roles = ['O' if role == '_' else role for role in roles]
#         all_roles.append(roles)
#     return all_roles

# def create_token_label_dataframe_from_columns(df):
#     sentence_ids = df["sentence_id"].tolist()
#     words = df['words'].tolist()
#     predictions = df['predictions'].tolist()
#     df_prepared = pd.DataFrame(columns=["sentence_id", "words", "predictions"])
#     for i, (sentence_id, word, prediction) in enumerate(zip(sentence_ids, words, predictions)):
#         for label in prediction:
#             df_prepared.loc[len(df_prepared)] = [sentence_id, word, label]
#     return df_prepared

# def concatenate_words(words):
#     return " ".join(words)

# def process_label(label, all_words):
#     start = label.get('start')
#     end = label.get('end')
#     if start is not None or end is not None:
#         start = int(start)
#         end = int(end)
#         accumulated_chars = 0
#         start_word_index = 0
#         end_word_index = 0
#         for i, word in enumerate(all_words):
#             word_length = len(word)
#             if accumulated_chars <= start:
#                 start_word_index = i
#             if accumulated_chars >= end:
#                 end_word_index = i - 1
#                 break
#             accumulated_chars += word_length + 1
#         if end_word_index is None:
#             end_word_index = len(all_words) - 1
#         extracted_word = ' '.join(all_words[start_word_index:end_word_index + 1])
#         return start, end, label.get('name'), extracted_word
#     else:
#         return None

# def annotate_sentence(df):
#     sentence = df['Text']
#     words = df['Words']
#     start = df['Start']
#     end = df['End']
#     annotation = df['Annotated Names']
#     annotated_words = df['Extracted Words']
#     result = []
#     char_index = 0
#     for word in words:
#         label = 'O'
#         word_start = char_index
#         word_end = char_index + len(word) - 1
#         if word in annotated_words.split():
#             label = annotation
#         result.append(label)
#         char_index = word_end + 1
#     return result


# def framenet_tags(FN_tags):
#     framenet_pattern = []
#     for FN_tag in FN_tags:
#         tag_type = FN_tag.get('tagType')
#         if tag_type == 5:
#             element = FN_tag.get('element', {})
#             element_name = element.get('name', '')
#             framenet_pattern.append(element_name)
#         else:
#             framenet_pattern.append('O')
#     return framenet_pattern

# def read_xml(xml_file, namespace):
#     table = []
#     for filename in tqdm(xml_file):
#         try:
#             tree = ET.parse(filename)
#             root = tree.getroot()
#             for sentence in root.findall('.//fn:sentence', namespace):
#                 text_element = sentence.find('fn:text', namespace)
#                 if text_element is not None and text_element.text:
#                     all_words = text_element.text.split()
#                     annotation_sets = sentence.findall('.//fn:annotationSet[@status="MANUAL"]', namespace)
#                     for annotation_set in annotation_sets:
#                         layers = annotation_set.findall('.//fn:layer[@name="FE"]', namespace)
#                         for layer in layers:
#                             labels = layer.findall('.//fn:label', namespace)
#                             for label in labels:
#                                 processed_label = process_label(label, all_words)
#                                 if processed_label is not None:
#                                     table.append((text_element.text,) + processed_label)
#         except ET.ParseError as e:
#             print(f"Error parsing XML file: {e}")
#             print(f"XML file Name:" + str(filename)) 
#     return table

# english_propbank_df = read_json_files("/home/data/Tavakoli/p_test/test_l/f_d/datas/en/prob/sentences/")
# english_propbank_df['predictions'] = english_propbank_df['predictions'].apply(lambda json_data : prepare_token_classification_data(json_data))
# english_propbank_df = create_token_label_dataframe_from_columns(english_propbank_df)
# english_propbank_df['sentences'] = english_propbank_df['words'].apply(lambda words : concatenate_words(words))
# english_propbank_df.to_csv("/home/data/Tavakoli/p_test/test_l/f_d/datas/en/prob/sentences/english_propbank_df.csv", index=False)
# # print(english_propbank_df.head())


# xml_folder_path = '/home/data/Tavakoli/p_test/test_l/f_d/datas/en/fram/fndata-1.7/lu/'
# xml_files = glob.glob(xml_folder_path + '/*.xml')
# namespace = {'fn': 'http://framenet.icsi.berkeley.edu'}
# table = read_xml(xml_files, namespace)

# english_framenet_df = pd.DataFrame(table, columns=['Text', 'Start', 'End', 'Annotated Names', 'Extracted Words'])
# english_framenet_df["Words"] = english_framenet_df["Text"].apply(lambda sentence: [word.strip() for word in sentence.split(' ')])
# english_framenet_df['Label'] = english_framenet_df.apply(annotate_sentence, axis=1)
# english_propbank_df.to_csv("/home/data/Tavakoli/p_test/test_l/f_d/datas/en/fram/english_framenet_df.csv", index=False)
# # print(english_framenet_df.head())

# # persian_json_data = '/home/data/Tavakoli/p_test/test_l/f_d/datas/fa/davodi.json'
# # persian_df = pd.read_json(persian_json_data)
# # persian_df = persian_df.drop(['_id', 'sentence', 'frame', 'lexicalUnit', 'status', 'issuer', 'is_active', 'createdAt', 'updatedAt', 'PId', 'lang', 'description', 'lexicalUnitHint', 'reviewer', 'lexicalUnitHelper', 'frameHelper', 'frameName', 'lexicalUnitName'], axis = 1)


# # persian_df['frameNetTags'] = persian_df['frameNetTags'].apply(lambda FN_tags : framenet_tags(FN_tags))
# # persian_df.to_csv('/home/data/Tavakoli/p_test/test_l/f_d/datas/fa/davodi.csv', index=False)
# # persian_df1 = pd.read_csv('/home/data/Tavakoli/p_test/test_l/f_d/datas/fa/davodi.json')
# # print(persian_df1.head())
# # print(persian_df1.shape)


persian_json_data = '/mnt/c/Users/Rpipc/Desktop/llama/farbod/f_d/datas/fa/persian_df_test.csv'
persian_df = pd.read_csv(persian_json_data)


max_seq_length = 4096 
dtype = None 
load_in_4bit = True # 
model_path = '/mnt/c/Users/Rpipc/Desktop/llama/farbod/f_d/fine tuned models/unsloth/checkpoint/SRL_unsloth_Llama-3.2-3B-Instruct-bnb-4bit/checkpoint-16'

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_path, 
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
)

messages = [
    {"role": "system", "content": "How can I help you?"},
    {"role": "user", "content": "You are an expert in the field of Semantic Role Labeling and lexical resources especially PropBank.\
You know anything about how to label sentence tokens with PropBank roles.\
Please use the following text:"+ persian_df['words'][0]+"Here are the propbank roles you have to use for labeling:" + str({'f', 'E', '0', 'X', 'O', ',', 'I', ']', 'U', 'R', '[', 'A', '4', '3', 'S', 'o', 'L', ' ', "'", 'G', 'C', 'T', 'M', 'V', '-', 'e', '_', 'D', 'J', 'P', 'B', 'N', 'r', '2', 'p', 'g', '1'})+"\
Your task is to generate PropBank roles for the provided text.\
The output should be a list of roles in a list format. IF a token does not have any role, put 'O'.\
Make sure that you do NOT use any roles other than the ones I provided in this prompt."},
]

FastLanguageModel.for_inference(model) # Enable native 2x faster inference
text_streamer = TextStreamer(tokenizer)
text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        )
    
input_ids = tokenizer([text], return_tensors="pt").to('cuda:0')

generated_ids = model.generate(
    **input_ids,
    streamer = text_streamer,
    max_new_tokens=100,
    pad_token_id=tokenizer.eos_token_id,  
    # do_sample=True                     
)

generated_ids = [
output_ids[len(input_ids):] for input_ids, output_ids in zip(input_ids.input_ids, generated_ids)
]
print(tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip())
