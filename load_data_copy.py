import pickle as pickle
import os
import pandas as pd
import torch
from koeda import EasyDataAugmentation
from koeda import AEDA
from koeda import EDA

class RE_Dataset(torch.utils.data.Dataset):
  """ Dataset 구성을 위한 class."""
  def __init__(self, pair_dataset, labels):
    self.pair_dataset = pair_dataset
    self.labels = labels

  def __getitem__(self, idx):
    item = {key: val[idx].clone().detach() for key, val in self.pair_dataset.items()}
    item['labels'] = torch.tensor(self.labels[idx])
    return item

  def __len__(self):
    return len(self.labels)

def preprocessing_dataset(dataset):
  """ 처음 불러온 csv 파일을 원하는 형태의 DataFrame으로 변경 시켜줍니다."""
  subject_entity = []
  object_entity = []
  for i,j in zip(dataset['subject_entity'], dataset['object_entity']):
    i = i[i.find('word')+8: i.find('start_idx')-4]
    j = j[j.find('word')+8: j.find('start_idx')-4]        

    # i = i[1:-1].split(',')[0].split(':')[1]
    # j = j[1:-1].split(',')[0].split(':')[1]

    subject_entity.append(i)
    object_entity.append(j)
  out_dataset = pd.DataFrame({'id':dataset['id'], 'sentence':dataset['sentence'],'subject_entity':subject_entity,'object_entity':object_entity,'label':dataset['label'],})
  return out_dataset

def load_data_generator(dataset_dir):
  augmenter = EDA(
          morpheme_analyzer = "Okt",  # Default = "Okt"
          alpha_sr = 0.1,
          alpha_ri = 0.1,
          alpha_rs = 0.1,
          prob_rd = 0.1
        )
  """ csv 파일을 경로에 맡게 불러 옵니다. """
  pd_dataset = pd.read_csv(dataset_dir)
#  if dataset_dir == ../dataset/train/train.csv :
  id_num = len(pd_dataset)
  for i in range(id_num) :
    new_sentence = augmenter(pd_dataset['sentence'][i])
    new_data = {'id' : id_num +i, 'sentence' : new_sentence, 
    "subject_entity" : pd_dataset[i]['subject_entity'], 
    "object_entity" : pd_dataset[i]['object_entity'], 
    "label" : pd_dataset[i]['label'], 
    "source" : pd_dataset[i]['source'], }

  dataset = preprocessing_dataset(pd_dataset)
  
  return dataset


def load_data(dataset_dir):
  pd_dataset = pd.read_csv(dataset_dir)
  dataset = preprocessing_dataset(pd_dataset)
  
  return dataset  

def tokenized_dataset(dataset, tokenizer):
  """ tokenizer에 따라 sentence를 tokenizing 합니다."""
  concat_entity = []
  for e01, e02 in zip(dataset['subject_entity'], dataset['object_entity']):
    temp = ''
    temp = e01 + '[SEP]' + e02
    concat_entity.append(temp)
  tokenized_sentences = tokenizer(
      concat_entity,
      list(dataset['sentence']),
      return_tensors="pt",
      padding=True,
      truncation=True,
      max_length=256,
      add_special_tokens=True,
      # roberta 쓸 때 사용
      #return_token_type_ids=False
      )
  return tokenized_sentences
