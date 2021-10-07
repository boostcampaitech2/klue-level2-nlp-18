import pickle as pickle
import os
import pandas as pd
import torch


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

def preprocessing_dataset(dataset, punct):
  """ 처음 불러온 csv 파일을 원하는 형태의 DataFrame으로 변경 시켜줍니다."""
  subject_entity = []
  object_entity = []
  for i,j in zip(dataset['subject_entity'], dataset['object_entity']):
    #i = i[i.find('word')+8: i.find('start_idx')-4]
    #j = j[j.find('word')+8: j.find('start_idx')-4]        

    i = i[1:-1].split(',')[0].split(':')[1]
    j = j[1:-1].split(',')[0].split(':')[1]

    subject_entity.append(i)
    object_entity.append(j)
  out_dataset = pd.DataFrame({'id':dataset['id'], 'sentence':dataset['sentence'],'subject_entity':subject_entity,'object_entity':object_entity,'label':dataset['label'],})
  return out_dataset

def load_data(dataset_dir, punct):
  """ csv 파일을 경로에 맡게 불러 옵니다. """
  pd_dataset = pd.read_csv(dataset_dir)
  dataset = preprocessing_dataset(pd_dataset,punct)

  return dataset

def tokenized_dataset(dataset, tokenizer):
  """ tokenizer에 따라 sentence를 tokenizing 합니다."""
  concat_entity = []
  for e01, e02 in zip(dataset['subject_word'], dataset['object_word']):
    temp = ''
    temp = e01 + '[SEP]' + e02
    concat_entity.append(temp)
  tokenized_sentences = tokenizer(
      concat_entity,
      #token_data,
      list(dataset['sentence']),
      return_tensors="pt",
      padding=True,
      truncation=True,
      max_length=256,
      add_special_tokens=True,
      return_token_type_ids=False
      )
  return tokenized_sentences

def add_special_sentence(dataset) :
    sentence_list = []
    type_list = []
    for sentence, sub_s,sub_e, sub_t, sub_word, obj_s, obj_e, obj_t, obj_word in zip(dataset['sentence'] , dataset['subject_start_idx'], dataset['subject_end_idx'], dataset['subject_type'], dataset['subject_word'] ,dataset['object_start_idx'], dataset['object_end_idx'], dataset['object_type'], dataset['object_word']) :
        sub_s_type = "[S:" + sub_t + "] "
        sub_e_type = " [/S:" + sub_t + "] "
        obj_s_type = "[O:" + obj_t + "] "
        obj_e_type = " [/O:" + obj_t + "] "
        type_list.append(sub_s_type)
        type_list.append(sub_e_type)
        type_list.append(obj_s_type)
        type_list.append(obj_e_type)

        if sub_s < obj_s :
            a = ''
            a += sentence[:sub_s] + sub_s_type + sub_word + sub_e_type + sentence[sub_e+1 : obj_s] + obj_s_type + obj_word + obj_e_type + sentence[obj_e+1:]
            sentence_list.append(a)
        else :
            a = ''
            a += sentence[:obj_s] + obj_s_type + obj_word + obj_e_type + sentence[obj_e+1 : sub_s] + sub_s_type + sub_word + sub_e_type + sentence[sub_e+1:]
            sentence_list.append(a)
    return sentence_list, list(set(type_list))
