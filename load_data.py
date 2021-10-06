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

def preprocessing_dataset_origin(dataset):
  """ 처음 불러온 csv 파일을 원하는 형태의 DataFrame으로 변경 시켜줍니다."""
  subject_entity = []
  object_entity = []
  for i,j in zip(dataset['subject_entity'], dataset['object_entity']):
    i = i[1:-1].split(',')[0].split(':')[1]
    j = j[1:-1].split(',')[0].split(':')[1]

    subject_entity.append(i)
    object_entity.append(j)
  out_dataset = pd.DataFrame({'id':dataset['id'], 'sentence':dataset['sentence'],'subject_entity':subject_entity,'object_entity':object_entity,'label':dataset['label'],})
  return out_dataset

def preprocessing_dataset(dataset,punct):
  """ 처음 불러온 csv 파일을 원하는 형태의 DataFrame으로 변경 시켜줍니다."""
  subject_entity = []
  object_entity = []
  subject_sidx = []
  subject_eidx=[]
  object_sidx=[]
  object_eidx=[]
  subject_rep=[]
  object_rep = []
  for id,(i,j) in enumerate(zip(dataset['subject_entity'], dataset['object_entity'])):
    si = i[1:-1].split(',')
    sj = j[1:-1].split(',')
    i = si[0].split(':')[1]
    j = sj[0].split(':')[1]
    i_s = si[-3].split(':')[1]
    i_e = si[-2].split(':')[1]
    j_s =sj[-3].split(':')[1]
    j_e = sj[-2].split(':')[1]
    i_t = si[-1].split(':')[1]
    j_t = sj[-1].split(':')[1]

    subject_entity.append(i[2:-1])
    object_entity.append(j[2:-1])
    subject_sidx.append(int(i_s))
    subject_eidx.append(int(i_e))
    object_sidx.append(int(j_s))
    object_eidx.append(int(j_e))

    if punct:
      subject_rep.append('@'+"*"+i_t[2:-1]+'*'+i[2:-1]+'@')
      object_rep.append('#'+'∧'+j_t[2:-1]+'∧'+j[2:-1]+'#')
    else:
      subject_rep.append('<S:'+i_t[2:-1]+'>'+i[2:-1]+'</S:'+i_t[2:-1]+'>')
      object_rep.append('<O:'+j_t[2:-1]+'>'+j[2:-1]+'</O:'+j_t[2:-1]+'>')

    
    
  out_dataset = pd.DataFrame({'id':dataset['id'], 'sentence':dataset['sentence'],'subject_entity':subject_entity,'subject_sidx':subject_sidx,'subject_eidx':subject_eidx,'subject_rep':subject_rep,'object_entity':object_entity,'object_sidx':object_sidx,'object_eidx':object_eidx,'object_rep':object_rep,'label':dataset['label'],})
  return out_dataset


def load_data_origin(dataset_dir):
  """ csv 파일을 경로에 맡게 불러 옵니다. """
  pd_dataset = pd.read_csv(dataset_dir)
  dataset = preprocessing_dataset(pd_dataset)
  return dataset

def load_data(dataset_dir,punct):
  """ csv 파일을 경로에 맡게 불러 옵니다. """
  pd_dataset = pd.read_csv(dataset_dir)
  dataset = preprocessing_dataset(pd_dataset,punct)
  for i in range(len(dataset)):
    sent = dataset['sentence'][i]
    sent = sent.replace(dataset['subject_entity'][i], dataset['subject_rep'][i])
    sent = sent.replace(dataset['object_entity'][i], dataset['object_rep'][i])
    dataset.at[i,'sentence']=sent

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
      )
  return tokenized_sentences