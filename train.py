import pickle as pickle
import os
import pandas as pd
import torch
import sklearn
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification, Trainer, TrainingArguments, RobertaConfig, RobertaTokenizer, RobertaForSequenceClassification, BertTokenizer
from load_data import *


def klue_re_micro_f1(preds, labels):
    """KLUE-RE micro f1 (except no_relation)"""
    label_list = ['no_relation', 'org:top_members/employees', 'org:members',
       'org:product', 'per:title', 'org:alternate_names',
       'per:employee_of', 'org:place_of_headquarters', 'per:product',
       'org:number_of_employees/members', 'per:children',
       'per:place_of_residence', 'per:alternate_names',
       'per:other_family', 'per:colleagues', 'per:origin', 'per:siblings',
       'per:spouse', 'org:founded', 'org:political/religious_affiliation',
       'org:member_of', 'per:parents', 'org:dissolved',
       'per:schools_attended', 'per:date_of_death', 'per:date_of_birth',
       'per:place_of_birth', 'per:place_of_death', 'org:founded_by',
       'per:religion']
    no_relation_label_idx = label_list.index("no_relation")
    label_indices = list(range(len(label_list)))
    label_indices.remove(no_relation_label_idx)
    return sklearn.metrics.f1_score(labels, preds, average="micro", labels=label_indices) * 100.0

def klue_re_auprc(probs, labels):
    """KLUE-RE AUPRC (with no_relation)"""
    labels = np.eye(30)[labels]

    score = np.zeros((30,))
    for c in range(30):
        targets_c = labels.take([c], axis=1).ravel()
        preds_c = probs.take([c], axis=1).ravel()
        precision, recall, _ = sklearn.metrics.precision_recall_curve(targets_c, preds_c)
        score[c] = sklearn.metrics.auc(recall, precision)
    return np.average(score) * 100.0

def compute_metrics(pred):
  """ validation을 위한 metrics function """
  labels = pred.label_ids
  preds = pred.predictions.argmax(-1)
  probs = pred.predictions

  # calculate accuracy using sklearn's function
  f1 = klue_re_micro_f1(preds, labels)
  auprc = klue_re_auprc(probs, labels)
  acc = accuracy_score(labels, preds) # 리더보드 평가에는 포함되지 않습니다.

  return {
      'micro f1 score': f1,
      'auprc' : auprc,
      'accuracy': acc,
  }


def label_to_num(label):
  num_label = []
  with open('dict_label_to_num.pkl', 'rb') as f:
    dict_label_to_num = pickle.load(f)
  for v in label:
    num_label.append(dict_label_to_num[v])
  
  return num_label

def pull_out_dictionary(df_input) :
  df = df_input.copy()
  df['subject_entity'] = df['subject_entity'].apply(lambda x: eval(x))
  df['object_entity'] = df['object_entity'].apply(lambda x: eval(x))
  
  df = df.assign(
    subject_word=lambda x: x['subject_entity'].apply(lambda x: x['word']),
    subject_start_idx=lambda x: x['subject_entity'].apply(lambda x: x['start_idx']),
    subject_end_idx=lambda x: x['subject_entity'].apply(lambda x: x['end_idx']),
    subject_type=lambda x: x['subject_entity'].apply(lambda x: x['type']),

    # object_entity
    object_word=lambda x: x['object_entity'].apply(lambda x: x['word']),
    object_start_idx=lambda x: x['object_entity'].apply(lambda x: x['start_idx']),
    object_end_idx=lambda x: x['object_entity'].apply(lambda x: x['end_idx']),
    object_type=lambda x: x['object_entity'].apply(lambda x: x['type']),
  )
  df = df.drop(['subject_entity', 'object_entity'], axis = 1)
  return df

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


def train():
  # load model and tokenizer
  # MODEL_NAME = "bert-base-uncased"



  # 다음에는 이거 한 번 써보자.
  # klue/roberta-large 

  # MODEL_NAME = "klue/bert-base"
  # tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

  # load dataset
  #train_dataset = load_data("../dataset/train/train.csv")
  #train_dataset = pull_out_dictionary(train_dataset)
  # dev_dataset = load_data("../dataset/train/dev.csv") # validation용 데이터는 따로 만드셔야 합니다.

  #train_label = label_to_num(train_dataset['label'].values)
  # dev_label = label_to_num(dev_dataset['label'].values)
  
  train_dataset = pd.read_csv("../dataset/train/train.csv")
  train_dataset = pull_out_dictionary(train_dataset)
  train_label = label_to_num(train_dataset['label'].values)
  
  #token_data, type_data = add_special_sentence(train_dataset)
  #dict_type = dict({'additional_special_tokens': type_data})
  # tokenizing dataset
  
  MODEL_NAME = "klue/roberta-large"

  tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, return_token_type_ids = False)
  #tokenizer.add_special_tokens(dict_type)  
  
  # tokenized_train = tokenized_dataset(train_dataset, tokenizer, token_data)
  tokenized_train = tokenized_dataset(train_dataset, tokenizer)
  # tokenized_dev = tokenized_dataset(dev_dataset, tokenizer)

  # make dataset for pytorch.
  RE_train_dataset = RE_Dataset(tokenized_train, train_label)
  # RE_dev_dataset = RE_Dataset(tokenized_dev, dev_label)

  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

  print(device)
  # setting model hyperparameter
  model_config =  AutoConfig.from_pretrained(MODEL_NAME)
  model_config.num_labels = 30

  model =  AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, config=model_config)
  print(model.config)
  model.parameters
  model.to(device)
  
  # 사용한 option 외에도 다양한 option들이 있습니다.
  # https://huggingface.co/transformers/main_classes/trainer.html#trainingarguments 참고해주세요.
  training_args = TrainingArguments(
    output_dir='./results',          # output directory
    save_total_limit=5,              # number of total save model.
    save_steps= 5000,                 # model saving step.
    num_train_epochs=20,              # total number of training epochs
    learning_rate= 5e-5, #5e-5,               # learning_rate
    per_device_train_batch_size=16,  # batch size per device during training
    per_device_eval_batch_size=16,   # batch size for evaluation
    warmup_steps= 5000,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    logging_steps=100,              # log saving step.
    evaluation_strategy='steps', # evaluation strategy to adopt during training
                                # `no`: No evaluation during training.
                                # `steps`: Evaluate every `eval_steps`.
                                # `epoch`: Evaluate every end of epoch.
    eval_steps = 5000,            # evaluation step.
    load_best_model_at_end = True 
  )
  trainer = Trainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=RE_train_dataset,         # training dataset
    eval_dataset=RE_train_dataset,             # evaluation dataset
    compute_metrics=compute_metrics         # define metrics function
  )

  # train model
  trainer.train()
  model.save_pretrained('./best_model')
def main():
  train()

if __name__ == '__main__':
  main()
