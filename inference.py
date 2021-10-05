from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import DataLoader
from load_data import *
import pandas as pd
import torch
import torch.nn.functional as F

import pickle as pickle
import numpy as np
import argparse
from tqdm import tqdm

def inference(model, tokenized_sent, device):
  """
    test dataset을 DataLoader로 만들어 준 후,
    batch_size로 나눠 model이 예측 합니다.
  """
  dataloader = DataLoader(tokenized_sent, batch_size=16, shuffle=False)
  model.eval()
  output_pred = []
  output_prob = []
  for i, data in enumerate(tqdm(dataloader)):
    with torch.no_grad():
      outputs = model(
          input_ids=data['input_ids'].to(device),
          attention_mask=data['attention_mask'].to(device),
          token_type_ids=data['token_type_ids'].to(device)
          )
    logits = outputs[0]
    prob = F.softmax(logits, dim=-1).detach().cpu().numpy()
    logits = logits.detach().cpu().numpy()
    result = np.argmax(logits, axis=-1)

    output_pred.append(result)
    output_prob.append(prob)
  
  return np.concatenate(output_pred).tolist(), np.concatenate(output_prob, axis=0).tolist()

def num_to_label(label):
  """
    숫자로 되어 있던 class를 원본 문자열 라벨로 변환 합니다.
  """
  origin_label = []
  with open('dict_num_to_label.pkl', 'rb') as f:
    dict_num_to_label = pickle.load(f)
  for v in label:
    origin_label.append(dict_num_to_label[v])
  
  return origin_label

def load_test_dataset(dataset_dir, tokenizer,punct):
  """
    test dataset을 불러온 후,
    tokenizing 합니다.
  """
  test_dataset = load_data(dataset_dir,punct)
  test_label = list(map(int,test_dataset['label'].values))
  # tokenizing dataset
  tokenized_test = tokenized_dataset(test_dataset, tokenizer)
  return test_dataset['id'], tokenized_test, test_label

def main(args):
  """
    주어진 dataset csv 파일과 같은 형태일 경우 inference 가능한 코드입니다.
  """
  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
  # load tokenizer
  Tokenizer_NAME = "klue/bert-base"
  tokenizer = AutoTokenizer.from_pretrained(Tokenizer_NAME)
  if not args.punct:
    special_tokens_dict = {'additional_special_tokens': ["<S:PER>","</S:PER>","<S:ORG>","</S:ORG:>","<O:PER>", "<O:ORG>", "<O:DAT>", "<O:LOC>", "<O:POH>", "<O:NOH>","</O:PER>", "</O:ORG>", "</O:DAT>", "</O:LOC>", "</O:POH>", "</O:NOH>"]}
  else:
    special_tokens_dict = {'additional_special_tokens': ["*PER*","@","*ORG*","#","∧PER∧", "∧ORG∧", "∧DAT∧", "∧LOC∧", "∧POH∧", "∧NOH∧"]}
  num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)

  ## load my model
  MODEL_NAME = args.model_dir # model dir.
  model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)
  model.resize_token_embeddings(len(tokenizer))
  model.parameters
  model.to(device)

  ## load test datset
  test_dataset_dir = "../dataset/test/test_data.csv"
  test_id, test_dataset, test_label = load_test_dataset(test_dataset_dir, tokenizer,args.punct)
  Re_test_dataset = RE_Dataset(test_dataset ,test_label)

  ## predict answer
  pred_answer, output_prob = inference(model, Re_test_dataset, device) # model에서 class 추론
  pred_answer = num_to_label(pred_answer) # 숫자로 된 class를 원래 문자열 라벨로 변환.
  
  ## make csv file with predicted answer
  #########################################################
  # 아래 directory와 columns의 형태는 지켜주시기 바랍니다.
  output = pd.DataFrame({'id':test_id,'pred_label':pred_answer,'probs':output_prob,})

  output.to_csv('./prediction/submission_punct.csv', index=False) # 최종적으로 완성된 예측한 라벨 csv 파일 형태로 저장.
  #### 필수!! ##############################################
  print('---- Finish! ----')


def cv_ensemble(args):
  """
    주어진 dataset csv 파일과 같은 형태일 경우 inference 가능한 코드입니다.
  """
  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
  # load tokenizer
  Tokenizer_NAME = "klue/bert-base"
  tokenizer = AutoTokenizer.from_pretrained(Tokenizer_NAME)
  if not args.punct:
    special_tokens_dict = {'additional_special_tokens': ["<S:PER>","</S:PER>","<S:ORG>","</S:ORG:>","<O:PER>", "<O:ORG>", "<O:DAT>", "<O:LOC>", "<O:POH>", "<O:NOH>","</O:PER>", "</O:ORG>", "</O:DAT>", "</O:LOC>", "</O:POH>", "</O:NOH>"]}
  else:
    special_tokens_dict = {'additional_special_tokens': ["*PER*","@","*ORG*","#","∧PER∧", "∧ORG∧", "∧DAT∧", "∧LOC∧", "∧POH∧", "∧NOH∧"]}
  num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)

  ## load test datset
  test_dataset_dir = "../dataset/test/test_data.csv"
  test_id, test_dataset, test_label = load_test_dataset(test_dataset_dir, tokenizer,args.punct)
  Re_test_dataset = RE_Dataset(test_dataset ,test_label)

  MODEL_NAME = args.cv_model_dir 

  ## load my model
  for i in range(args.n_split):
    print("cv #",i+1)

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME+ "/" + str(i))
    model.resize_token_embeddings(len(tokenizer))
    model.parameters
    model.to(device)

    ## predict answer
    pred_answer, output_prob = inference(model, Re_test_dataset, device) # model에서 class 추론

    if i==0:
      cv_prob = np.array(output_prob)
    else:
      cv_prob += np.array(output_prob)
 
  cv_prob/=args.n_split
  pred_answer = [np.argmax(cv_prob[idx], axis=-1) for idx in range(len(cv_prob))]
  pred_answer = num_to_label(pred_answer) # 숫자로 된 class를 원래 문자열 라벨로 변환.
  cv_prob = cv_prob.tolist()
  ## make csv file with predicted answer
  #########################################################
  # 아래 directory와 columns의 형태는 지켜주시기 바랍니다.
  output = pd.DataFrame({'id':test_id,'pred_label':pred_answer,'probs':cv_prob,})

  output.to_csv('./prediction/submission_entity.csv', index=False) # 최종적으로 완성된 예측한 라벨 csv 파일 형태로 저장.
  #### 필수!! ##############################################
  print('---- Finish! ----')

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  
  # model dir
  parser.add_argument('--model_dir', type=str, default="./best_model")
  parser.add_argument('--cv_model_dir', type=str, default="./best_model/cv")
  parser.add_argument('--n_split', type=int, default=5)
  parser.add_argument('--cv', type=bool, default=False)
  parser.add_argument('--punct', type=bool, default=False)
  args = parser.parse_args()
  print(args)
  if args.cv==True:
    cv_ensemble(args)
  else:
    main(args)
