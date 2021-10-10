# (Boost Camp AI Tech P-Stage) KLUE
## TEAM NLP-mafia 🔫

## 🔎 대회 개요
![image](https://user-images.githubusercontent.com/68593821/136526930-1da880aa-ae38-497c-a312-3b3ffdd97925.png)

문장 내의 단어 간 관계성을 파악하는 것은 의미나 의도를 해석함에 있어 많은 도움을 줍니다.

그림의 예시와 같이 요약된 정보를 사용하여 QA 시스템을 구축, 활용이 가능하며, 요약된 언어 정보를 바탕으로 효율적인 시스템 및 서비스 구성이 가능합니다.

관계 추출(Relation Extraction)은 문장의 단어(Entity)에 대한 속성과 관계를 예측하는 문제입니다.

이번 대회에서는 문장과 단어에 대한 정보를 통해, 문장 속에서 단어 사이의 관계를 추론하는 모델을 학습시킵니다.


### Task
```
sentence: 오라클(구 썬 마이크로시스템즈)에서 제공하는 자바 가상 머신 말고도 각 운영 체제 개발사가 제공하는 자바 가상 머신 및 오픈소스로 개발된 구형 버전의 온전한 자바 VM도 있으며, GNU의 GCJ나 아파치 소프트웨어 재단(ASF: Apache Software Foundation)의 하모니(Harmony)와 같은 아직은 완전하지 않지만 지속적인 오픈 소스 자바 가상 머신도 존재한다.
subject_entity: 썬 마이크로시스템즈
object_entity: 오라클

relation: 단체:별칭 (org:alternate_names)
```
- **input**: sentence, subject_entity, object_entity
- **output**: relation 30개 중 하나를 예측한 pred_label, 그리고 30개 클래스 각각에 대해 예측한 확률 probs

### Evaluation
- Micro F1 score
  - micro-precision과 micro-recall의 조화 평균이며, 각 샘플에 동일한 importance를 부여해, 샘플이 많은 클래스에 더 많은 가중치를 부여
  - 데이터 분포상 많은 부분을 차지하고 있는 no_relation class는 제외하고 F1 score가 계산

## 📁 Data
 - train.csv: 총 32,470개
 - test_data.csv: 총 7,765개 (정답 라벨 blind = 100으로 임의 표현)

**label**<br>
<img src = "https://user-images.githubusercontent.com/68593821/136531490-c15fa28f-7c60-44c6-9b39-b3c306aa8dc3.png" width="700px">

### Data Augmentation
 - subject entity, object entity 바꿔 data augmentation
 - Augmentation 이후 train 데이터 : 총 53,375개

## ✔️ Training

### model
 - klue/bert-base
 - klue/roberta-base and roberta-large and roberta-small
 - kykim/bert-kor-base
 - ainize/klue-bert-base-mrc

### Typed Entity Marker
 - typed entity marker<br>
  ``` <S:PER>이순신 </S:PER> ```
 - typed entity marker (punct)<br>
   ``` @ * PER * 이순신 @ ```
 - augmented_data 경우, typed entity marker 부착 불가
   
### Stratified K-Fold
### Optuna
 - hyperparameter tuning

### train.py
```
$ python train.py \
  --cv (default=False) \
  --n_split (cv n_split , default=5) \
  --punct (Typed Entity Marker(punct) , default=False) \
```

## ✔️ Infenrece
### inference.py
```
$ python inference.py \
  --model_dir (model_filepath) \
  --cv_model_dir (cv_model_filepath) \
  --n_split (cv n_split , default=5) \
  --cv (default=False) \
  --punct (Typed Entity Marker(punct) , default=False) \
```
- 테스트 결과는 prediction 폴더에 submission_*_*.csv 형태로 저장
