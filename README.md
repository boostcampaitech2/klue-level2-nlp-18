# (Boost Camp AI Tech P-Stage) KLUE
## TEAM NLP-mafia π«

## π λν κ°μ
![image](https://user-images.githubusercontent.com/68593821/136526930-1da880aa-ae38-497c-a312-3b3ffdd97925.png)

λ¬Έμ₯ μμμ λ¨μ΄κ°μ κ΄κ³μ±μ νμνλ κ²μ μλ―Έλ μλλ₯Ό ν΄μν¨μ μμ΄μ λ§μ λμμ μ€λλ€.

κ·Έλ¦Όμ μμμ κ°μ΄ μμ½λ μ λ³΄λ₯Ό μ¬μ©ν΄ QA μμ€ν κ΅¬μΆκ³Ό νμ©μ΄ κ°λ₯νλ©°, μ΄μΈμλ μμ½λ μΈμ΄ μ λ³΄λ₯Ό λ°νμΌλ‘ ν¨μ¨μ μΈ μμ€ν λ° μλΉμ€ κ΅¬μ±μ΄ κ°λ₯ν©λλ€.

κ΄κ³ μΆμΆ(Relation Extraction)μ λ¬Έμ₯μ λ¨μ΄(Entity)μ λν μμ±κ³Ό κ΄κ³λ₯Ό μμΈ‘νλ λ¬Έμ μλλ€. κ΄κ³ μΆμΆμ μ§μ κ·Έλν κ΅¬μΆμ μν ν΅μ¬ κ΅¬μ± μμλ‘, κ΅¬μ‘°νλ κ²μ, κ°μ  λΆμ, μ§λ¬Έ λ΅λ³νκΈ°, μμ½κ³Ό κ°μ μμ°μ΄μ²λ¦¬ μμ© νλ‘κ·Έλ¨μμ μ€μν©λλ€. λΉκ΅¬μ‘°μ μΈ μμ°μ΄ λ¬Έμ₯μμ κ΅¬μ‘°μ μΈ tripleμ μΆμΆν΄ μ λ³΄λ₯Ό μμ½νκ³ , μ€μν μ±λΆμ ν΅μ¬μ μΌλ‘ νμν  μ μμ΅λλ€.

μ΄λ² λνμμλ λ¬Έμ₯, λ¨μ΄μ λν μ λ³΄λ₯Ό ν΅ν΄ ,λ¬Έμ₯ μμμ λ¨μ΄ μ¬μ΄μ κ΄κ³λ₯Ό μΆλ‘ νλ λͺ¨λΈμ νμ΅μν΅λλ€. μ΄λ₯Ό ν΅ν΄ μ°λ¦¬μ μΈκ³΅μ§λ₯ λͺ¨λΈμ΄ λ¨μ΄λ€μ μμ±κ³Ό κ΄κ³λ₯Ό νμνλ©° κ°λμ νμ΅ν  μ μμ΅λλ€. μ°λ¦¬μ modelμ΄ μ λ§ μΈμ΄λ₯Ό μ μ΄ν΄νκ³  μλ μ§, νκ°ν΄ λ³΄λλ‘ ν©λλ€.

### Task
```
sentence: μ€λΌν΄(κ΅¬ μ¬ λ§μ΄ν¬λ‘μμ€νμ¦)μμ μ κ³΅νλ μλ° κ°μ λ¨Έμ  λ§κ³ λ κ° μ΄μ μ²΄μ  κ°λ°μ¬κ° μ κ³΅νλ μλ° κ°μ λ¨Έμ  λ° μ€νμμ€λ‘ κ°λ°λ κ΅¬ν λ²μ μ μ¨μ ν μλ° VMλ μμΌλ©°, GNUμ GCJλ μνμΉ μννΈμ¨μ΄ μ¬λ¨(ASF: Apache Software Foundation)μ νλͺ¨λ(Harmony)μ κ°μ μμ§μ μμ νμ§ μμ§λ§ μ§μμ μΈ μ€ν μμ€ μλ° κ°μ λ¨Έμ λ μ‘΄μ¬νλ€.
subject_entity: μ¬ λ§μ΄ν¬λ‘μμ€νμ¦
object_entity: μ€λΌν΄

relation: λ¨μ²΄:λ³μΉ­ (org:alternate_names)
```
- **input**:Β sentence, subject_entity, object_entity
- **output**:Β relation 30κ° μ€ νλλ₯Ό μμΈ‘ν pred_label, κ·Έλ¦¬κ³  30κ° ν΄λμ€ κ°κ°μ λν΄ μμΈ‘ν νλ₯  probs

### Evaluation
- Micro F1 score
  - micro-precisionκ³Ό micro-recallμ μ‘°ν νκ· μ΄λ©°, κ° μνμ λμΌν importanceλ₯Ό λΆμ¬ν΄, μνμ΄ λ§μ ν΄λμ€μ λ λ§μ κ°μ€μΉλ₯Ό λΆμ¬
  - λ°μ΄ν° λΆν¬μ λ§μ λΆλΆμ μ°¨μ§νκ³  μλ no_relation classλ μ μΈνκ³  F1 scoreκ° κ³μ°

![image](https://user-images.githubusercontent.com/68593821/136528347-dc7cf952-86b9-4d08-9e90-bf24b3e36c6e.png)<br>
![image](https://user-images.githubusercontent.com/68593821/136528364-08bdbdab-a922-48bd-91d3-7b64cfe8aaaa.png)<br>
![image](https://user-images.githubusercontent.com/68593821/136528383-f27d4fa0-b95f-4584-a952-08afdae69d46.png)

## π Data
 - train.csv: μ΄ 32470κ°
 - test_data.csv: μ΄ 7765κ° (μ λ΅ λΌλ²¨ blind = 100μΌλ‘ μμ νν)

**label**<br>
<img src = "https://user-images.githubusercontent.com/68593821/136531490-c15fa28f-7c60-44c6-9b39-b3c306aa8dc3.png" width="700px">

### Data Augmentation
 - subject entity, object entity λ°κΏ data augmentation
 - Augmentation μ΄ν train λ°μ΄ν° : μ΄ 53375κ°

## βοΈ Training

### model
 - klue/bert-base
 - klue/roberta-base and roberta-large and roberta-small
 - kykim/bert-kor-base
 - ainize/klue-bert-base-mrc

### Typed Entity Marker
 - typed entity marker<br>
  ``` <S:PER>μ΄μμ  </S:PER> ```
 - typed entity marker (punct)<br>
   ``` @ * PER * μ΄μμ  @ ```
   
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

## βοΈ Infenrece
### inference.py
```
$ python inference.py \
  --model_dir (model_filepath) \
  --cv_model_dir (cv_model_filepath) \
  --n_split (cv n_split , default=5) \
  --cv (default=False) \
  --punct (Typed Entity Marker(punct) , default=False) \
```
