# 1.0

## 1.1.

리뉴얼

# 3.1. (20220901 initial commit)

- 20220928
  - classification - pytorch - kobert
    - mylocalmodules/train.py
      - `train` 함수 부분에 마지막 에포크의 가중치 저장하는 부분 추가

    - mylocalmodulel/utils.py --> 삭제
  
- 20220908
  - classification - pytorch - kobert
    - mylocalmodules/train.py
      - `get_output` 함수 부분에 신뢰도 점수 계산하는 부분 추가

      - `get_output` 함수 부분에 2순위 예측값 계산하는 부분 추가

    - model_test.py
      - 결과 csv 에 신뢰도 점수 계산 추가.

      - 2순위 예측값 표시 부분 추가

- 20220902
  - classification - pytorch - kobert
    - mylocalmodules/train.py
      - train_history 를 매 epoch 마다 저장 하는 방식으로 변경

    - model_test.py
      - 모델 결과 파일의 변경된 저장 경로에 맞춰서 테스트용 코드도 경로설정 변경
      - 테스트도 항상 gpu 를 활용하도록 변경
      - parsing 부분에서 모델 불러오는 변수 외 모든 학습인자 부분 제거
      - 학습인자는 학습된 네트워크에서 지정된 값들을 활용하는 방식으로 변경하고,
        이 값 들은 args_setting.json 파일을 통해 불러오는 방식으로 변경.

- 20220901
  - classification - pytorch - kobert
    - 모델 결과 파일의 전반적인 경로 변경
      - 기존: phase / dataset_model / network / batch_epoch / result.*
      - 변경: phase / dataset_model / condition_order / epoch / result.*
      - ... condition_order / 에는 args_setting.json 파일과 \<datasets>.json 파일도 저장됨.
      - condition 은 args setting 을 의미하며 condition order 가 다르다면 args setting 값이 다르다는 의미임.
    - mylocalmodules/train.py
      - `train` 함수에서 받는 인자값 중 batch_size 제거.
    - mylocalmodules/utils.py
      - `get_save_path` 함수 추가

# 3.0(20220613 initial commit)

- 20220830

  - classification - pytorch - kobert

    - sh.run 파일을 프로젝트 별로 각각 생성
    - main.py
      - train이 시작되기 전 args 정보를 저장
      - dataset json 정보를 train 후 best 결과에만 저장하는 방식에서 train 전 저장하는 방식으로 변경
    - mylocalmodules/train.py
      - `model_test` 함수 추가
    - model_test.py (신규)
      - 학습된 모델 가중치를 활용하여 테스트를 진행하는 코드.
      - 정답 라벨이 있는 경우와 없는 경우를 구분하여 사용하는 부분 추가.


    - data_loader.py
      - `train_val_test_df_separate `  함수에서 test 부분을 제외하고 train 및 validation 데이터로만 나뉘게 변경
      - `train_val_test_df_separate `  train 갯수를 계산할때 올림 방식이 아니라 내림 방식으로 변경

- 20220630

  - classification - pytorch - kobert

    - Dataset 및 Network 부분의 class 함수 대폭 수정.
  - 입력 인수 및 결과 데이터를 얻어내는 알고리즘을 명확히 파악하고 원하는 형태로 조정.
    - string ids, attention mask, segment id, label 값이 로딩 되도록 변경

  - classification - pytorch - bert

    - Dataset 및 Network 부분의 class 함수 대폭 수정.
    - 입력 인수 및 결과 데이터를 얻어내는 알고리즘을 명확히 파악하고 원하는 형태로 조정.
    - string ids, attention mask, segment id, label 값이 로딩 되도록 변경
    - kobert 와 비슷한 형식이 되도록 수정

    

- 20220613

  - classification - pytorch - kobert

    - train.py

      - utils.py 의 함수들을 modelutils.py, trainutils.py, datasetutils.py 로 구분
        ```python
        # my local module
        from mylocalmodules import datasetutils as dutils
        from mylocalmodules import modelutils as mutils
        from mylocalmodules import trainutils as tutils
        from mylocalmodules import utils as lutils
        ```

        

      - parsing 에서 단락 구분 추가
        ```python
        # parsing
        # path parsing
        # data parsing
        # train parsing
        # envs parsing
        ```

        

      - 모든 데이터셋은 xlsx, csv, txt 의 raw data 형태로 입력 받도록 변경. 
        dutils: load_raw_data 함수 추가.
        gutils: make_class_dict 함수 추가.

      - json dataset 화 시키는 과정은 train 코드에 통합.
        dutils: JsonDataset 클래스 삭제.
        dutils: get_dataset 함수 추가. 
        dutils: get_dataloader 간소화. (get_dataset 함수 추가로 인해 tokenizer, max_len 부분 삭제)
        new dataset 은 학습 후 결과 데이터중 하나로써 저장됨.

      - result_df 만드는 코드 추가.

      - test 부분 삭제(향후 추가 예정)

      - 학습 후 저장되는 파일 변경
        상위폴더 이름: \<best_model_name>

        - confusion matrix.csv
        - result_df.csv
        - \<datasetname> .json
        - train_history.json
        - weight.pt

    - trainutils.py
      - get_optimizer, get_loss_function, get_scheduler, get_output, train 함수로 구성된 템플릿 구성 예정.
      - best 값 저장 부분에 confusion matrix 및 예측 결과 라벨 리스트 추가.

## 2.1.2. (20220608 initial commit)

- 20220610
  - classification - pytorch - kobert
    - utils.py 의 함수들을 modelutils.py, trainutils.py, datasetutils.py 로 구분
    - train.py 의 train 함수에서 amp, val_annotation_df 인수 추가.
    - validation 시 정확도 추출 및 confusion matrix, result df 저장 부분 추가(예정)

- 20220608
  - classification - pytorch - kobert
    - train.py 내의 코드 수정
    - 1차 보완 완료.

## 2.1.1. (20220516 initial commit)

- 20220607

  - classification - pytorch - kobert
    - logging 기능 추가.

- 20220531

  - classification-pytorch-kobert
    - csv_to_json.py 를 raw_to_dataset.py 로 변경

    - raw_to_dataset.py
      csv, xlsx, txt 파일을 학습용 데이터셋인 json 형태로 가공하는 코드.

- 20220519

  - classification - pytorch - kobert

    - utils 파일 적용 방식을 함수 하나하나 불러오는 방식에서 utils 자체를 불러오는 방식으로 변경
      ```python
      # my global module
      sys.path.append('/home/kimyh/ai')
      from myglobalmodule import utils as gutils
      
      # my local module
      from mylocalmodule import utils as lutils
      from mylocalmodule import kobertutils as kbutils
      ```

      

- 20220516

  - classification - pytorch - kobert 
    - 전체백업 진행
    - 학습코드
      - train.py
      - predict.py
    - 모듈
      - \__init__.py
      - bertutils.py --> kobertutils.py 로 변경
      - utils.py
    - 실행파일
      - comment_type_predict.sh
      - comment_type_train.sh
    - 데이터 전처리
      - csv_to_json.py

## 2.1. (20220329 initial commit)

- 20220513
  - pytorch - kobert
    - predict.py, comment_type_predict.sh 추가
  
- 20220429
  - pytorch

- 20220329
  - pytorch_bert
    - train.py, predict.py
      - 코드 구조를 함수를 통한 모듈화 진행
      - sh 실행파일을 통한 실행가능하도록 변경

# 2.0 (20220321 initial commit)

- 20220321
  - pytorch_bert
    - train.py
      - argparser 를 통한 옵션 조정 추가

# 1.0(20220221 initial commit)

- 20220316
  - pytorch_bert
    - 전체 백업용 커밋
    - csv_to_json.py 추가
  
- 20220303
  - pytorch_bert
    - 전반적으로 1차 완성형.

- 20220221
  - pytorch_bert
    - trained_model
    - utils
    - make_report_df.py
    - predict.py
    - sample_predict.py
    - train.py

- 20220224
  - pytorch_bert
    - train.py
      - 코드 상에서 데이터셋을 불러오는 코드와 bert 모델을 불러오는 코드의 위치 변경.
      - jupyter 에서의 실행을 염두한 순서로 지정.
      - 코드 마지막 부분에 활용한 데이터셋의 pred data를 통해 자동으로 평가를 진행하는 코드 추가.
      - 평가 결과를 각각 json 및 csv 로 저장하는 코드 추가.
    - predict.py
      - 코드 상에서 데이터셋을 불러오는 코드와 bert 모델을 불러오는 코드의 위치 변경.
      - good, bad 를 각각 나누는 것이 아닌 라벨링 방식으로 추가하여 통합 df를 만드는 것으로 변경
    - make_report_df.py
      - 코드 함수화 하여 train, predict 에 추가.
      - 삭제



