# image_segmentation_pstage3
제한된 서버 환경에서 벗어나 COLAB으로 확장하여 학습 및 추론 기능을 손쉽게 할 수 있도록 함

## 시작하기
- 코드를 clone한 후 google drive에 업로드
- dataset을 다운 후, data 폴더에 넣기

파일 구조
```
.
|-- data
   |-- batch_01_vt
   |-- batch_02_vt
   |-- batch_03
   test.json
   train_all.json
   train.json
   val.json
|-- code
   |-- saved
   |-- submission
      sample_submission.csv
   requirements.txt
   config1_1.json
   config1_2.json
   image_segmentation_experiments_flatform.ipynb
   dataset.py
   model.py
   loss.py
   train.py
   inference.py
   utils.py
   ...
```

- image_segmentation_experiments_flatform.ipynb를 colab으로 열기

## image_segmentation_experiments_flatform.ipynb 사용법
### config 파일 설정
- 각각의 config 파일(config1_1.json, config1_2.json 등)은 하나의 실험에 대응됨
- 원하는 학습, 저장, 모니터링 속성을 config 파일에서 설정 후, image_segmentation_experiments_flatform.ipynb의 내부 변수 <config 파일 path>(config_path1_1 or config_path1_2 등)를 수정

### Command Line Interface(CLI) 기능
```
!pip install -U git+https://github.com/qubvel/segmentation_models.pytorch
!pip install wandb -qqq
!pip install -r requirements.txt
```
위 코드 실행 후 회색 버튼의 `RESTART RUNTIME` 눌러 runtime 초기화 수행
(library 충돌 문제로 런타임 초기화 후 코드를 다시 실행해야 함)

- wandb 로그인 
wandb로 학습 모니터링을 수행하기 위함
```
!wandb login
```

- config 파일 설정 후, 그 path를 arguments로 넘겨주어 학습 및 추론을 수행
```
<config 파일 path> = '/config/file/path/you/set'
!python train.py --from_only_config True --config_path $<config 파일 path>
!python inference.py --from_only_config True --config_path $<config 파일 path>
```

### Line by line 기능
- Jupyter notebook training, Jupyter notebook inference 섹션에서 block 단위로 실행하면 training 및 inference & submission 기능을 이용할 수 있음

## config.json
파일의 속성들

### 학습 관련 속성
- dataset
- dataset_dir
- train_augmentation
- val_augmentation
- test_augmentation
- model
- encoder
- criterion
- optimizer
- lr
- scheduler
- scheduler_parameter
- batch_size
- random_seed
- epochs
- val_every

### 저장 관련 속성
- saved_model_name
- saved_inference_config_path
- saved_dir

### 제출 관련 속성
- submission_dir
- submission_user_key

### 학습 모니터링 관련 속성
- is_wandb
- wandb_project_name
- wandb_group
- wandb_experiment_name

### config 파일 속성 설정 주의 사항
학습시 `saved_model_name`, `wandb_group`를 실험에 맞춰 설정

### config 파일 별 주의사항
각 config 파일마다 `saved_inference_config_path`를 다르게 설정하여야 함


## 학습 모니터링 기능 in wandb
- train loss
- learning rate
- validation loss
- validation mIoU
![trainig, validation info visualization](./sample_images/training_validation_visualization.jpg?raw=true)

- validation에서 각 class별 IoU
![validation IoU per class visualization1](./sample_images/class_IoU_visualization1.jpg?raw=true)
![validation IoU per class visualization2](./sample_images/class_IoU_visualization2.jpg?raw=true)
