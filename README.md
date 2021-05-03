# image_segmentation_pstage3
제한된 서버 환경에서 벗어나 COLAB으로 확장하여 학습 및 추론 기능을 손쉽게 할 수 있도록 함

## 시작하기
- 코드를 clone한 후 google drive에 업로드
- 데이터를 다운받은 후, data 폴더에 넣기

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
   config.json
   image_segmentation_experiments_flatform.ipynb
   dataset.py
   ...
```

- image_segmentation_experiments_flatform.ipynb를 colab으로 열기
```
# !pip install git+https://github.com/rwightman/pytorch-image-models.git
!pip install -U git+https://github.com/qubvel/segmentation_models.pytorch
!pip install wandb -qqq
!pip install -r requirements.txt
```
위 코드 실행 후 회색 버튼의 런타임 초기화 수행(library 충돌 문제로 런타임 초기화 후 실행해야 함)

- wandb 로그인 
wandb로 학습 모니터링을 수행하기 위함
```
!wandb login
```

- config.json를 수정하고 후, image_segmentation_experiments_flatform.ipynb 내부에 config_path를 수정해야 함. 아래 흩어져 있는 4줄의 코드를 수정해야 함
```
!python train.py --from_only_config True --config_path <config.json path>
!python inference.py --from_only_config True --config_path <config.json path>
config_path = <config.json path>
train_config_path = <config.json path>
```

 
