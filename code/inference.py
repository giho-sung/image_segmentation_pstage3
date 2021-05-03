import torch
import albumentations as A
from torch.utils.data import Dataset, DataLoader

from importlib import import_module, reload
import os
import pandas as pd
import numpy as np
from utils import label_accuracy_score
import cv2

# 시각화를 위한 라이브러리
import matplotlib
import matplotlib.pyplot as plt


import argparse
import json
import requests
from urllib.parse import urlparse, parse_qsl, urlencode, urlunparse

def load_model(device, num_classes, args):
    
    # load model skeleton
    model_module = getattr(import_module('model'), args.model)
    model = model_module(num_classes=num_classes, args=args)  
    
    # best model 저장된 경로
    model_path = os.path.join(args.saved_dir, args.model + '_best_model.pt')
    
    # best model 불러오기
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)
    
    return model

    
@torch.no_grad()
def inference(model, data_loader, device):
    '''
    '''

    
    size = 256
    transform = A.Compose([A.Resize(256, 256)])
    print('Start prediction.')
    model.eval()
    
    file_name_list = []
    preds_array = np.empty((0, size*size), dtype=np.long)
    
    with torch.no_grad():
        for step, (imgs, image_infos) in enumerate(data_loader):

            # inference (512 x 512)
            outs = model(torch.stack(imgs).to(device))
            oms = torch.argmax(outs.squeeze(), dim=1).detach().cpu().numpy()
            
            # resize (256 x 256)
            temp_mask = []
            for img, mask in zip(np.stack(imgs), oms):
                transformed = transform(image=img, mask=mask)
                mask = transformed['mask']
                temp_mask.append(mask)

            oms = np.array(temp_mask)
            
            oms = oms.reshape([oms.shape[0], size*size]).astype(int)
            preds_array = np.vstack((preds_array, oms))
            
            file_name_list.append([i['file_name'] for i in image_infos])
    print("End prediction.")
    file_names = [y for x in file_name_list for y in x]
    
    return file_names, preds_array



def save_submission_csv(submission, args):

    # submission.csv로 저장
    output_file = args.model + '.csv'
    output_file_path = os.path.join(args.submission_dir, output_file)
    submission.to_csv(output_file_path, index=False)
    
    return submission, output_file_path
    
def submit(user_key='', file_path = '', desc=""):
    if not user_key:
        raise Exception("No UserKey" )
    url = urlparse('http://ec2-13-124-161-225.ap-northeast-2.compute.amazonaws.com:8000/api/v1/competition/28/presigned_url/?description=&hyperparameters={%22training%22:{},%22inference%22:{}}')
    qs = dict(parse_qsl(url.query))
    qs['description'] = desc
    parts = url._replace(query=urlencode(qs))
    url = urlunparse(parts)

    print(url)
    headers = {
        'Authorization': user_key
    }
    res = requests.get(url, headers=headers)
    print(res.text)
    data = json.loads(res.text)
    
    submit_url = data['url']
    body = {
        'key':'app/Competitions/000028/Users/{}/Submissions/{}/output.csv'.format(str(data['submission']['user']).zfill(8),str(data['submission']['local_id']).zfill(4)),
        'x-amz-algorithm':data['fields']['x-amz-algorithm'],
        'x-amz-credential':data['fields']['x-amz-credential'],
        'x-amz-date':data['fields']['x-amz-date'],
        'policy':data['fields']['policy'],
        'x-amz-signature':data['fields']['x-amz-signature']
    }
    requests.post(url=submit_url, data=body, files={'file': open(file_path, 'rb')})
    
    
    
    
if __name__=='__main__':
    
    # -- args
    parser = argparse.ArgumentParser()
    
    def str2bool(str):
        if str == 'True':
            return True
        return False

    # from_only_config ignores all argments
    parser.add_argument('--from_only_config', type=str2bool, default=False, help='it loads argments only from config.json (default: False)')
    parser.add_argument('--config_path', type=str, default='/opt/ml/code/config.json', help='config_path (default: /opt/ml/code/config.json)')

    # Data and model checkpoints directories
    parser.add_argument('--dataset', type=str, default='CustomDataset', help='dataset (default: CustomDataset)')
    parser.add_argument('--dataset_dir', type=str, default='/opt/ml/input/data', help='dataset_dir (default: /opt/ml/input/data)')
    parser.add_argument('--train_augmentation', type=str, default='BaseAugmentation', help='train_augmentation (default: BaseAugmentation)')
    parser.add_argument('--val_augmentation', type=str, default='BaseAugmentation', help='val_augmentation (default: BaseAugmentation)')
    parser.add_argument('--test_augmentation', type=str, default='TestAugmentation', help='test_augmentation (default: TestAugmentation)')
    parser.add_argument('--model', type=str, default='FCN8s', help='model (default: FCN8s)')
    parser.add_argument('--encoder', type=str, default='resnet101', help='encoder (default: resnet101)')
    parser.add_argument('--criterion', type=str, default='cross_entropy', help='criterion (default: cross_entropy)')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer (default: Adam)')
    parser.add_argument('--lr', type=int, default=1e-4, help='lr (default: 1e-4)')
    parser.add_argument('--scheduler', type=str, default='', help='scheduler (default: '')')
    parser.add_argument('--batch_size', type=int, default=16, help='batch_size (default: 16)')
    parser.add_argument('--random_seed', type=int, default=21, help='random_seed (default: 21)')
    parser.add_argument('--epochs', type=int, default=20, help='epochs (default: 20)')
    parser.add_argument('--val_every', type=int, default=1, help='val_every (default: 1)')
    
    parser.add_argument('--saved_inference_config_path', type=str, default='/opt/ml/code/inference_config.json', help='saved_inference_config_path (default: /opt/ml/code/inference_config.json)')
    parser.add_argument('--saved_dir', type=str, default='/opt/ml/code/saved', help='saved_dir (default: /opt/ml/code/saved)')
    parser.add_argument('--submission_dir', type=str, default='/opt/ml/code/submission', help='submission_dir (default: /opt/ml/code/submission)')
    parser.add_argument('--submission_user_key', type=str, default='', help='submission_user_key (default: '')')
    
    parser.add_argument('--is_wandb', type=int, default=1, help='is_wandb (default: 1)')
    parser.add_argument('--wandb_project_name', type=str, default='pstage3_image_segmentation', help='wandb_project_name (default: pstage3_image_segmentation)')
    parser.add_argument('--wandb_group', type=str, default='experiments_group_name', help='wandb_group (default: experiments_group_name)')
    parser.add_argument('--wandb_experiment_name', type=str, default='', help='wandb_experiment_name (default: '')')
  
    args, unknown = parser.parse_known_args()
    
    # put arguments from (inference) config_file            
    print(f'from_only_config: {args.from_only_config}')
    if args.from_only_config:
        # load (inference) config file
        # TODO 파일 안 열릴 때 종료하는 예외처리
        with open(args.config_path, 'r') as f:
            config_dict = json.load(f)
        with open(config_dict['saved_inference_config_path'], 'r') as f:
            inference_config_dict = json.load(f)

        args = argparse.Namespace()
        d = vars(args)
        for key, value in inference_config_dict.items():
            d[key] = value    
    # add variable to argments for record
    d = vars(args)

      
    print(args)
    
    
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    
    dataset_module = getattr(import_module("dataset"), args.dataset)
    # -- dataset
    
    test_path =os.path.join(args.dataset_dir, 'test.json')
    test_dataset = dataset_module(
        data_dir=args.dataset_dir,
        anotation_file=test_path,
        mode='test'
    )
    num_classes = test_dataset.num_classes  # 12
    category_names = test_dataset.category_names
    
    # -- augmentation
    
    transform_module = getattr(import_module("dataset"), args.test_augmentation)
    test_transform = transform_module()
    test_dataset.set_transform(test_transform)
    
    # -- data Loader
    # collate_fn needs for batch
    def collate_fn(batch):
        return tuple(zip(*batch))    
    
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=args.batch_size,
                                          pin_memory=use_cuda,
                                          num_workers=4,
                                          drop_last=True,
                                          collate_fn=collate_fn)
    
    # -- model

    model = load_model(device, num_classes, args).to(device)
    
    # TODO image show
    
    # -- inference
    file_names, preds = inference(model, test_loader, device)
    
    # -- submission 
    if not os.path.isdir(args.submission_dir):
        os.mkdir(args.submission_dir)

    submission_file_path = os.path.join(args.submission_dir, 'sample_submission.csv')
    # sample_submisson.csv 열기
    submission = pd.read_csv(submission_file_path, index_col=None)

    # PredictionString 대입
    for file_name, string in zip(file_names, preds):
        submission = submission.append({"image_id" : file_name, "PredictionString" : ' '.join(str(e) for e in string.tolist())}, 
                                       ignore_index=True)
        
    submission, output_file_path = save_submission_csv(submission, args)
    
    # auto submission
    desc = f'''model:{args.model}, encoder:{args.encoder}, loss:{args.criterion}, optimizer:{args.optimizer}, lr:{args.lr}, epoch:{args.best_epoch}/{args.epochs}, best mIoU:{args.best_mIoU}, batch size:{args.batch_size}\n
            train augmentation:{args.train_augmentation}, val augmentation:{args.val_augmentation}, test augmentation:{args.test_augmentation} '''  # "DECONV not pretrained 20 epoch and no augmentation"  # 수정 필요 : 파일에 대한 설명
    user_key = args.submission_user_key
    # -- submit to server
    submit(user_key, output_file_path, desc)
    
    print(f'submission path: {output_file_path}')
    print('ended inference and submission')
    

