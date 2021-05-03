import os
import random
import time
import json
import warnings 
warnings.filterwarnings('ignore')
import argparse

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from utils import label_accuracy_score, get_IoU_on_class_and_mIoU_on_image
from importlib import import_module, reload
import cv2

import numpy as np
import pandas as pd

# 전처리를 위한 라이브러리
from pycocotools.coco import COCO
import torchvision
import torchvision.transforms as transforms

import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp

from loss import create_criterion


# 시각화를 위한 라이브러리
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import wandb


plt.rcParams['axes.grid'] = False

# seed 고정
def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


# it saves best mIoU
def train(args, model, data_loader, val_loader, criterion, optimizer, device, category_names, val_dataset):
    num_epochs = args.epochs
    val_every = args.val_every
    saved_dir = args.saved_dir
    
    
    print('Start training..')
    # best_loss = np.inf
    bset_mIoU = -1
    val_iou_df = pd.DataFrame({'category':category_names})
    for epoch in range(num_epochs):
        model.train()
        for step, (images, masks, _) in enumerate(data_loader):
            images = torch.stack(images)       # (batch, channel, height, width)
            masks = torch.stack(masks).long()  # (batch, channel, height, width)
            
            # gpu 연산을 위해 device 할당
            images, masks = images.to(device), masks.to(device)
                  
            # inference
            outputs = model(images)
            
            # loss 계산 (cross entropy loss)
            loss = criterion(outputs, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # step 주기에 따른 loss 출력
            if (step + 1) % 25 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(
                    epoch+1, num_epochs, step+1, len(data_loader), loss.item()))
                
            if args.is_wandb:
                wandb.log({"train-loss":loss.item()})
        
        # validation 주기에 따른 loss 출력 및 best model 저장
        if (epoch + 1) % val_every == 0:
            avrg_loss, mIoU, IoU_per_class, image_id_with_low_mIoU_dict = validation(epoch + 1, model, val_loader, criterion, device)         
            if mIoU > bset_mIoU: #avrg_loss < best_loss:
                args.best_epoch = epoch + 1
                print('Best performance at epoch: {}'.format(epoch + 1))
                print('Save model in', saved_dir)
                bset_mIoU = mIoU
                # best_loss = avrg_loss
                args.best_mIoU = mIoU
                save_model(model, saved_dir, file_name=args.model + '_best_model.pt')
            
            if args.is_wandb:
                wandb.log({'val-loss':avrg_loss, 'val-IoU':mIoU}, commit=False)
                # 클래스별 IoU
                val_iou_df['IoU_'+str(epoch)] = IoU_per_class
                # wandb.log({'val-iou':wandb.Table(dataframe=val_iou_df)}, commit=False)
                fig, ax = plt.subplots(figsize=(20,10))
                # 
                val_iou_fig_df = val_iou_df.fillna(0)
                val_iou_fig_df.set_index('category', inplace=True)
                val_iou_fig_df = val_iou_fig_df.T
                val_iou_fig_df = val_iou_fig_df.reset_index()
                val_iou_fig_df = val_iou_fig_df.rename(columns = {'index':'epoch'})
                val_iou_fig_df = val_iou_fig_df.melt('epoch', var_name='Category', value_name='IoU')
                sns.lineplot(data=val_iou_fig_df, x='epoch', y='IoU', hue='Category',
                            ax=ax)
                ax.set_ylim(0,1)
                wandb.log({'val-iou':wandb.Image(fig)}, commit=False)
                
                # keep lowest n file and plot
                keep_n = 10
                image_id_with_low_mIoU_dict = {k: v for k, v in sorted(image_id_with_low_mIoU_dict.items(), key=lambda item: item[1])}
                # get images from file
                image_id_with_low_mIoU = list(image_id_with_low_mIoU_dict.items())[:keep_n]
                fig_shape = (int(np.ceil(keep_n / 2)), 2 * 3)
                fig, axis = plt.subplots(nrows=fig_shape[0], ncols=fig_shape[1], figsize=(fig_shape[0] * 5, fig_shape[1] * 3))
                
                
                # validation dataset module로 image와 mask 불러오기
                for i in range(fig_shape[0]):
                    
                    image_id, mIoU = image_id_with_low_mIoU[i * 2 + 0]
                    image, mask, image_infos = val_dataset[image_id]
                    out = model(torch.unsqueeze(image, 0).to(device))
                    predicted_mask = torch.argmax(out.squeeze(), dim=0).detach().cpu().numpy()
                    file_name = image_infos['file_name']
                    # image1
                    axis[i][0].set_title(f'{file_name}: mIoU-{mIoU:.4}')
                    axis[i][0].imshow(image.permute([1,2,0]))
                    # mask1
                    axis[i][1].set_title(f'{file_name} mask')
                    axis[i][1].imshow(mask)
                    # prediction1
                    axis[i][2].set_title(f'{file_name} predicted mask')
                    axis[i][2].imshow(predicted_mask)                    
                    ######
                    
                    
                    image_id, mIoU = image_id_with_low_mIoU[i * 2 + 1]
                    image, mask, image_infos = val_dataset[image_id]
                    out = model(torch.unsqueeze(image, 0).to(device))
                    predicted_mask = torch.argmax(out.squeeze(), dim=0).detach().cpu().numpy()
                    file_name = image_infos['file_name']
                    # image2
                    axis[i][3].set_title(f'{file_name}: mIoU-{mIoU:.4}')
                    axis[i][3].imshow(image.permute([1,2,0]))
                    # mask1
                    axis[i][4].set_title(f'{file_name} mask')
                    axis[i][4].imshow(mask)
                    # prediction2
                    axis[i][5].set_title(f'{file_name} predicted mask')
                    axis[i][5].imshow(predicted_mask)            
                    ######
                wandb.log({f'val-figure:{epoch + 1}':wandb.Image(fig)})
                
                
                
        
        
        # plot val_iou_df
        # print(val_iou_df)

                
def validation(epoch, model, data_loader, criterion, device):
    print('Start validation #{}'.format(epoch))
    low_mIoU_threshold = 0.3
    model.eval()
    with torch.no_grad():
        total_loss = 0
        cnt = 0
        mIoU_list = []
        IoU_per_class_list = []
        image_id_with_low_mIoU_dict = {}
        for step, (images, masks, image_infos) in enumerate(data_loader):
            
            images = torch.stack(images)       # (batch, channel, height, width)
            masks = torch.stack(masks).long()  # (batch, channel, height, width)

            images, masks = images.to(device), masks.to(device)            

            outputs = model(images)
            loss = criterion(outputs, masks)
            total_loss += loss
            cnt += 1
            
            outputs = torch.argmax(outputs.squeeze(), dim=1).detach().cpu().numpy()
            
            # get IoU for analysis
            _masks = masks.detach().cpu().numpy()
            mIoU = label_accuracy_score(_masks, outputs, n_class=12)[2]
            mIoU_list.append(mIoU)
            iou_per_class, miou_per_image_list = get_IoU_on_class_and_mIoU_on_image(_masks, outputs, n_class=12)
            IoU_per_class_list.append(iou_per_class)
            
            # get file name with low mIoU
            miou_per_image_indices = [index for index, miou in enumerate(miou_per_image_list) if miou < low_mIoU_threshold]
            for image_index in miou_per_image_indices:
                image_id_with_low_mIoU_dict[image_infos[image_index]['id']] = miou_per_image_list[image_index]
                
            
        avrg_loss = total_loss / cnt
        print('Validation #{}  Average Loss: {:.4f}, mIoU: {:.4f}'.format(epoch, avrg_loss, np.mean(mIoU_list)))
        
    return avrg_loss, np.mean(mIoU_list), np.mean(IoU_per_class_list, axis=0), image_id_with_low_mIoU_dict


def save_model(model, saved_dir, file_name='fcn8s_best_model.pt'):
    # 모델 저장 함수 정의
    if not os.path.isdir(saved_dir):                                                           
        os.mkdir(saved_dir)
    check_point = {'net': model.state_dict()}
    output_path = os.path.join(saved_dir, file_name)
    torch.save(model.state_dict(), output_path)
    




if __name__=='__main__':
    
    print('pytorch version: {}'.format(torch.__version__))
    print('GPU 사용 가능 여부: {}'.format(torch.cuda.is_available()))

    print(torch.cuda.get_device_name(0))
    print(torch.cuda.device_count())
    
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
    

    # put arguments from config_file
    print(f'from_only_config: {args.from_only_config}')
    if args.from_only_config:
        # load config file
        # TODO 파일 안 열릴 때 종료하는 예외처리
        with open(args.config_path, 'r') as f:
            config_dict = json.load(f)
        args = argparse.Namespace()
        d = vars(args)
        for key, value in config_dict.items():
            d[key] = value
    
    # add variable to argments for record
    d = vars(args)
    d['best_epoch'] = -1
    d['best_mIoU'] = -1
      
    print(args)
    
    if args.is_wandb:
        if args.wandb_experiment_name == "":
            args.wandb_experiment_name = f'{args.model},enc:{args.encoder},loss:{args.criterion},optm:{args.optimizer},sche:{args.scheduler},bs:{args.batch_size},ep:{args.epochs}'
        wandb.init(project=args.wandb_project_name,
                  group=args.wandb_group,
                  name=args.wandb_experiment_name
                  )
    
    
    seed_everything(args.random_seed)

    # -- setting
    use_cuda = torch.cuda.is_available()
    device = "cuda" if torch.cuda.is_available() else "cpu"   # GPU 사용 가능 여부에 따라 device 정보 저장

    # -- dataset
    train_path = os.path.join(args.dataset_dir, 'train.json')
    val_path = os.path.join(args.dataset_dir, 'val.json')
    test_path =os.path.join(args.dataset_dir, 'test.json')

    dataset_module = getattr(import_module("dataset"), args.dataset)  # default: BaseAugmentation
    train_dataset = dataset_module(
        data_dir=args.dataset_dir,
        anotation_file=train_path,
        mode='train'
    )
    val_dataset = dataset_module(
        data_dir=args.dataset_dir,
        anotation_file=val_path,
        mode='val'
    )
    test_dataset = dataset_module(
        data_dir=args.dataset_dir,
        anotation_file=test_path,
        mode='test'
    )
    num_classes = train_dataset.num_classes  # 12
    category_names = train_dataset.category_names

    # -- augmentation
    transform_module = getattr(import_module("dataset"), args.train_augmentation)
    train_transform = transform_module()
    train_dataset.set_transform(train_transform)

    transform_module = getattr(import_module("dataset"), args.val_augmentation)
    val_transform = transform_module()
    val_dataset.set_transform(val_transform)

    transform_module = getattr(import_module("dataset"), args.test_augmentation)
    test_transform = transform_module()
    test_dataset.set_transform(test_transform)

    # --  DataLoader

    # collate_fn needs for batch
    def collate_fn(batch):
        return tuple(zip(*batch))

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               pin_memory=use_cuda,
                                               num_workers=4,
                                               drop_last=True,
                                               collate_fn=collate_fn)

    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, 
                                             batch_size=args.batch_size,
                                             shuffle=False,
                                             pin_memory=use_cuda,
                                             num_workers=4,
                                             drop_last=True,
                                             collate_fn=collate_fn)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=args.batch_size,
                                              pin_memory=use_cuda,
                                              num_workers=4,
                                              drop_last=True,
                                              collate_fn=collate_fn)
    
    # -- model -- 
    model_module = getattr(import_module('model'), args.model)
    model = model_module(num_classes=num_classes, args=args)
    print(f'model: {args.model}')

    # 구현된 model에 임의의 input을 넣어 output이 잘 나오는지 test
    x = torch.randn([1, 3, 512, 512])
    print("input shape : ", x.shape)
    out = model(x).to(device)
    print("output shape : ", out.size())

    model = model.to(device)
    
    # -- loss --
    # Loss function 정의
    criterion = create_criterion(args.criterion) # nn.CrossEntropyLoss()

    # Optimizer 정의
    opt_module = getattr(import_module("torch.optim"), args.optimizer)  # default: SGD
    optimizer = opt_module(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr,
            weight_decay=1e-6
        )

    train(args, model, train_loader, val_loader, criterion, optimizer, device, category_names, val_dataset)
    
    # save args for inference config
    with open(args.saved_inference_config_path, 'w') as f_json:
        json.dump(vars(args), f_json)

    