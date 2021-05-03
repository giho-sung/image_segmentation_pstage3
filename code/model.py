import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import segmentation_models_pytorch as smp

# Custom Model Template
class MyModel(nn.Module):
    def __init__(self, num_classes, args=None):
        super().__init__()

        """
        1. 위와 같이 생성자의 parameter 에 num_claases 를 포함해주세요.
        2. 나만의 모델 아키텍쳐를 디자인 해봅니다.
        3. 모델의 output_dimension 은 num_classes 로 설정해주세요.
        """

    def forward(self, x):
        """
        1. 위에서 정의한 모델 아키텍쳐를 forward propagation 을 진행해주세요
        2. 결과로 나온 output 을 return 해주세요
        """
        return x
    

class FCN8s(nn.Module):
    def __init__(self, num_classes, args=None):
        super(FCN8s,self).__init__()
        self.pretrained_model = models.vgg16(pretrained = True)
        features, classifiers = list(self.pretrained_model.features.children()), list(self.pretrained_model.classifier.children())

        self.features_map1 = nn.Sequential(*features[0:17])
        self.features_map2 = nn.Sequential(*features[17:24])
        self.features_map3 = nn.Sequential(*features[24:31])
        
        # Score pool3
        self.score_pool3_fr = nn.Conv2d(256, num_classes, 1)
        
        # Score pool4        
        self.score_pool4_fr = nn.Conv2d(512, num_classes, 1)        
        
        # fc6 ~ fc7
        self.conv = nn.Sequential(nn.Conv2d(512, 4096, kernel_size = 1),
                                  nn.ReLU(inplace=True),
                                  nn.Dropout(),
                                  nn.Conv2d(4096, 4096, kernel_size = 1),
                                  nn.ReLU(inplace=True),
                                  nn.Dropout()
                                  )
        
        # Score
        self.score_fr = nn.Conv2d(4096, num_classes, kernel_size = 1)
        
        # UpScore2 using deconv
        self.upscore2 = nn.ConvTranspose2d(num_classes,
                                           num_classes,
                                           kernel_size=4,
                                           stride=2,
                                           padding=1)
        
        # UpScore2_pool4 using deconv
        self.upscore2_pool4 = nn.ConvTranspose2d(num_classes, 
                                                 num_classes, 
                                                 kernel_size=4,
                                                 stride=2,
                                                 padding=1)
        
        # UpScore8 using deconv
        self.upscore8 = nn.ConvTranspose2d(num_classes, 
                                           num_classes,
                                           kernel_size=16,
                                           stride=8,
                                           padding=4)
    
    def forward(self, x):
        pool3 = h = self.features_map1(x)
        pool4 = h = self.features_map2(h)
        h = self.features_map3(h)
        
        h = self.conv(h)
        h = self.score_fr(h)
       
        score_pool3c = self.score_pool3_fr(pool3)    
        score_pool4c = self.score_pool4_fr(pool4)
        
        # Up Score I
        upscore2 = self.upscore2(h)
        
        # Sum I
        h = upscore2 + score_pool4c
        
        # Up Score II
        upscore2_pool4c = self.upscore2_pool4(h)
        
        # Sum II
        h = upscore2_pool4c + score_pool3c
        
        # Up Score III
        upscore8 = self.upscore8(h)
        
        return upscore8
    
    
    

class FCN16s(nn.Module):
    def __init__(self, num_classes, args=None):
        super(FCN16s,self).__init__()
        self.pretrained_model = models.vgg16(pretrained = True)
        features, classifiers = list(self.pretrained_model.features.children()), list(self.pretrained_model.classifier.children())

        self.features_map1 = nn.Sequential(*features[0:17])
        self.features_map2 = nn.Sequential(*features[17:24])
        self.features_map3 = nn.Sequential(*features[24:31])
                
        # Score pool4        
        self.score_pool4_fr = nn.Conv2d(512, num_classes, 1)        
        
        # fc6 ~ fc7
        self.conv = nn.Sequential(nn.Conv2d(512, 4096, kernel_size = 1),
                                  nn.ReLU(inplace=True),
                                  nn.Dropout(),
                                  nn.Conv2d(4096, 4096, kernel_size = 1),
                                  nn.ReLU(inplace=True),
                                  nn.Dropout()
                                  )
        
        # Score
        self.score_fr = nn.Conv2d(4096, num_classes, kernel_size = 1)
        
        # UpScore2 using deconv
        self.upscore2 = nn.ConvTranspose2d(num_classes,
                                           num_classes,
                                           kernel_size=4,
                                           stride=2,
                                           padding=1)
        

        # UpScore16 using deconv
        self.upscore16 = nn.ConvTranspose2d(num_classes, 
                                           num_classes,
                                           kernel_size=32,
                                           stride=16,
                                           padding=8)
        
    
    def forward(self, x):
        h = self.features_map1(x)
        pool4 = h = self.features_map2(h)
        h = self.features_map3(h)
        
        h = self.conv(h)
        h = self.score_fr(h)
       
        score_pool4c = self.score_pool4_fr(pool4)
        
        # Up Score I
        upscore2 = self.upscore2(h)
        
        # Sum I
        h = upscore2 + score_pool4c
        
        # Up Score II
        upscore16 = self.upscore16(h)
        
        
        return upscore16
    
    
class FCN32s(nn.Module):
    def __init__(self, num_classes, args=None):
        super(FCN32s,self).__init__()
        self.pretrained_model = models.vgg16(pretrained = True)
        features, classifiers = list(self.pretrained_model.features.children()), list(self.pretrained_model.classifier.children())

        self.features_map1 = nn.Sequential(*features[0:17])
        self.features_map2 = nn.Sequential(*features[17:24])
        self.features_map3 = nn.Sequential(*features[24:31])
        
        # fc6 ~ fc7
        self.conv = nn.Sequential(nn.Conv2d(512, 4096, kernel_size = 1),
                                  nn.ReLU(inplace=True),
                                  nn.Dropout(),
                                  nn.Conv2d(4096, 4096, kernel_size = 1),
                                  nn.ReLU(inplace=True),
                                  nn.Dropout()
                                  )
        
        # Score
        self.score_fr = nn.Conv2d(4096, num_classes, kernel_size = 1)
        

        # UpScore16 using deconv
        self.upscore32 = nn.ConvTranspose2d(num_classes, 
                                           num_classes,
                                           kernel_size=64,
                                           stride=32,
                                           padding=16)
        
    
    def forward(self, x):
        h = self.features_map1(x)
        pool4 = h = self.features_map2(h)
        h = self.features_map3(h)
        
        h = self.conv(h)
        h = self.score_fr(h)
               
        # Up Score I
        upscore32 = self.upscore32(h)
                
        
        return upscore32
    
    
class DeconvNet(nn.Module):
    
    def __init__(self, num_classes=12, args=None):
        super(DeconvNet, self).__init__()
        
        def CBR(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
            return nn.Sequential(
                nn.Conv2d(in_channels=in_channels,
                          out_channels=out_channels,
                          kernel_size=kernel_size,
                          stride=stride,
                          padding=padding),
                nn.BatchNorm2d(out_channels),
                nn.ReLU())
        def DCB(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
            return nn.Sequential(
                nn.ConvTranspose2d(in_channels=in_channels,
                                   out_channels=out_channels,
                                   kernel_size=kernel_size,
                                   stride=stride,
                                   padding=padding),
                nn.BatchNorm2d(out_channels),
                nn.ReLU())
        
        # 224 x 224
        # conv1
        self.conv1_1 = CBR(3, 64, 3, 1, 1)
        self.conv1_2 = CBR(64, 64, 3, 1, 1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True, return_indices=True)
        
        # 112 x 112
        # conv2
        self.conv2_1 = CBR(64, 128, 3, 1, 1)
        self.conv2_2 = CBR(128, 128, 3, 1, 1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True, return_indices=True)

        # 56 x 56
        # conv3
        self.conv3_1 = CBR(128, 256, 3, 1, 1)
        self.conv3_2 = CBR(256, 256, 3, 1, 1)
        self.conv3_3 = CBR(256, 256, 3, 1, 1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True, return_indices=True)   

        # 28 x 28
        # conv4
        self.conv4_1 = CBR(256, 512, 3, 1, 1)
        self.conv4_2 = CBR(512, 512, 3, 1, 1)
        self.conv4_3 = CBR(512, 512, 3, 1, 1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True, return_indices=True)   

        # 14 x 14
        # conv5
        self.conv5_1 = CBR(512, 512, 3, 1, 1)
        self.conv5_2 = CBR(512, 512, 3, 1, 1)
        self.conv5_3 = CBR(512, 512, 3, 1, 1)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True, return_indices=True)   
        
        # 7 x 7
        # fc6
        self.fc6 = CBR(512, 4096, 7, 1, 0)
        self.drop6 = nn.Dropout2d(0.5)
        
        # 1 x 1
        # fc7
        self.fc7 = CBR(4096, 4096, 1, 1, 0)
        self.drop7 = nn.Dropout2d(0.5)
        
        # 1 x 1
        # fc6-deconv
        self.fc6_deconv = DCB(4096, 512, 7, 1, 0)
        
        # 7 x 7
        # unpool5
        self.unpool5 = nn.MaxUnpool2d(2, stride=2)
        self.deconv5_1 = DCB(512, 512, 3, 1, 1)
        self.deconv5_2 = DCB(512, 512, 3, 1, 1)
        self.deconv5_3 = DCB(512, 512, 3, 1, 1)
        
        # 14 x 14
        # unpool4
        self.unpool4 = nn.MaxUnpool2d(2, stride=2)
        self.deconv4_1 = DCB(512, 512, 3, 1, 1)
        self.deconv4_2 = DCB(512, 512, 3, 1, 1)
        self.deconv4_3 = DCB(512, 256, 3, 1, 1)        
        
        # 28 x 28
        # unpool3
        self.unpool3 = nn.MaxUnpool2d(2, stride=2)
        self.deconv3_1 = DCB(256, 256, 3, 1, 1)
        self.deconv3_2 = DCB(256, 256, 3, 1, 1)
        self.deconv3_3 = DCB(256, 128, 3, 1, 1)           

        # 56 x 56
        # unpool2
        self.unpool2 = nn.MaxUnpool2d(2, stride=2)
        self.deconv2_1 = DCB(128, 128, 3, 1, 1)
        self.deconv2_2 = DCB(128, 64, 3, 1, 1)

        # 112 x 112
        # unpool1
        self.unpool1 = nn.MaxUnpool2d(2, stride=2)
        self.deconv1_1 = DCB(64, 64, 3, 1, 1)
        self.deconv1_2 = DCB(64, 64, 3, 1, 1)
        
        # 224 x 224
        self.score_fr = nn.Conv2d(64, num_classes, kernel_size=1, stride=1, padding=0 , dilation=1)
        
    def forward(self, x):
        # conv1
        h = self.conv1_1(x)
        h = self.conv1_2(h)
        h, pool1_indices =  self.pool1(h) 
        
        # 112 x 112
        # conv2
        h = self.conv2_1(h)
        h = self.conv2_2(h)
        h, pool2_indices = self.pool2(h)

        # 56 x 56
        # conv3
        h = self.conv3_1(h)
        h = self.conv3_2(h)
        h = self.conv3_3(h)
        h, pool3_indices = self.pool3(h)

        # 28 x 28
        # conv4
        h = self.conv4_1(h)
        h = self.conv4_2(h)
        h = self.conv4_3(h)
        h, pool4_indices = self.pool4(h)

        # 14 x 14
        # conv5
        h = self.conv5_1(h)
        h = self.conv5_2(h)
        h = self.conv5_3(h)
        h, pool5_indices = self.pool5(h)
        
        # 7 x 7
        # fc6
        h = self.fc6(h)
        h = self.drop6(h)
        
        # 1 x 1
        # fc7
        h = self.fc7(h)
        h = self.drop7(h)
        
        # 1 x 1
        # fc6-deconv
        h = self.fc6_deconv(h)
        
        # 7 x 7
        # unpool5
        h = self.unpool5(h, pool5_indices)
        h = self.deconv5_1(h)
        h = self.deconv5_2(h)
        h = self.deconv5_3(h)
        
        # 14 x 14
        # unpool4
        h = self.unpool4(h, pool4_indices)
        h = self.deconv4_1(h)
        h = self.deconv4_2(h)
        h = self.deconv4_3(h)        
        
        # 28 x 28
        # unpool3
        h = self.unpool3(h, pool3_indices)
        h = self.deconv3_1(h)
        h = self.deconv3_2(h)
        h = self.deconv3_3(h)          

        # 56 x 56
        # unpool2
        h = self.unpool2(h, pool2_indices)
        h = self.deconv2_1(h)
        h = self.deconv2_2(h)

        # 112 x 112
        # unpool1
        h = self.unpool1(h, pool1_indices)
        h = self.deconv1_1(h)
        h = self.deconv1_2(h)
        
        # 224 x 224
        h = self.score_fr(h)     
        return h
    

class DeepLabV1(nn.Module):
    def __init__(self, backbone, classifier, upsampling=8):
        super(DeepLabV1, self).__init__()
        '''
        [TODO]

        ''' 

    def forward(self, x):
        '''
        [TODO]

        '''
        
        return out
    

class DeepLabV2(nn.Module):
    ## VGG 위에 ASPP 쌓기
    def __init__(self, backbone, classifier, upsampling=8):
        super(DeepLabV2, self).__init__()
        '''
        [TODO]

        ''' 

    def forward(self, x):
        '''
        [TODO]

        '''
        return out
    
    
class DeepLabV3(nn.Sequential):
    def __init__(self, n_classes, n_blocks, atrous_rates):
        super(DeepLabV3, self).__init__()
        '''
        [TODO]

        ''' 

    def forward(self, x):
        '''
        [TODO]

        '''
        return out
    
    

class DilatedNet(nn.Module):
    def __init__(self, backbone, classifier, context_module):
        super(DilatedNet, self).__init__()
        '''
        [TODO]

        ''' 

    def forward(self, x):
        '''
        [TODO]

        '''
        return x
    

class SegNet(nn.Module):
    def __init__(self, num_classes=12, init_weights=True):
        super(SegNet, self).__init__()
        
        '''
        [TODO]

        ''' 

    def forward(self, x):
        '''
        [TODO]

        '''   
        
        return h
    
    

class UNet(nn.Module):
    def __init__(self, num_classes=12, args=None):
        super(UNet, self).__init__()
        def CBR2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False):
            return nn.Sequential(
                                 nn.Conv2d(in_channels=in_channels,
                                           out_channels=out_channels,
                                           kernel_size=kernel_size,
                                           stride=stride,
                                           padding=padding,
                                           bias=bias),
                                 nn.BatchNorm2d(num_features=out_channels),
                                 nn.ReLU()
                                )
        
            
        # encoder1
        self.enc1_1 = CBR2d(3, 64, 3, 1, 1, False)
        self.enc1_2 = CBR2d(64, 64, 3, 1, 1, False)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        
        # encoder2
        self.enc2_1 = CBR2d(64, 128, 3, 1, 1, False)
        self.enc2_2 = CBR2d(128, 128, 3, 1, 1, False)
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        # encoder3
        self.enc3_1 = CBR2d(128, 256, 3, 1, 1, False)
        self.enc3_2 = CBR2d(256, 256, 3, 1, 1, False)
        self.pool3 = nn.MaxPool2d(kernel_size=2)

        # encoder4
        self.enc4_1 = CBR2d(256, 512, 3, 1, 1, False)
        self.enc4_2 = CBR2d(512, 512, 3, 1, 1, False)
        self.pool4 = nn.MaxPool2d(kernel_size=2)
        
        # encoder5 and decoder 5
        self.enc5_1 = CBR2d(512, 1024, 3, 1, 1, False)
        self.enc5_2 = CBR2d(1024, 1024, 3, 1, 1, False)
        self.upconv4 = nn.ConvTranspose2d(in_channels=1024,
                                         out_channels=512,
                                         kernel_size=2,
                                         stride=2,
                                         bias=True)
        
        # decoder4
        self.dec4_1 = CBR2d(1024, 512, 3, 1, 1, False)
        self.dec4_2 = CBR2d(512, 512, 3, 1, 1, False)
        self.upconv3 = nn.ConvTranspose2d(in_channels=512,
                                         out_channels=256,
                                         kernel_size=2,
                                         stride=2,
                                         bias=True)

        # decoder3
        self.dec3_1 = CBR2d(512, 256, 3, 1, 1, False)
        self.dec3_2 = CBR2d(256, 256, 3, 1, 1, False)
        self.upconv2 = nn.ConvTranspose2d(in_channels=256,
                                         out_channels=128,
                                         kernel_size=2,
                                         stride=2,
                                         bias=True)
        
        # decoder2
        self.dec2_1 = CBR2d(256, 128, 3, 1, 1, False)
        self.dec2_2 = CBR2d(128, 128, 3, 1, 1, False)
        self.upconv1 = nn.ConvTranspose2d(in_channels=128,
                                         out_channels=64,
                                         kernel_size=2,
                                         stride=2,
                                         bias=True)
        
        # decoder1
        self.dec1_1 = CBR2d(128, 64, 3, 1, 1, False)
        self.dec1_2 = CBR2d(64, 64, 3, 1, 1, False)
        # output segmentation map
        self.score_fr = nn.Conv2d(in_channels=64,
                                 out_channels=num_classes,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0,
                                 bias=True)
        
        
    def crop_img(self, in_tensor, out_size):
        '''
        Args:
            in_tensor(tensor) : tensor to be cut
            out_size(int) : size of cut
        '''
        dim1, dim2 = in_tensor.size()[2:]
        out_tensor = in_tensor[:,
                               :,
                               int((dim1 - out_size)/2):int((dim1 + out_size)/2),
                               int((dim2 - out_size)/2):int((dim2 + out_size)/2)]
        return out_tensor
        
        
        
    def forward(self, x):
        
        # encoder1
        h = self.enc1_1(x)
        enc1_2 = self.enc1_2(h)
        h = self.pool1(enc1_2)
        
        # encoder2
        h = self.enc2_1(h)
        enc2_2 = self.enc2_2(h)
        h = self.pool2(enc2_2)

        # encoder3
        h = self.enc3_1(h)
        enc3_2 = self.enc3_2(h)
        h = self.pool3(enc3_2)

        # encoder4
        h = self.enc4_1(h)
        enc4_2 = self.enc4_2(h)
        h = self.pool4(enc4_2)
        
        # encoder5 and decoder 5
        h = self.enc5_1(h)
        h = self.enc5_2(h)
        h = self.upconv4(h)
        
        crop_enc4_2 = self.crop_img(enc4_2, h.size()[2])
        h = torch.cat((h, crop_enc4_2), dim=1)
        
        # decoder4
        h = self.dec4_1(h)
        h = self.dec4_2(h)
        h = self.upconv3(h)
        
        crop_enc3_2 = self.crop_img(enc3_2, h.size()[2])
        h = torch.cat((h, crop_enc3_2), dim=1)
        
        # decoder3
        h = self.dec3_1(h)
        h = self.dec3_2(h)
        h = self.upconv2(h)
        
        crop_enc2_2 = self.crop_img(enc2_2, h.size()[2])
        h = torch.cat((h, crop_enc2_2), dim=1)     
        
        # decoder2
        h = self.dec2_1(h)
        h = self.dec2_2(h)
        h = self.upconv1(h)
        
        crop_enc1_2 = self.crop_img(enc1_2, h.size()[2])
        h = torch.cat((h, crop_enc1_2), dim=1)
        
        # decoder1
        h = self.dec1_1(h)
        h = self.dec1_2(h)
        h = self.score_fr(h)
        
        return h
    
    
    
# 출처 : https://jinglescode.github.io/2019/12/02/biomedical-image-segmentation-u-net-nested/
class conv_block_nested(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch):
        super(conv_block_nested, self).__init__()
        '''
        [TODO]

        ''' 

    def forward(self, x):
        '''
        [TODO]

        '''
        return output

class UNetPlusPlus(nn.Module):

    def __init__(self, in_ch=3, out_ch=1, n1=64, height=512, width=512, supervision=True, args=None):
        super(UNetPlusPlus, self).__init__()

        '''
        [TODO]

        ''' 

    def forward(self, x):
        '''
        [TODO]

        '''
        return output
    
class PSPNet(nn.Module):
    
    def __init__(self, in_channels=3, num_classes=12, args=None):
        super(PSPNet, self).__init__()
        if args:
            encoder_name = args.encoder
        else:
            encoder_name ='efficientnet-b0'
    
        self.model =  smp.PSPNet(encoder_name=encoder_name, in_channels=in_channels, classes=num_classes, encoder_weights='imagenet', activation=None) 
        
    def forward(self, x):
        return self.model(x)
    
class FPN(nn.Module):
    
    def __init__(self, in_channels=3, num_classes=12, args=None):
        super(FPN, self).__init__()
        if args:
            encoder_name = args.encoder
        else:
            encoder_name ='efficientnet-b0'
    
        self.model =  smp.FPN(encoder_name=encoder_name, in_channels=in_channels, classes=num_classes, encoder_weights='imagenet', activation=None) 
        
    def forward(self, x):
        return self.model(x)