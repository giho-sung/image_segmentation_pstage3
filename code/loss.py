import torch
import torch.nn as nn
import torch.nn.functional as F


# https://discuss.pytorch.org/t/is-this-a-correct-implementation-for-focal-loss-in-pytorch/43327/8
class FocalLoss(nn.Module):
    def __init__(self, weight=None,
                 gamma=5., reduction='mean'):
        nn.Module.__init__(self)
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, input_tensor, target_tensor):
        log_prob = F.log_softmax(input_tensor, dim=1)
        prob = torch.exp(log_prob)
        return F.nll_loss(
            ((1 - prob) ** self.gamma) * log_prob,
            target_tensor,
            weight=self.weight,
            reduction=self.reduction
        )


class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes=12, smoothing=0.1, dim=1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))


# https://gist.github.com/SuperShinyEyes/dcc68a08ff8b615442e3bc6a9b55a354
class F1Loss(nn.Module):
    def __init__(self, classes=12, epsilon=1e-7, device='cuda'):
        super().__init__()
        self.classes = classes
        self.epsilon = epsilon
        self.device = device
    def forward(self, y_pred, y_true):
        assert y_pred.ndim == 4
        assert y_true.ndim == 3
        y_true = to_one_hot(y_true, self.classes, self.device)
        y_pred = F.softmax(y_pred, dim=1)

        tp = (y_true * y_pred).sum(dim=0).to(torch.float32)
        tn = ((1 - y_true) * (1 - y_pred)).sum(dim=0).to(torch.float32)
        fp = ((1 - y_true) * y_pred).sum(dim=0).to(torch.float32)
        fn = (y_true * (1 - y_pred)).sum(dim=0).to(torch.float32)

        precision = tp / (tp + fp + self.epsilon)
        recall = tp / (tp + fn + self.epsilon)

        f1 = 2 * (precision * recall) / (precision + recall + self.epsilon)
        f1 = f1.clamp(min=self.epsilon, max=1 - self.epsilon)
        return 1 - f1.mean()

    
def to_one_hot(tensor, nClasses, device):
    n,h,w = tensor.size()
    one_hot = torch.zeros(n,nClasses,h,w).to(device).scatter_(1,tensor.view(n,1,h,w),1)
    return one_hot

class mIoULoss(torch.nn.Module):
    def __init__(self, weight=None, size_average=True, n_classes=12, device='cuda'):
        super(mIoULoss, self).__init__()
        self.classes = n_classes
        self.device = device
    def forward(self, inputs, target):
        # inputs => N x Classes x H x W
        # target_oneHot => N x Classes x H x W
        inputs = inputs.to(self.device)
        target = target.to(self.device)
        SMOOTH = 1e-6
        N = inputs.size()[0]
        inputs = F.softmax(inputs,dim=1)
        target_oneHot = to_one_hot(target, self.classes, self.device)
        # Numerator Product
        inter = inputs * target_oneHot
        ## Sum over all pixels N x C x H x W => N x C
        inter = inter.view(N,self.classes,-1).sum(2) + SMOOTH
        #Denominator 
        union= inputs + target_oneHot - (inputs*target_oneHot)
        ## Sum over all pixels N x C x H x W => N x C
        union = union.view(N,self.classes,-1).sum(2) + SMOOTH
        loss = inter/union
        ## Return average loss over classes and batch
        return 1 -loss.mean()

class UnionMinusIntesectionLoss(torch.nn.Module):
    def __init__(self, weight=None, size_average=True, n_classes=12, target_shape=(12, 512, 512), device='cuda'):
        super(UnionMinusIntesectionLoss, self).__init__()
        self.classes = n_classes
        self.device = device
        self.UILossweight = 1. / (target_shape[1] * target_shape[2] / n_classes)
    def forward(self, inputs, target):
        # inputs => N x Classes x H x W
        # target_oneHot => N x Classes x H x W
        inputs = inputs.to(self.device)
        target = target.to(self.device)
        SMOOTH = 1e-6
        N = inputs.size()[0]
        inputs = F.softmax(inputs,dim=1)
        target_oneHot = to_one_hot(target, self.classes, self.device)
        # Numerator Product
        inter = inputs * target_oneHot
        ## Sum over all pixels N x C x H x W => N x C
        inter = inter.view(N,self.classes,-1).sum(2) + SMOOTH
        #Denominator 
        union= inputs + target_oneHot - (inputs*target_oneHot)
        ## Sum over all pixels N x C x H x W => N x C
        union = union.view(N,self.classes,-1).sum(2) + SMOOTH
        loss = union - inter
        ## Return average loss over classes and batch
        
        return loss.mean() * self.UILossweight
    
class WeightedCrossEntropy(torch.nn.Module):
    '''
    weight is reciprocal ratio on the number of pixel annotation in train_all.json
    '''
    def __init__(self, weight=torch.Tensor([3.05107996e-04, 3.62784246e-01, 9.67408198e-03, 2.34700842e-03,
       2.93149291e-02, 2.36560588e-02, 2.69401083e-02, 7.57726622e-03,
       1.42296838e-02, 1.80541606e-03, 4.80674631e-01, 4.06914633e-02]), device='cuda'):
        super(WeightedCrossEntropy, self).__init__()
        
        weight = weight.to(device)
        self.weighted_cross_entropy = nn.CrossEntropyLoss(weight=weight)
        
    def forward(self, inputs, target):
        return self.weighted_cross_entropy.forward(inputs, target)
        
        
class CombinedCrossEntropyAndMIoULoss(torch.nn.Module):
    def __init__(self):
        super(CombinedCrossEntropyAndMIoULoss ,self).__init__()
        self.CrossEntropyLoss = nn.CrossEntropyLoss()
        self.mIoULoss = mIoULoss()
        
    def forward(self, inputs, target):
        return self.CrossEntropyLoss.forward(inputs, target) + self.mIoULoss.forward(inputs, target)
        

class CombinedCrossEntropyAndF1Loss(torch.nn.Module):
    def __init__(self):
        super(CombinedCrossEntropyAndF1Loss ,self).__init__()
        self.CrossEntropyLoss = nn.CrossEntropyLoss()
        self.F1Loss = F1Loss()
        
    def forward(self, inputs, target):
        return self.CrossEntropyLoss.forward(inputs, target) + self.F1Loss.forward(inputs, target)

    
class CombinedWeightedCrossEntropyAndF1Loss(torch.nn.Module):
    def __init__(self):
        super(CombinedCrossEntropyAndF1Loss ,self).__init__()
        self.WeightedCrossEntropy = WeightedCrossEntropy()
        self.F1Loss = F1Loss()
        
    def forward(self, inputs, target):
        return self.WeightedCrossEntropy.forward(inputs, target) + self.F1Loss.forward(inputs, target)

class CombinedFocalLossAndF1Loss(torch.nn.Module):
    def __init__(self):
        super(CombinedFocalLossAndF1Loss ,self).__init__()
        self.FocalLoss = FocalLoss()
        self.F1Loss = F1Loss()
        
    def forward(self, inputs, target):
        return self.FocalLoss.forward(inputs, target) + self.F1Loss.forward(inputs, target)

class CombinedFocalLossAndUILoss(torch.nn.Module):
    def __init__(self):
        super(CombinedFocalLossAndUILoss ,self).__init__()
        self.FocalLoss = FocalLoss()
        self.UnionMinusIntesectionLoss = UnionMinusIntesectionLoss()
        
        
    def forward(self, inputs, target):
        return self.FocalLoss.forward(inputs, target) + self.UnionMinusIntesectionLoss.forward(inputs, target)    

class CombinedLabelSmoothingLossAndUnionMinusIntesectionLoss(torch.nn.Module):   
    def __init__(self):
        super(CombinedLabelSmoothingLossAndUnionMinusIntesectionLoss ,self).__init__()
        self.LabelSmoothingLoss = LabelSmoothingLoss()
        self.UnionMinusIntesectionLoss = UnionMinusIntesectionLoss()
        
        
    def forward(self, inputs, target):
        return self.LabelSmoothingLoss.forward(inputs, target) + self.UnionMinusIntesectionLoss.forward(inputs, target)   

class CombinedWeightedCrossEntropyAndUnionMinusIntesectionLoss(torch.nn.Module):
    def __init__(self):
        super(CombinedWeightedCrossEntropyAndUnionMinusIntesectionLoss ,self).__init__()
        self.WeightedCrossEntropy = WeightedCrossEntropy()
        self.UnionMinusIntesectionLoss = UnionMinusIntesectionLoss()
        
    def forward(self, inputs, target):
        return self.WeightedCrossEntropy.forward(inputs, target) + self.UnionMinusIntesectionLoss.forward(inputs, target)
    
_criterion_entrypoints = {
    'cross_entropy': nn.CrossEntropyLoss,
    'focal': FocalLoss,
    'label_smoothing': LabelSmoothingLoss,
    'f1': F1Loss,
    'mIoULoss' : mIoULoss,
    'weighted_cross_entropy': WeightedCrossEntropy,
    'cross_entropy_with_mIoU': CombinedCrossEntropyAndMIoULoss,
    'cross_entropy_with_f1': CombinedCrossEntropyAndF1Loss,
    'weighted_cross_entropy_with_f1': CombinedCrossEntropyAndF1Loss,
    'focal_with_f1': CombinedFocalLossAndF1Loss,
    'UILoss': UnionMinusIntesectionLoss,
    'focal_with_UILoss': CombinedFocalLossAndUILoss,
    'label_smoothing_with_UILoss': CombinedLabelSmoothingLossAndUnionMinusIntesectionLoss,
    'weighted_cross_entropy_with_UILoss': CombinedWeightedCrossEntropyAndUnionMinusIntesectionLoss
    
}


def criterion_entrypoint(criterion_name):
    return _criterion_entrypoints[criterion_name]


def is_criterion(criterion_name):
    return criterion_name in _criterion_entrypoints


def create_criterion(criterion_name, **kwargs):
    if is_criterion(criterion_name):
        create_fn = criterion_entrypoint(criterion_name)
        criterion = create_fn(**kwargs)
    else:
        raise RuntimeError('Unknown loss (%s)' % criterion_name)
    return criterion