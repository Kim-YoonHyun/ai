'''image'''
from torchvision.models.segmentation import deeplabv3_resnet101
from torchvision.models.segmentation import fcn_resnet101

import timm
from torch import nn
class GetModel(nn.Module):
    '''
    timm 모듈을 통해 모델을 생성하는 클래스
    '''
    def __init__(self, model_name, pretrained, n_outputs):
        '''
        parameters
        ----------
        model_name: str
            timm 을 통해 불러올 모델 이름
        
        pretrained: bool
            pretrained weight 사용 여부
        
        n_outputs: int
            결과값 갯수
        '''
        super(GetModel, self).__init__()
        if model_name == 'deeplabv3_resnet101':
            self.model = deeplabv3_resnet101(pretrained=pretrained, num_classes=n_outputs)

        # self.model = timm.create_model(model_name, pretrained=pretrained)
        # self.model.classifier = nn.Sequential(
        #     nn.Linear(in_features = 1792, out_features=625),
        #     nn.ReLU(),
        #     nn.Dropout(p=0.3),
        #     nn.Linear(in_features = 625, out_features=512),
        #     nn.ReLU(),
        #     nn.Linear(in_features = 512, out_features=256),
        #     nn.ReLU(),
        #     nn.Linear(in_features=256, out_features=128),
        #     nn.ReLU(),
        #     nn.Linear(in_features=128, out_features=n_outputs)
        # )
        
    def forward(self, x):
        output = self.model(x)
        return output



