import timm
import timm2
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
        self.model = timm.create_model(model_name, pretrained=pretrained)
        self.model.classifier = nn.Sequential(
            nn.Linear(in_features = 1792, out_features=625),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(in_features = 625, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features = 512, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=n_outputs)
        )
        
    def forward(self, x):
        '''
        학습 진행 method

        parameters
        ----------
        x: tensor, shape=(batch_size, channel, width, height)
        '''
        output = self.model(x)
        return output


# class Swin_v2_cr_huge_224(nn.Module):
#     def __init__(self, n_outputs:int, **kwargs):
#         super(Swin_v2_cr_huge_224, self).__init__()
   
#         self.model = timm2.models.create_model('swinv2_cr_huge_224', pretrained=True)
#         self.model.head = nn.Sequential(
#             nn.Linear(in_features = 1792, out_features=625),
#             nn.ReLU(),
#             nn.Dropout(p=0.3),
#             nn.Linear(in_features=625, out_features=256),
#             nn.ReLU(),
#             nn.Linear(in_features=256, out_features=n_outputs)
#         )
        
#     def forward(self, x):
#         output = self.model(x)
#         return output


# class Swin_v2_cr_small_224(nn.Module):
#     def __init__(self, n_outputs:int, **kwargs):
#         super(Swin_v2_cr_small_224, self).__init__()
   
#         self.model = timm2.models.create_model('swinv2_cr_small_224', pretrained=True)
#         self.model.head = nn.Sequential(
#             nn.Linear(in_features=768, out_features=256),
#             nn.ReLU(),
#             nn.Dropout(p=0.3),
#             nn.Linear(in_features=256, out_features=128),
#             nn.ReLU(),
#             nn.Linear(in_features=128, out_features=n_outputs)
#         )
        
#     def forward(self, x):
#         output = self.model(x)
#         return output

# F = nn.functional

# class CustomNet(nn.Module):

#     def __init__(self, n_inputs:int, n_outputs:int, **kwargs):
#         """
#         Args:
#             n_input(int): feature 수
#             n_output(int): class 수

#         Notes:
#             fc: fully connected layer
#         """
#         super(CustomNet, self).__init__()
#         self.n_input = n_inputs
#         self.n_output = n_outputs

#         self.linear = nn.Linear(self.n_input, self.n_output)

#     def forward(self, x):
#         output = self.linear(x)
        
#         return output


# def get_model(model_name:str, model_args:dict):
#     if model_name == 'Linear':
#         return CustomNet(**model_args)
#     if model_name == 'effnet':
#         return EffNet(**model_args)
#     if model_name == 'swin_v2_cr_small_224':
#         return Swin_v2_cr_small_224(**model_args)
#     if model_name == 'swin_v2_cr_huge_224':
#         return Swin_v2_cr_huge_224(**model_args)


