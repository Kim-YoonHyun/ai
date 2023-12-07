import torch.nn as nn

# Q 네트워크
class DQN(nn.Module):

    def __init__(self, input_num, output_num):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(input_num, 150)
        self.relu1 = nn.ReLU()
        self.layer2 = nn.Linear(150, 100)
        self.relu2 = nn.ReLU()
        self.layer3 = nn.Linear(100, output_num)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu1(x)
        x = self.layer2(x)
        x = self.relu2(x)
        output = self.layer3(x)
        return output