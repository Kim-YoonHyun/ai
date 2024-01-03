import sys
import os
import torch.nn as nn

class LossFunction(nn.Module):
    def __init__(self, gradient_accumulation_steps):
        super().__init__()
        self.gradient_accumulation_steps = gradient_accumulation_steps
        pass
    
    def forward(self, pred_loss, b_label):
        loss = pred_loss
        if self.gradient_accumulation_steps > 1:
            loss = pred_loss / self.gradient_accumulation_steps
            
        return loss
    