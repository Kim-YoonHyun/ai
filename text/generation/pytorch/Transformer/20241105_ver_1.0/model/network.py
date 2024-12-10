import sys
import os
sys.path.append(os.getcwd())
import torch.nn as nn
from layers import embedutils as embm
from layers import encoderutils as encm
from layers import decoderutils as decm
from mylocalmodules import dataloader as dam


class Transformer(nn.Module):
    def __init__(self, device, max_len, vocab_size, d_model, d_ff, n_heads, pad_idx,
                 enc_layer_num, dec_layer_num, dropout_p):
        super(Transformer, self).__init__()
        self.device = device
        self.n_heads = n_heads
        self.pad_idx = pad_idx
        
        # Encoder
        self.encoder = encm.Encoder(
            layer_num=enc_layer_num, 
            max_len=max_len,
            vocab_size=vocab_size,
            d_model=d_model, 
            d_ff=d_ff,
            n_heads=n_heads, 
            dropout_p=dropout_p
        )
            
        # Decoder
        self.decoder = decm.Decoder(
            layer_num=dec_layer_num,
            max_len=max_len,
            vocab_size=vocab_size,
            d_model=d_model, 
            d_ff=d_ff,
            n_heads=n_heads, 
            dropout_p=dropout_p
        )
        

    def forward(self, x, y):#, enc_self_mask=None, la_mask=None, enc_dec_mask=None):
        # encoder self-mask
        enc_self_mask, _ = dam.get_self_mask(x, self.n_heads, self.pad_idx)
        enc_self_mask = enc_self_mask.to(self.device)

        # encoder
        enc_out = self.encoder(x, enc_self_mask)
        
        # decoder look-ahead mask
        _, la_mask = dam.get_self_mask(y, self.n_heads, self.pad_idx)
        la_mask = la_mask.to(self.device)
        
        # encoder-decoder mask
        enc_dec_mask = dam.get_enc_dec_mask(x, y, self.n_heads, self.pad_idx)
        enc_dec_mask = enc_dec_mask.to(self.device)
        
        # decoder
        output = self.decoder(
            y=y, 
            enc_out=enc_out,
            look_ahead_mask=la_mask,
            enc_dec_mask=enc_dec_mask
        )
        
        return output
    
    
