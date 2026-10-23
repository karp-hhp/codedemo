import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """位置编码模块
    """
    def forward(self, x):
        pass
    
class MultiHeadAttention(nn.Module):
    """多头注意力机制模块

    Args:
        nn (_type_): _description_
    """
    def forward(self, query, key, value, mask):
        pass
    
class PositionWiseFeedForward(nn.Module):
    """位置前馈网络模块
    """
    def forward(self, x):
        pass
    
# ----编码器核心层-----
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention()
        self.feed_forward = PositionWiseFeedForward()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask):
        # 1.多头自注意力
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # 2.前馈网络
        # self.feed_forward本来是一个对象，但是为什么能像函数一样的调用呢？，是因为PositionWiseFeedForward 继承自nn.Module,
        # 而nn.Module 定义了__call__这个特殊方法，这个方法的作用就是能让实例被 ()调用，就像函数类似, 而PositionWiseFeedForward为什么要写
        # forward 函数呢？因为nn.Module 要求子类需要写的业务逻辑，这个forward函数会在 nn.Module自己定义的__call__中被调用，
        # 所以self.feed_forward(x) 其实调用到了PositionWiseFeedForward.forward(x)函数了； 同理其他的几个对象都是一样的意思
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(attn_output))
        
        return x
    
# ----解码器核心层-----
class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention()
        self.cross_attn = MultiHeadAttention()
        self.feed_forward = PositionWiseFeedForward()
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, encoder_output, src_mask, tgt_mask):
        # 1.掩码多头自注意力（对自己）
        attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # 2. 交叉注意力（对编码器输出）
        cross_attn_output = self.cross_attn(x, encoder_output, encoder_output, src_mask)
        x = self.norm2(x + self.dropout(cross_attn_output))
        
        # 3. 前馈网络
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        
        return x