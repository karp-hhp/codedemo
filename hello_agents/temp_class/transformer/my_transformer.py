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
    def __init__(self, d_model, num_heads):
        """_summary_

        Args:
            d_model (_type_): 模型隐藏维度，即每个token的向量长度，典型值为512
            num_heads (_type_): 注意力头数，把d_model 切成几份并行算注意力，典型值为8
            所以d_model 必须能被 num_heads整除，所以你就看到了下面的 assert d_model % num_heads == 0判断
        """
        super(MultiHeadAttention, self).__init__()
        # assert 条件，"错误信息"   即条件为false，就输出 “错误信息”
        assert d_model % num_heads == 0, "d_model 必须能被num_heads整除"
        
        self.d_model = d_model
        self.num_heads = num_heads
        # // 是整除（向下取整），不是 /（返回 float）
        self.d_k = d_model // num_heads
        
        # 定义Q, K, V 和输出的线性变换层
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def scaled_dot_product_attention(self, Q, K, V, mask = None):
        # 1. 计算注意力得分 （QK^T）
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # 2. 应用掩码（如果提供）
        if mask is not None:
            # 将掩码中为0的位置设置为一个非常小的负数，这样softmax后会接近0
            attn_scores = attn_scores.masked_fill(mask==0, -1e9)
            
        # 3. 计算注意力权重(Softmax)
        attn_scores = torch.softmax(attn_scores, dim=-1)
        
        # 4. 加权求和（权重 * V）
        output = torch.matmul(attn_scores, dim=-1)
        return output
    
    def split_heads(self, x):
        # 将输入x的形状从（batch_size, seq_length, d_model）变换成（batch_size, num_heads, seq_length, d_k）
        # batch_size:一个batch里面有几个样本（比如2个句子），seq_length:每个样本有多少个token，d_model:每个token的向量长度（比如512）
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)
    
    def combine_heads(self, x):
        # 将输入x的形状从（batch_size, num_heads, seq_length, d_k）
        # 变回（batch_size, seq_length, d_model）
        batch_size, num_heads, seq_length, d_k = x.size()
        return x.transpose(1,2).contiguous().view(batch_size, seq_length, self.d_model)
    
    def forward(self, query, key, value, mask):
        # 1.对Q，K，V进行线性变换
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))
        
        # 2.计算缩放点积注意力
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # 3. 合并多头输出并进行最终的线性变换
        output = self.W_o(self.combine_heads(attn_output))
        return output
    
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
    