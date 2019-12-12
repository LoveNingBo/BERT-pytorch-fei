import torch.nn as nn
from .gelu import GELU

"""
Transformer Encoder由 6 层相同的层组成，每一层分别由两部分组成：
第一部分是 multi-head self-attention （残差连接+归一化）
第二部分是 position-wise feed-forward network，是一个全连接层 （残差连接+归一化）

embedding后经过位置编码，经过一个Multi-Head Attention，然后残差相加归一化，然后经过一个FeedForward全连接层
然后残差相加归一化，这就是Encoder的全部组成。
事实上，Transformer编码器中没有self-attention模块，只有Multi-Head Attention模块，而Multi-Head Attention
内部计算就是多个self-attention而已，其中self-attention是纯粹的矩阵相乘，没有任何需要学习的参数
"""



class PositionwiseFeedForward(nn.Module):
	def __init__(self,d_model,d_ff,dropout=0.1):
		super(PositionwiseFeedForward,self).__init__()

		self.w_1=nn.Linear(d_model,d_ff)
		self.w_2=nn.Linear(d_ff,d_model)
		self.dropout=nn.Dropout(dropout)
		self.activation=GELU()
	def forward(self,x):
		return self.w_2(self.dropout(self.activation(self.w_1(x))))