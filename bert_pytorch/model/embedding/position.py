import torch.nn as nn
import torch 
import math

class PositionalEmbedding(nn.Module):
	def __init__(self,d_model,max_len=512):

		"""
        d_model: 一个标量。模型的维度，论文默认是512
        max_seq_len: 一个标量。文本序列的最大长度
        位置编码与词向量编码维度都是512，是为了能够相加在一起
		Transformer位置编码的每一个维度对应正弦曲线，波长构成了从 2*pi 到 10000*2*pi 的等比数列。
		注意：Bert/Gpt使用的是绝对位置编码，而Transformer使用的相对位置编码

		"""

		super().__init__()

		pe=torch.zero(max_len,d_model).float()
		pe.require_grad=False

		position=torch.arange(0,max_len).float().unsqueenze(1)
		div_term=(  torch.arange(0,d_model,2).float()* -(  math.log(10000.0)  /d_model)  ).exp()

		pe[:,0::2]=torch.sin(position*div_term)
		pe[:,1::2]=torch.cos(position*div_term)

		pe=pe.unsqueeze(0)
		self.register_buffer('pe',pe)

	def forward(self,x):
		return self.pe[:,x.size(1)]
 