#####缩放点积Attention ######

import torch.nn as nn
import torch.nn.functional as F 
import torch

import math

class Attention(nn.Module):
	"""
	Computed Scaled Dot Product Attention 

	torch.mm 是二维张量相乘
	torch.bmm是三维张量相乘
	torch.matmul是高维张量相乘
	"""

	def forward(self,query,key,value,mask=None,dropout=None):
		"""
		q*k/sqrt(q.size(-1))   假设q、k的是维度为1*64的列向量，那么sqrt(q.size(-1))=8
		这样做的目的是使模型在训练过程中具有更加稳定的梯度 /sqrt(q.size(-1)) 并不是唯一选择，经验所得
		假设词向量为512，那么经过 W_q、W_k、W_v三个矩阵的变换可以得到 q、k、v，分别是64维，目的是将来使
		Mutil-head Attention的输出拼接到一起后恢复为512维，Transformer使用8个attention heads   64*8=512
		"""
		scores=torch.matmul(query,key.transpose(-2,-1))/math.sqrt(query.size(-1)) 
		if mask is not None:
			score=score.masked_fill(mask==0,-1e9)

		"""
		假设score 得到了112、96,直接去计算softmax计算量会很大，通过缩放可以减小计算量，同时如果不缩放会使神经元陷入饱和区，梯度更新太小
		计算softmax的时候，是 (e^a)/(e^a+e^b+e^c...)，利用masked_fill使mask为0的地方为负无穷，就可以保证将来softmax后的权重接近0
		"""

		p_attn=F.softmax(score,dim=-1)

		if drop_out is not None:
			p_attn=dropout(p_attn)

        """
        由于scaled_dot_product_attention是self-attention中的一种，可以通过把所有的query拼成一个矩阵，与key拼接成的矩阵相乘，得到softmax的矩阵，
        然后再与value构成的矩阵相乘，就可以得到self-attention的计算结果，所以self-attention 是通过并行计算完成的
        """

		return torch.matmul(p_attn,value),p_attn
 
print("Everything is OK.")

