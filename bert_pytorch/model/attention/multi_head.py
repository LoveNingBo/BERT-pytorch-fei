#####多头注意力机制######
import torch 
from .single import Attention

class MultiHeadedAttention(nn.Module):
	"""
	初始化时需要定义 “头数” 和模型尺寸
	"""

	def __init__(self,h,d_model,dropout=0.1):
		super().__init__()

		assert d_model%h==0  

		"""
		模型的维度需要能够整除 “头数”，例如query为512维，经过线性映射后还是512维，多头数为8，那么每个头的维度为512/8=64
		多头attention就是将 512维的 Q\W\V矩阵经过线形变换，然后在512维度上拆分成8个头，则每个头为64维度，最终每个头再经过
		scaled_dot_product_attention计算得到各自的 八个 V'，然后concat到一起又恢复为512维，所以Transformer输入输出维度相等
		"""
	
		self.d_k=d_model//h
		self.h=h

		self.linear_layers=nn.Modulelist([nn.Linear(d_mdoel,d_model) for _ in range(3)])
		self.output_layer=nn.Linear(d_model,d_model)
		self.attention=Attention()

		self.dropout=nn.Dropout(p=dropout)

	def forward(self,query,key,value,mask=None):

		batch_size=query.size(0)

		# d_model=d_k*h

		query,key,value=[l(x).view(batch_size,-1,self.h,self.d_k).transpose(1,2) for l,x in zip(self.linear_layers,(query,key,value)) ]

		x,atten=self.attention(query,key,value,mask=mask,dropout=self.dropout)

		x=x.transpose(1,2).contiguous().view(batch_size,-1,self.h*self.d_k)

		return self.output_layer(x)


		"""
		ModuleList 和 Sequential 这两种 nn containers，
		ModuleList 就是一个储存各种模块的 list，这些模块之间没有联系，
		没有实现 forward 功能，但相比于普通的 Python list，
		ModuleList 可以把添加到其中的模块和参数自动注册到网络上。
		而Sequential 内的模块需要按照顺序排列，要保证相邻层的输入输出大小相匹配，
		内部 forward 功能已经实现，可以使代码更加整洁。
		"""