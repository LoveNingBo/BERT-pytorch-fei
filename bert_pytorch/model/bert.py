import torch.nn as nn
from .transformer import TransformerBlock
from .embedding import BERTEmbedding

class BERT(nn.Module):
	def __init__(self,vocab_size,hidden=786,n_layer=12,attn_heads=12,dropout=0.1)
		super().__init__()
		self.hidden=hidden
		self.n_layers=n_layers
		self.attn_heads=attn_heads

		self.feed_forward_hidden=hidden*4
		self.embedding=BERTEmbedding(vocab_size=vocab_size,embed_size=hidden)

		self.transformer_blocks=nn.ModuleList([TransformerBlock(hidden,attn_heads,hidden*4,dropout), for _ in range(n_layers)])

	def forward(self,x,segment_info):

		mask=(x>0).unsqueeze(1).repeat(1,x.size(1),1).unsqueeze(1)

		x=slef.embedding(x,segment_info)
		for transformer in self.transformer_blocks:
			x=transformer.forward(x,mask)
		return x


		"""
		看源码的结论是，bert只用到了Transformer中的Encoder模块，并且bert本质就是一个Transformer block的堆栈，层层堆叠
		"""