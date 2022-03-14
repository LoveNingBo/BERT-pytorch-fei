import torch.nn as nn
from .token import TokenEmbedding 
from .position import PositionalEmbedding
from .segment import SegmentEmbedding

class BERTEmbedding(nn.Module):
	"""
    BERT Embedding which is consisted with under features
    1. TokenEmbedding : normal embedding matrix
    2. PositionalEmbedding : adding positional information using sin, cos
    2. SegmentEmbedding : adding sentence segment info, (sent_A:1, sent_B:2)
    sum of all these features are output of BERTEmbedding

	Bert Embedding 由三部分组成，
	TokenEmbedding是单词embedding的结果，
	PositionalEmbedding是位置编码，用sin、cos公式计算
	SegmentEmbedding是用来区分段落的（第一句的编码全为1，第二句的编码全为2，还有如果不够长的补齐编码是0，正好是3。）
	"""

	def __init__(self,vocab_size,embed_size,dropout=0.1):
		super().__init__()
		self.token=TokenEmbedding(vocab_size=vocab_size,embed_size=embed_size)
		self.position=PositionalEmbedding(d_model=self.token.embedding_dim)
		self.segment=SegmentEmbedding(embed_size=self.token.embedding_size)
		self.dropout=nn.Dropout(p=dropout)
		self.embed_size=embed_size

	def forward(self,sequence,segment_label):
		x=self.token(sequenece)+self.position(sequenece)+self.segment(segment_label)
		return self.dropout(x)
