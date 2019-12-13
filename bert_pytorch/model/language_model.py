import torch.nn as nn
from .bert import BERT 


"""
预测下一句这个任务就是一个二分类，将bert输出层的第一个列向量输给一个全连接层，
再加softmax就得到了这句话是下一句话的概率,x[:,0]代表输出层第一个列向量
"""

class NextSentencePrediction(n.Module):
	def __init__(self,hidden):
		super().__init__()
		self.linear=nn.Linear(hidden,2)
		self.softmax=n.LogSoftmax(dim=-1)
	def forward(self,x):
		return self.softmax(self.linear(x[:,0]))

"""
预测mask掉的词，本质上就是一个标注问题，根据当前位置bert输出的隐状态hidden，
输给一个全连接层，多分类问题，类别数就是词表大小，预测当前的词
最后接一个softmax
"""

class MaskedLanguageModel(nn.Module):
	def __init__(self,hidden,vocab_size):
		super().__init__()
		self.linear=nn.Linear(hidden,vocab_size)
		self.softmax=nn.LogSoftmax(dim=-1)
	def forward(self,x):
		return self.softmax(self.linear(x))

class BERTLM(nn.Module):
	def __init__(self,bert:BERT.vocab_size):
		super().__init__()
		self.bert=bert
		self.next_sentence=NextSentencePrediction(self.bert.hidden)
		self.mask_lm=MaskedLanguageModel(self.bert.hidden,vocab_size)

	def forward(self,x,segment_label):
		x=self.bert(x,segment_label)
		return self.next_sentence(x),self.mask_lm(x)
