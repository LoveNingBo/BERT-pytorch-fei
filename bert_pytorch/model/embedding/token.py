import torch.nn

"""
Python3.x 和 Python2.x 的一个区别是: 
Python 3 可以使用直接使用 super().xxx 代替 super(Class, self).xxx 
原来是等价的啊
"""


class TokenEmbedding(nn.Embedding):
	def __init__(self,vocab_size,embed_size=512):
		super().__init__(vocab_size,embed_size,padding_idx=0)
		#这句话涉及到父类的初始化问题，如果没有用初始化方法初始化父类的值，会导致父类构造函数中的初始值无法继承的问题