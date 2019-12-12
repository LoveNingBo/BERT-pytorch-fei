import torch.nn as nn
import torch
import math

"""
在激活函数领域，大家公式的鄙视链应该是：Elus > Relu > Sigmoid 
这些激活函数都有自身的缺陷， sigmoid容易饱和，Elus与Relu缺乏随机因素
GELUs正是在激活中引入了随机正则的思想，是一种对神经元输入的概率描述，
直观上更符合自然的认识，同时实验效果要比Relus与ELUs都要好。
GELUs其实是 dropout、zoneout、Relus的综合，GELUs对于输入乘以一个0,1组成的mask，
而该mask的生成则是依概率随机的依赖于输入。假设输入为X, mask为m，
则m服从一个伯努利分布(Φ(x)\Phi(x)Φ(x), Φ(x)=P(X&lt;=x),X服从标准正太分布\Phi(x)=P(X&lt;=x),
 X服从标准正太分布Φ(x)=P(X<=x),X服从标准正太分布)，这么选择是因为神经元的输入趋向于正太分布，
 这么设定使得当输入x减小的时候，输入会有一个更高的概率被dropout掉，这样的激活变换就会随机依赖于输入了。

 GELU(x)=0.5x(1+tanh[(2/π)^0.5 (x+0.044715*x^3)])

 **********Bert与Transformer中用的方法不同，Transformer中就是用了一下ReLU*******
"""

class GELU(nn.Module):
	def forward(self,x):      
		return 0.5*x*(1+torch.tanh(math.sqrt(2/math.pi)*(x+0.044715*torch.pow(x,3))))                                                             
