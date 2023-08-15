import torch
import torch.nn.functional as F
import torch.nn as nn
x = torch.tensor([[-0.4136, -1.1973]])
target = torch.tensor([[0]]) # 标签 这里还有一个torch.tensor与torch.Tensor的知识点https://blog.csdn.net/weixin_40607008/article/details/107348254
one_hot = F.one_hot(target).float() # 对标签进行one_hot编码
softmax = torch.exp(x)/torch.sum(torch.exp(x), dim = 1).reshape(-1, 1)
logsoftmax = torch.log(softmax)
nllloss = -torch.sum(one_hot*logsoftmax)/target.shape[0]
print(nllloss)
###下面用torch.nn.function实现一下以验证上述结果的正确性
logsoftmax = F.log_softmax(x, dim = 1)
nllloss = F.nll_loss(logsoftmax, target) # 无需对标签做one_hot编码
print(nllloss)
###最后我们直接用torch.nn.CrossEntropyLoss验证一下以上两种方法的正确性
cross_entropy = F.cross_entropy(x, target)
print(cross_entropy)
