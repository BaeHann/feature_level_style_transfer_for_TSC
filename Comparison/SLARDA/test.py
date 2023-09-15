import torch
"""
#验证python中数学运算的类型
a = 450//2
print(a)
"""
"""
#生成discriminator的标签
a = torch.randn(5,3)
a = a.long()
b = a.size(0)
print(b)

a = [1]*6
b = torch.tensor(a)
c = b.long()
print(b)
"""

a = torch.randn(2)
print(a)
b = a.squeeze()
print(b)