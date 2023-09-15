#learnt from https://github.com/agrija9/Deep-Unsupervised-Domain-Adaptation/blob/master/CDAN

import torch
import torch.nn as nn
import math
import numpy as np

from widgets import AdversarialNetworkforCDAN


class RandomLayer(nn.Module):
    def __init__(self, input_dim_list=[], output_dim=1024,with_nvidia=True):
        super(RandomLayer, self).__init__()
        self.input_num = len(input_dim_list)
        self.output_dim = output_dim
        self.random_matrix = [torch.randn(input_dim_list[i], output_dim) for i in range(self.input_num)]
        if with_nvidia:
            for i in range(self.input_num):
                self.random_matrix[i] = self.random_matrix[i].float().cuda()
    def forward(self, input_list):
        return_list = [torch.mm(input_list[i], self.random_matrix[i]) for i in range(self.input_num)]
        return_tensor = return_list[0] / math.pow(float(self.output_dim), 1.0/len(return_list))
        for single in return_list[1:]:
            return_tensor = torch.mul(return_tensor, single)
        return return_tensor
"""
    def cuda(self):
        super(RandomLayer, self).cuda()
        self.random_matrix = [val.cuda() for val in self.random_matrix]
"""
#用输出的概率值算熵
def Entropy(input_):  #输入得是softmax后的
    bs = input_.size(0)
    epsilon = 1e-5
    entropy = -input_ * torch.log(input_ + epsilon)
    entropy = torch.sum(entropy, dim=1) #会降维度
    return entropy
#两个算反向梯度的函数
def grl_hook(coeff):
    def fun1(grad):
        return -coeff*grad.clone()
    return fun1
def calc_coeff(iter_num, high=1.0, low=0.0, alpha=100.0, max_iter=50.0):
	return np.float(2.0 * (high - low) / (1.0 + np.exp(-alpha * iter_num / max_iter)) - (high - low) + low)


#def CDAN_loss 需要在其他模型定下来以后再具体地去写
#这是一个计算CDAN下的区分度并且结合了WGAN的函数  #这里的概率都是softmax前的，以后OS_CNN框架统一
def CDAN(input_target, input_g_from_source, prob_target, prob_g_from_source, ad_net:AdversarialNetworkforCDAN, \
         random_layer=None):
    input_target = torch.flatten(input_target,1)
    input_g_from_source = torch.flatten(input_g_from_source,1)
    prob_target = torch.nn.functional.softmax(prob_target,dim=1)
    prob_g_from_source = torch.nn.functional.softmax(prob_g_from_source,dim=1)
    if random_layer is None:
        fusion_target = torch.bmm(prob_target.unsqueeze(2), input_target.unsqueeze(1))
        target_out = ad_net(fusion_target.view(-1, input_target.size(1) * prob_target.size(1)))
        fusion_source = torch.bmm(prob_g_from_source.unsqueeze(2), input_g_from_source.unsqueeze(1))
        g_source_out = ad_net(fusion_source.view(-1, input_g_from_source.size(1) * prob_g_from_source.size(1)))
    else:
        fusion_target = random_layer.forward([input_target, prob_target])
        target_out = ad_net(fusion_target.view(-1, fusion_target.size(1)))
        fusion_source = random_layer.forward([input_g_from_source, prob_g_from_source])
        g_source_out = ad_net(fusion_source.view(-1, fusion_source.size(1)))
    #entropy_target = Entropy(nn.Softmax(dim=1)(prob_target))
    #entropy_g_from_source = Entropy(nn.Softmax(dim=1)(prob_g_from_source))
    entropy_target = Entropy(prob_target)
    entropy_g_from_source = Entropy(prob_g_from_source)
    coeff = ad_net.coeff
    entropy_target.register_hook(grl_hook(coeff))
    entropy_g_from_source.register_hook(grl_hook(coeff))
    weight_target = 1.0+torch.exp(-entropy_target)
    weight_g_from_source = 1.0+torch.exp(-entropy_g_from_source)
    weight_target = weight_target / torch.sum(weight_target).detach().item()
    weight_target.view(-1, 1)
    weight_g_from_source = weight_g_from_source / torch.sum(weight_g_from_source).detach().item()
    weight_g_from_source.view(-1,1)
    #因为使用了以熵算出的权重，所以不用torch.mean，而是用权重各自施加对应的样本后加和即可
    distance_target = torch.sum(weight_target * target_out)
    distance_g_from_source = torch.sum(weight_g_from_source * g_source_out)
    #return distance_g_from_source - distance_target
    return distance_target - distance_g_from_source