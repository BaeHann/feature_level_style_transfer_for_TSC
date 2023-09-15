#应该是不需要像MADAN那样不同adapted source domain彼此融合的方法，因为我们要做也只能在提纯之后的表征上去做这些事，所以直观上看起来可能
#没有图片那么容易想象
import numpy as np

import torch
import torch.nn as nn

def grl_hook(coeff):
    def fun1(grad):
        return -coeff*grad.clone()
    return fun1
def calc_coeff(iter_num, high=1.0, low=0.0, alpha=2.0, max_iter=50.0):
	return np.float(2.0 * (high - low) / (1.0 + np.exp(-alpha * iter_num / max_iter)) - (high - low) + low)
#在source那里通过概率判断来源(对Target转过来的用这个，从Source经过Target再回到Source的除了这个还要去做概率判别)
class FeatureDiscriminatorforSource(nn.Module):
    def __init__(self, length_of_feature):
        super(FeatureDiscriminatorforSource, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(length_of_feature, 800),
            nn.LeakyReLU(0.2),
            nn.Linear(800, 400),
            nn.LeakyReLU(0.2),
            nn.Linear(400, 50),
            nn.LeakyReLU(0.2),
            nn.Linear(50, 1),
        )
        self.iter_num = -1
        self.alpha = 100.0
        self.low = 0.0
        self.high = 1.0
        self.max_iter = 20.0
    def forward(self, probs):
        #得模仿CDAN那样设置一个梯度翻转层并且要设计好coeff
        if self.training:
            self.iter_num += 1
        if self.iter_num >= self.max_iter:
            self.iter_num = self.max_iter
        coeff = calc_coeff(self.iter_num, self.high, self.low, self.alpha, self.max_iter)
        probs = probs * 1.0
        probs.register_hook(grl_hook(coeff))
        value = self.model(probs)
        return value


#试图学习target和source间不同labelprobability可能的对应关系
class ProbTransfer(nn.Module):
    def __init__(self, num_of_channels): #好好读一下OS_CNN的代码，最后拉成一维Tensor，长度是原来Adpativeaveragepool前的通道数
        super(ProbTransfer, self).__init__()
        self.model = nn.LSTM(input_size=num_of_channels, hidden_size=num_of_channels,batch_first=True)

    def forward(self, features_of_target_before_linear):
        features_of_target_before_linear = torch.unsqueeze(features_of_target_before_linear,1)
        input = torch.cat((features_of_target_before_linear,features_of_target_before_linear), dim=1) #人为过两次
        _ , (features_of_source_before_linear, _) = self.model(input)
        return torch.squeeze(features_of_source_before_linear, dim=0) #注意这个输出可能是三维的还要squeeze一下才行


#source domain那里的概率来源用这个来加速训练（target domain那里的CDAN式判别的loss直接写到CDAN函数里了）
def wgan_loss(values_from_target_side, values_from_s2t2s,values_from_source_side):
    W_loss = -torch.mean(values_from_target_side)-torch.mean(values_from_s2t2s) + torch.mean(values_from_source_side)
    return W_loss


#想一下两个不同尺寸的数据集经过特征提取后如何变为同一尺寸的表征
#是S迁就T，并且想一下batch_size那一个维度为何无影响
class DimensionUnification(nn.Module):
    def __init__(self, source_channel, target_channel, source_length, target_length):
        super(DimensionUnification, self).__init__()
        self.length_unification = nn.Linear(in_features=source_length, out_features=target_length)
        self.relu1 = nn.ReLU()
        self.channel_unification = nn.Conv1d(in_channels=source_channel, out_channels=target_channel, kernel_size=1)
        self.relu2 = nn.ReLU()
    def forward(self, source_feature):
        length_transformed = self.length_unification(source_feature)
        length_transformed = self.relu1(length_transformed)
        channel_length_transformed = self.channel_unification(length_transformed)
        channel_length_transformed = self.relu2(channel_length_transformed)
        return channel_length_transformed


#CDAN的判别器,返回wasserstain距离
def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)

#这个应该不用调整了，原来人家都可以分类图片的，到了时序信号这里应该也可以
class AdversarialNetworkforCDAN(nn.Module):
    def __init__(self, in_feature, hidden_size):
        super(AdversarialNetworkforCDAN, self).__init__()
        self.ad_layer1 = nn.Linear(in_feature, hidden_size)
        self.ad_layer2 = nn.Linear(hidden_size, hidden_size)
        self.ad_layer3 = nn.Linear(hidden_size, 1)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.2)
        #self.sigmoid = nn.Sigmoid()
        self.apply(init_weights)
        self.iter_num = -1
        self.alpha = 100.0
        self.low = 0.0
        self.high = 1.0
        self.max_iter = 20.0
        self.coeff = np.float(0.001)
    def forward(self, x):
        # print("inside ad net forward",self.training)
        if self.training:
            self.iter_num += 1
        if self.iter_num >= self.max_iter:
            self.iter_num = self.max_iter
        coeff = calc_coeff(self.iter_num, self.high, self.low, self.alpha, self.max_iter)
        self.coeff = coeff
        x = x * 1.0
        x.register_hook(grl_hook(coeff))
        x = self.ad_layer1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.ad_layer2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        y = self.ad_layer3(x)
        #y = self.sigmoid(y)
        return y


#负责确定NF流inference时负责确定Target和Source的noise融合机制
#NF流的输入和噪声各种尺寸上完全相同且NF不需要length(和batch_size)的大小，只需要指定输入通道数以及一些内部需要操作时用到的通道数即可
class NoiseTransfer(nn.Module):
    def __init__(self, noise_channel, length_of_noise,with_nvidia=True):
        super(NoiseTransfer, self).__init__()
        self.apply_learnable_weight = nn.Conv1d(noise_channel, noise_channel, 1)
        self.activation_selu = nn.SELU()  #向标准正态逼近  #千万不能少了()
        if with_nvidia:
            self.target_avg = torch.zeros([noise_channel,length_of_noise]).float().cuda() #require_grad可能还要再斟酌一下#而且多次运算不会出问题吗？
            self.source_avg = torch.zeros([noise_channel,length_of_noise]).float().cuda()
        else:
            self.target_avg = torch.zeros([noise_channel, length_of_noise])  # require_grad可能还要再斟酌一下#而且多次运算不会出问题吗？
            self.source_avg = torch.zeros([noise_channel, length_of_noise])
        self.time = 0
        self.cal_num_target = 0
        self.cal_num_source = 0
    def forward(self, target_noise_batch, source_noise_batch):#target_avg是没有batch那一维度的
        self.time += 1
        batch_target = target_noise_batch.size(0)
        batch_source = source_noise_batch.size(0)
        if self.time == 1 :
            self.target_avg = self.target_avg + torch.mean(target_noise_batch, dim=0)
            self.source_avg = self.source_avg + torch.mean(source_noise_batch, dim=0)
        else:
            self.target_avg = self.target_avg + (batch_target/self.cal_num_target) * torch.mean(target_noise_batch, dim=0)
            self.source_avg = self.source_avg + (batch_source/self.cal_num_source) * torch.mean(source_noise_batch, dim=0)
        self.cal_num_target += batch_target
        self.cal_num_source += batch_source
        general_distance = self.target_avg - self.source_avg
        learned_general_distance = self.apply_learnable_weight(general_distance)
        learned_general_distance = self.activation_selu(learned_general_distance)
        self.source_avg = self.source_avg.detach()
        self.target_avg = self.target_avg.detach()
        return learned_general_distance + source_noise_batch  #第一次输出的可能就直接是target的了，后续在coeff设计上或许可以尝试抵消这一劣势

#最后的multi-source概率投票模块还没写,
