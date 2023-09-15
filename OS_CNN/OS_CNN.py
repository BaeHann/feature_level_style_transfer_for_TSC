# learnt from https://github.com/Wensi-Tang/OS-CNN/blob/master/Classifiers/OS_CNN/OS_CNN.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


def calculate_mask_index(kernel_length_now, largest_kernel_lenght):
    right_zero_mast_length = math.ceil((largest_kernel_lenght - 1) / 2) - math.ceil((kernel_length_now - 1) / 2)
    left_zero_mask_length = largest_kernel_lenght - kernel_length_now - right_zero_mast_length
    return left_zero_mask_length, left_zero_mask_length + kernel_length_now


def creat_mask(number_of_input_channel, number_of_output_channel, kernel_length_now, largest_kernel_lenght):
    ind_left, ind_right = calculate_mask_index(kernel_length_now, largest_kernel_lenght)
    mask = np.ones((number_of_input_channel, number_of_output_channel, largest_kernel_lenght))
    mask[:, :, 0:ind_left] = 0
    mask[:, :, ind_right:] = 0
    return mask


def creak_layer_mask(layer_parameter_list):
    largest_kernel_lenght = layer_parameter_list[-1][-1]
    mask_list = []
    init_weight_list = []
    bias_list = []
    for i in layer_parameter_list:
        conv = torch.nn.Conv1d(in_channels=i[0], out_channels=i[1], kernel_size=i[2])
        ind_l, ind_r = calculate_mask_index(i[2], largest_kernel_lenght)
        big_weight = np.zeros((i[1], i[0], largest_kernel_lenght))
        big_weight[:, :, ind_l:ind_r] = conv.weight.detach().numpy()

        bias_list.append(conv.bias.detach().numpy())
        init_weight_list.append(big_weight)

        mask = creat_mask(i[1], i[0], i[2], largest_kernel_lenght)
        mask_list.append(mask)

    mask = np.concatenate(mask_list, axis=0)
    init_weight = np.concatenate(init_weight_list, axis=0)
    init_bias = np.concatenate(bias_list, axis=0)
    return mask.astype(np.float32), init_weight.astype(np.float32), init_bias.astype(np.float32)


class build_layer_with_layer_parameter(nn.Module):
    def __init__(self, layer_parameters,  relu_or_not_at_last_layer = True,with_nvidia=True):
        super(build_layer_with_layer_parameter, self).__init__()
        self.relu_or_not_at_last_layer = relu_or_not_at_last_layer
        os_mask, init_weight, init_bias = creak_layer_mask(layer_parameters)

        in_channels = os_mask.shape[1]
        out_channels = os_mask.shape[0]
        max_kernel_size = os_mask.shape[-1]
        if with_nvidia:
            self.weight_mask = torch.from_numpy(os_mask).float().cuda()
        else:
            self.weight_mask = torch.from_numpy(os_mask)
        self.padding = nn.ConstantPad1d((int((max_kernel_size - 1) / 2), int(max_kernel_size / 2)), 0)

        self.conv1d = torch.nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=max_kernel_size)
        self.conv1d.weight = nn.Parameter(torch.from_numpy(init_weight), requires_grad=True)
        self.conv1d.bias = nn.Parameter(torch.from_numpy(init_bias), requires_grad=True)

        self.bn = nn.BatchNorm1d(num_features=out_channels)

    def forward(self, X):
        self.conv1d.weight.data = self.conv1d.weight * self.weight_mask
        # self.conv1d.weight.data.mul_(self.weight_mask)
        result_1 = self.padding(X)
        result_2 = self.conv1d(result_1)
        result_3 = self.bn(result_2)
        if self.relu_or_not_at_last_layer:
            result = F.relu(result_3)
            return result
        else:
            return result_3

#这个由于是以提取后的特征作为输入，所以输入尺寸（尤其是通道数）上可能得在看看
class OS_CNN(nn.Module):
    def __init__(self, layer_parameter_list, n_class, few_shot=False):
        super(OS_CNN, self).__init__()
        self.few_shot = few_shot
        self.layer_parameter_list = layer_parameter_list
        self.layer_list = []

        for i in range(len(layer_parameter_list)):
            layer = build_layer_with_layer_parameter(layer_parameter_list[i])
            self.layer_list.append(layer)

        self.net = nn.Sequential(*self.layer_list)

        self.averagepool = nn.AdaptiveAvgPool1d(1)

        out_put_channel_numebr = 0
        for final_layer_parameters in layer_parameter_list[-1]:
            out_put_channel_numebr = out_put_channel_numebr + final_layer_parameters[1]

        self.hidden = nn.Linear(out_put_channel_numebr, n_class)
        self.length_before_classification = out_put_channel_numebr
    def forward(self, X):

        X = self.net(X)

        X = self.averagepool(X)
        X_f = torch.squeeze(X,dim=-1)

        if not self.few_shot:
            X = self.hidden(X_f)
        return X, X_f





#以下是下方特征提取层用的os_cnn_res
class OS_block(nn.Module):
    def __init__(self, layer_parameter_list, relu_or_not_at_last_layer=True):
        super(OS_block, self).__init__()
        self.layer_parameter_list = layer_parameter_list
        self.layer_list = []
        self.relu_or_not_at_last_layer = relu_or_not_at_last_layer

        for i in range(len(layer_parameter_list)):
            if i != len(layer_parameter_list) - 1:
                using_relu = True
            else:
                using_relu = self.relu_or_not_at_last_layer

            layer = build_layer_with_layer_parameter(layer_parameter_list[i], using_relu)
            self.layer_list.append(layer)

        self.net = nn.Sequential(*self.layer_list)

    def forward(self, X):

        X = self.net(X)

        return X

#虽然本文只用含一个OS_block的那种res结构，但是为了保险起见还是直接用原作者的代码了
def layer_parameter_list_input_change(layer_parameter_list, input_channel):
    new_layer_parameter_list = []
    for i, i_th_layer_parameter in enumerate(layer_parameter_list):
        if i == 0:
            first_layer_parameter = []
            for cov_parameter in i_th_layer_parameter:
                first_layer_parameter.append((input_channel, cov_parameter[1], cov_parameter[2]))
            new_layer_parameter_list.append(first_layer_parameter)
        else:
            new_layer_parameter_list.append(i_th_layer_parameter)
    return new_layer_parameter_list


class SampaddingConv1D_BN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(SampaddingConv1D_BN, self).__init__()
        self.padding = nn.ConstantPad1d((int((kernel_size - 1) / 2), int(kernel_size / 2)), 0)
        self.conv1d = torch.nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size)
        self.bn = nn.BatchNorm1d(num_features=out_channels)

    def forward(self, X):
        X = self.padding(X)
        X = self.conv1d(X)
        X = self.bn(X)
        return X


class Res_OS_layer(nn.Module):
    def __init__(self, layer_parameter_list, out_put_channel_numebr):
        super(Res_OS_layer, self).__init__()
        self.layer_parameter_list = layer_parameter_list
        self.net = OS_block(layer_parameter_list, False)
        self.res = SampaddingConv1D_BN(layer_parameter_list[0][0][0], out_put_channel_numebr, 1)

    def forward(self, X):
        temp = self.net(X)
        shot_cut = self.res(X)
        block = F.relu(torch.add(shot_cut, temp))
        return block

#这里删除了原代码中的分类用组件
class OS_CNN_res(nn.Module):
    def __init__(self, layer_parameter_list, n_layers=1):
        super(OS_CNN_res, self).__init__()
        #self.few_shot = few_shot
        self.layer_parameter_list = layer_parameter_list
        self.n_layers = n_layers

        out_put_channel_numebr = 0
        for final_layer_parameters in layer_parameter_list[-1]:
            out_put_channel_numebr = out_put_channel_numebr + final_layer_parameters[1]
        new_layer_parameter_list = layer_parameter_list_input_change(layer_parameter_list, out_put_channel_numebr)

        self.net_1 = Res_OS_layer(layer_parameter_list, out_put_channel_numebr)

        self.net_list = []
        for i in range(self.n_layers - 1):
            temp_layer = Res_OS_layer(new_layer_parameter_list, out_put_channel_numebr)
            self.net_list.append(temp_layer)
        if self.n_layers > 1:
            self.net = nn.Sequential(*self.net_list)

        #self.averagepool = nn.AdaptiveAvgPool1d(1)
        #self.hidden = nn.Linear(out_put_channel_numebr, n_class)

    def forward(self, X):

        temp = self.net_1(X)
        if self.n_layers > 1:
            temp = self.net(temp)
        #X = self.averagepool(temp)
        #X = X.squeeze_(-1)

        #if not self.few_shot:
        #    X = self.hidden(X)
        return temp
    #因为这里用的都是只有一个layer的res_net，所有就直接返回
    def return_last_layer(self):
        return self.net_1.net