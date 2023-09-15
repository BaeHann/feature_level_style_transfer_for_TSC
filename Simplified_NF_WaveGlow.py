import torch
from torch.autograd import Variable
import torch.nn.functional as F

#对waveglow进行了简化，仅3个1*1卷积和WN；并且WN中不再是8层那么复杂，由于在保证通用性的前提下通道数实在难以把控，所以放弃"收割"
#其他一些内部模块在尺寸上（尤其是维度等）均有所缩减以避免模型过大以及过拟合

class Invertible1x1Conv(torch.nn.Module):
    """
    learned from https://github.com/NVIDIA/waveglow/blob/master/glow.py
    """
    def __init__(self, c):
        super(Invertible1x1Conv, self).__init__()
        self.conv = torch.nn.Conv1d(c, c, kernel_size=1, stride=1, padding=0,
                                    bias=False)
        # Sample a random orthonormal matrix to initialize weights
        W = torch.qr(torch.FloatTensor(c, c).normal_())[0]
        # Ensure determinant is 1.0 not -1.0
        if torch.det(W) < 0:
            W[:,0] = -1*W[:,0]
        W = W.view(c, c, 1)
        self.conv.weight.data = W

    def forward(self, z, reverse=False):
        # shape
        batch_size, group_size, n_of_groups = z.size()
        W = self.conv.weight.squeeze()
        if reverse:
            if not hasattr(self, 'W_inverse'):
                # Reverse computation
                W_inverse = W.float().inverse()
                W_inverse = Variable(W_inverse[..., None])
                #if z.type() == 'torch.cuda.HalfTensor':
                   # W_inverse = W_inverse.half()
                self.W_inverse = W_inverse
            z = F.conv1d(z, self.W_inverse, bias=None, stride=1, padding=0)
            return z
        else:
            # Forward computation
            log_det_W = batch_size * n_of_groups * torch.logdet(W)
            z = self.conv(z)
            return z, log_det_W

@torch.jit.script
def fused_add_tanh_sigmoid_multiply(input_a, input_b, n_channels):
    """
      learned from https://github.com/NVIDIA/waveglow/blob/master/glow.py
    """
    n_channels_int = n_channels[0]
    in_act = input_a+input_b
    t_act = torch.tanh(in_act[:, :n_channels_int, :])
    s_act = torch.sigmoid(in_act[:, n_channels_int:, :])
    acts = t_act * s_act
    return acts
class WN(torch.nn.Module):  #这部分写的挺好，不用改几乎
    """
        learned from https://github.com/NVIDIA/waveglow/blob/master/glow.py
    """
    def __init__(self, n_in_channels, n_layers, n_channels,
                 kernel_size):  #kernel_size一般是3
        super(WN, self).__init__()
        #assert(kernel_size % 2 == 1)
        #assert(n_channels % 2 == 0)
        self.n_layers = n_layers  #原文中是8，这里暂定是8
        self.n_channels = n_channels
        self.in_layers = torch.nn.ModuleList()
        self.res_skip_layers = torch.nn.ModuleList()

        start = torch.nn.Conv1d(n_in_channels, n_channels, 1)
        start = torch.nn.utils.weight_norm(start, name='weight')
        self.start = start

        # Initializing last layer to 0 makes the affine coupling layers
        # do nothing at first.  This helps with training stability
        end = torch.nn.Conv1d(n_channels, 2*n_in_channels, 1)
        end.weight.data.zero_()
        end.bias.data.zero_()
        self.end = end

        cond_layer = torch.nn.Conv1d(n_in_channels, 2*n_channels*n_layers, 1)
        self.cond_layer = torch.nn.utils.weight_norm(cond_layer, name='weight')

        for i in range(n_layers):
            dilation = 2 ** i
            padding = int((kernel_size*dilation - dilation)/2)
            in_layer = torch.nn.Conv1d(n_channels, 2*n_channels, kernel_size,
                                       dilation=dilation, padding=padding)
            in_layer = torch.nn.utils.weight_norm(in_layer, name='weight')
            self.in_layers.append(in_layer)


            # last one is not necessary
            if i < n_layers - 1:
                res_skip_channels = 2*n_channels
            else:
                res_skip_channels = n_channels
            res_skip_layer = torch.nn.Conv1d(n_channels, res_skip_channels, 1)
            res_skip_layer = torch.nn.utils.weight_norm(res_skip_layer, name='weight')
            self.res_skip_layers.append(res_skip_layer)

    def forward(self, forward_input):
        audio = forward_input
        audio = self.start(audio)
        output = torch.zeros_like(audio)
        n_channels_tensor = torch.IntTensor([self.n_channels])

        spect = self.cond_layer(forward_input)

        for i in range(self.n_layers):
            spect_offset = i*2*self.n_channels
            acts = fused_add_tanh_sigmoid_multiply(
                self.in_layers[i](audio),
                spect[:,spect_offset:spect_offset+2*self.n_channels,:],
                n_channels_tensor)

            res_skip_acts = self.res_skip_layers[i](acts)
            if i < self.n_layers - 1:
                audio = audio + res_skip_acts[:,:self.n_channels,:]
                output = output + res_skip_acts[:,self.n_channels:,:]
            else:
                output = output + res_skip_acts

        return self.end(output)

class WaveGlow(torch.nn.Module):
    """
        learned from https://github.com/NVIDIA/waveglow/blob/master/glow.py
    """
    def __init__(self, n_flows, n_group, n_channels_for_WN):
        super(WaveGlow, self).__init__()
        assert(n_group % 2 == 0)
        self.n_flows = n_flows #原文中设定的是12，现在是3
        self.n_group = n_group  #原始数据通道数
        #self.n_early_every = n_early_every #原文是4，现在设置为2，中间步骤仅收割1次
        #self.n_early_size = n_early_size  #原文是2，再看吧
        self.WN = torch.nn.ModuleList()
        self.convinv = torch.nn.ModuleList()

        n_half = int(n_group/2)

        # Set up layers with the right sizes based on how many dimensions
        # have been output already
        #n_remaining_channels = n_group   #在不同步骤中收割特定的通道
        for k in range(n_flows):
            self.convinv.append(Invertible1x1Conv(n_group))
            self.WN.append(WN(n_half, 8, n_channels_for_WN, 3))
        #self.n_remaining_channels = n_remaining_channels  # Useful during inference

    def forward(self, forward_input):
        """
        forward_input[0] = mel_spectrogram:  batch x n_mel_channels x frames
        forward_input[1] = audio: batch x time
        """
        audio = forward_input
        #audio = audio.unfold(1, self.n_group, self.n_group).permute(0, 2, 1)
        #output_audio = []
        log_s_list = []
        log_det_W_list = []

        for k in range(self.n_flows):
            #if k % self.n_early_every == 0 and k > 0:
                #output_audio.append(audio[:,:self.n_early_size,:])
                #audio = audio[:,self.n_early_size:,:]

            audio, log_det_W = self.convinv[k](audio)
            log_det_W_list.append(log_det_W)

            n_half = int(audio.size(1)/2)
            audio_0 = audio[:,:n_half,:]
            audio_1 = audio[:,n_half:,:]

            output = self.WN[k](audio_0)
            log_s = output[:, n_half:, :]
            b = output[:, :n_half, :]
            audio_1 = torch.exp(log_s)*audio_1 + b
            log_s_list.append(log_s)

            audio = torch.cat([audio_0, audio_1],1)

        #output_audio.append(audio)
        return audio, log_s_list, log_det_W_list

    def infer(self, audio, sigma=1.0):
        #audio = noise[:,self.n_group-self.n_remaining_channels:self.n_group,:]
        #cur_channels = self.n_remaining_channels
        for k in reversed(range(self.n_flows)):
            n_half = int(audio.size(1)/2)
            audio_0 = audio[:,:n_half,:]
            audio_1 = audio[:,n_half:,:]

            output = self.WN[k](audio_0)

            s = output[:, n_half:, :]
            b = output[:, :n_half, :]
            audio_1 = (audio_1 - b)/torch.exp(s)
            audio = torch.cat([audio_0, audio_1],1)
            audio = self.convinv[k](audio, reverse=True)

            #if k % self.n_early_every == 0 and k > 0:
               # z = noise[:,self.n_group-cur_channels-self.n_early_size:self.n_group-cur_channels,:]
               # cur_channels += self.n_early_size
                #audio = torch.cat((sigma*z, audio),1)
        return audio
"""
    @staticmethod
    def remove_weightnorm(model):
        waveglow = model
        for WN in waveglow.WN:
            WN.start = torch.nn.utils.remove_weight_norm(WN.start)
            WN.in_layers = remove(WN.in_layers)
            WN.cond_layer = torch.nn.utils.remove_weight_norm(WN.cond_layer)
            WN.res_skip_layers = remove(WN.res_skip_layers)
        return waveglow
"""
"""
def remove(conv_list):
    new_conv_list = torch.nn.ModuleList()
    for old_conv in conv_list:
        old_conv = torch.nn.utils.remove_weight_norm(old_conv)
        new_conv_list.append(old_conv)
    return new_conv_list
"""
class WaveGlowLoss(torch.nn.Module):
    """
        learned from https://github.com/NVIDIA/waveglow/blob/master/glow.py
    """
    def __init__(self, sigma=1.0):
        super(WaveGlowLoss, self).__init__()
        self.sigma = sigma
    def forward(self, model_output):
        z, log_s_list, log_det_W_list = model_output
        for i, log_s in enumerate(log_s_list):
            if i == 0:
                log_s_total = torch.sum(log_s)
                log_det_W_total = log_det_W_list[i]
            else:
                log_s_total = log_s_total + torch.sum(log_s)
                log_det_W_total += log_det_W_list[i]

        loss = torch.sum(z*z)/(2*self.sigma*self.sigma) - log_s_total - log_det_W_total
        return loss/(z.size(0)*z.size(1)*z.size(2))