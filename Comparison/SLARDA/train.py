from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader

from Comparison.SLARDA.models import Discriminator_ATT
from DataSource import TrainData, TestData
from OS_CNN.OS_CNN_Structure_build import generate_layer_parameter_list
from OS_CNN.OS_CNN import OS_CNN_res, OS_CNN, layer_parameter_list_input_change
import torch
import torch.nn as nn
import numpy as np

from utils import eval_target_model_being_pretrained

def eval_source_model_being_pretrained(target_channel_resize, target_feature_extraction_module:OS_CNN_res, target_classification_module:OS_CNN,\
                                       target_dataloader,cur_epoch,whether_test=False,with_nvidia=True):
    predict_list = np.array([])
    label_list = np.array([])
    for i, (x, y) in enumerate(target_dataloader):
        if with_nvidia:
            x = x.float().cuda()
            y_predict, _ = target_classification_module(target_feature_extraction_module(target_channel_resize(x)))
            y_predict = y_predict.detach().cpu().numpy()
            y_predict = np.argmax(y_predict, axis=1)
            predict_list = np.concatenate((predict_list, y_predict), axis=0)
            label_list = np.concatenate((label_list, y.numpy()), axis=0)
        else:
            y_predict, _ = target_classification_module(target_feature_extraction_module(target_channel_resize(x)))
            y_predict = y_predict.detach().numpy()
            y_predict = np.argmax(y_predict, axis=1)
            predict_list = np.concatenate((predict_list, y_predict), axis=0)
            label_list = np.concatenate((label_list, y.numpy()), axis=0)
    acc = accuracy_score(predict_list, label_list)
    if whether_test==False:
        str_out = "epoch_num:"+str(cur_epoch)+" accuracy_for_train:"+str(acc)
    else:
        str_out = "epoch_num:" + str(cur_epoch) + " accuracy_for_test:" + str(acc)
    print(str_out)
def set_requires_grad(model, requires_grad=True):
    for param in model.parameters():
        param.requires_grad = requires_grad
class CPC(nn.Module):
    def __init__(self, num_channels, gru_hidden_dim, timestep):
        super(CPC, self).__init__()
        self.num_channels = num_channels
        self.hidden_dim = gru_hidden_dim
        self.gru = nn.GRU(num_channels, self.hidden_dim, num_layers=1, bidirectional=False, batch_first=True)
        self.timestep = timestep
        self.Wk = nn.ModuleList([nn.Linear(self.hidden_dim, num_channels) for i in range(self.timestep)])
        self.lsoftmax = nn.LogSoftmax(dim=-1)

    def forward(self, features):
        z = features  # features are (batch_size, #channels, seq_len)
        # seq_len = z.shape[2]
        z = z.transpose(1, 2)


        batch = z.shape[0]
        t_samples = torch.randint(self.timestep//2, size=(1,)).long()  # randomly pick time stamps

        nce = 0  # average over timestep and batch
        encode_samples = torch.empty((self.timestep, batch, self.num_channels)).float()  # e.g. size 12*8*512

        for i in np.arange(1, self.timestep + 1):
            encode_samples[i - 1] = z[:, t_samples + i, :].view(batch, self.num_channels)  # z_tk e.g. size 8*512
        forward_seq = z[:, :t_samples + 1, :]  # e.g. size 8*100*512
        output, _ = self.gru(forward_seq)  # output size e.g. 8*100*256
        c_t = output[:, t_samples, :].view(batch, self.hidden_dim)  # c_t e.g. size 8*256
        pred = torch.empty((self.timestep, batch, self.num_channels)).float()  # e.g. size 12*8*512
        for i in np.arange(0, self.timestep):
            linear = self.Wk[i]
            pred[i] = linear(c_t)  # Wk*c_t e.g. size 8*512
        for i in np.arange(0, self.timestep):
            total = torch.mm(encode_samples[i], torch.transpose(pred[i], 0, 1))  # e.g. size 8*8
            nce += torch.sum(torch.diag(self.lsoftmax(total)))  # nce is a tensor
        nce /= -1. * batch * self.timestep
        return nce

source_pretain_epoch = 70
target_train_epoch = 450
if __name__== "__main__" : #Adapt the original code to the scenarios at which this paper aims
    #没有教师模型
    #准备所需dataloader
    target_label_dict = {}
    source_label_dict = {}
    target_train_dataset = TrainData("../../Multivariate_ts", "SelfRegulationSCP2/SelfRegulationSCP2_TRAIN.ts", target_label_dict)
    target_test_dataset = TestData("../../Multivariate_ts", "SelfRegulationSCP2/SelfRegulationSCP2_TEST.ts", target_label_dict)
    source_train_dataset = TrainData("../../Multivariate_ts", "MotorImagery/MotorImagery_TRAIN.ts", source_label_dict)#对于模型来说最后的种类数不一定一样，所以OS_CNN的linear层参数不能共享
    target_train_loader = DataLoader(target_train_dataset, batch_size=30, shuffle=True)
    source_train_loader = DataLoader(source_train_dataset, batch_size=30, shuffle=True)
    target_test_loader = DataLoader(target_test_dataset, batch_size=30, shuffle=True)
    #开始创建所需模块
    target_original_length = target_train_dataset.time_length
    target_original_channel = target_train_dataset.in_channel
    target_num_class = target_train_dataset.num_class
    source_original_length = source_train_dataset.time_length
    source_original_channel = source_train_dataset.in_channel
    source_num_class = source_train_dataset.num_class
    #assert target_original_channel == source_original_channel, "Sorry, the channel number of these two datasets have to be the same!"
    paramenter_number_of_layer_list1 = [8 * 128 * target_original_channel, 5 * 128 * 256 + 2 * 256 * 128]
    paramenter_number_of_layer_list2 = [8 * 128 * source_original_channel, 5 * 128 * 256 + 2 * 256 * 128]
    Max_kernel_size = 89  # set by the author of OS_CNN
    # 创建source和target数据集的模块，注意这里用不到原论文中所谓的教师模块
    target_receptive_field_shape = min(int(target_original_length / 4), Max_kernel_size)
    source_layer_parameter_list = generate_layer_parameter_list(1,
                                                                target_receptive_field_shape,
                                                                paramenter_number_of_layer_list1,
                                                                target_original_channel)
    source_feature_extraction_module = OS_CNN_res(source_layer_parameter_list)
    target_feature_extraction_module = OS_CNN_res(source_layer_parameter_list)
    new_source_input_channels = 0
    for final_layer_parameters in source_layer_parameter_list[-1]:
        new_source_input_channels = new_source_input_channels + final_layer_parameters[1]
    new_source_layer_parameter_list = layer_parameter_list_input_change(source_layer_parameter_list, \
                                                                        new_source_input_channels)
    source_classification_module = OS_CNN(new_source_layer_parameter_list, source_num_class)
    target_classification_module = OS_CNN(new_source_layer_parameter_list, target_num_class)
    target_feature_extraction_module = target_feature_extraction_module.cuda()
    target_length_trans =  nn.Linear(target_original_length,source_original_length)
    target_length_trans = target_length_trans.cuda()
    target_classification_module = target_classification_module.cuda()
    source_feature_extraction_module = source_feature_extraction_module.cuda()
    source_classification_module = source_classification_module.cuda()
    source_channel_resize = nn.Conv1d(source_original_channel,target_original_channel,1)
    source_channel_resize = source_channel_resize.cuda()
    optimizer_source_channel_resize = torch.optim.Adam(source_channel_resize.parameters(), lr=0.002)
    optimizer_target_feature_extraction = torch.optim.Adam(target_feature_extraction_module.parameters(), lr=0.002)
    optimizer_target_length_trans = torch.optim.Adam(target_length_trans.parameters(), lr=0.002)
    optimizer_target_classification = torch.optim.Adam(target_classification_module.parameters(), lr=0.002)
    optimizer_source_feature_extraction = torch.optim.Adam(source_feature_extraction_module.parameters(), lr=0.002)
    optimizer_source_classification = torch.optim.Adam(source_classification_module.parameters(), lr=0.002)
    scheduler_source_channel_resize = torch.optim.lr_scheduler.StepLR(optimizer_source_channel_resize, step_size=25, gamma=0.5)
    scheduler_target_feature_extraction = torch.optim.lr_scheduler.StepLR(optimizer_target_feature_extraction,
                                                                          step_size=25, gamma=0.5)
    scheduler_target_length_trans = torch.optim.lr_scheduler.StepLR(optimizer_target_length_trans,
                                                                          step_size=25, gamma=0.5)
    scheduler_target_classification = torch.optim.lr_scheduler.StepLR(optimizer_target_classification, step_size=25,
                                                                      gamma=0.5)
    scheduler_source_feature_extraction = torch.optim.lr_scheduler.StepLR(optimizer_source_feature_extraction,
                                                                          step_size=25, gamma=0.5)
    scheduler_source_classification = torch.optim.lr_scheduler.StepLR(optimizer_source_classification, step_size=25,
                                                                      gamma=0.5)
    #source模型的预训练
    criterion = nn.CrossEntropyLoss()
    criterion_disc = nn.BCEWithLogitsLoss()
    SL_CPC = CPC(new_source_input_channels,64,source_original_length//2)
    SL_CPC = SL_CPC.cuda()
    optimizer_sl_cpc = torch.optim.Adam(SL_CPC.parameters(),lr=0.002)
    scheduler_sl_cpc = torch.optim.lr_scheduler.StepLR(optimizer_sl_cpc, step_size=25, gamma=0.5)
    for epoch_num in range(source_pretain_epoch):
        SL_CPC.train()
        source_channel_resize.train()
        source_feature_extraction_module.train()
        source_classification_module.train()
        for batch_idx, (data, target) in enumerate(source_train_loader):
            optimizer_sl_cpc.zero_grad()
            optimizer_source_channel_resize.zero_grad()
            optimizer_source_feature_extraction.zero_grad()
            optimizer_source_classification.zero_grad()
            data = data.float().cuda()
            target = target.cuda()
            feature = source_feature_extraction_module(source_channel_resize(data))
            prediction , _ = source_classification_module(feature)
            sl_loss = SL_CPC(feature)
            classification_loss = criterion(prediction,target)
            loss = 2*sl_loss + classification_loss #可能还得调调权重
            loss.backward()
            print("source: epoch " + str(epoch_num) + " batch " + str(batch_idx)+" " + str(loss.data.cpu().numpy()) + " while "+\
                " classification_loss " + (str(classification_loss.data.cpu().numpy())) + " sl_loss " + (str(sl_loss.data.cpu().numpy())))
            optimizer_source_channel_resize.step()
            optimizer_source_feature_extraction.step()
            optimizer_source_classification.step()
            optimizer_sl_cpc.step()
        scheduler_source_feature_extraction.step()
        scheduler_source_classification.step()
        scheduler_sl_cpc.step()
        scheduler_source_channel_resize.step()

        source_channel_resize.eval()
        source_feature_extraction_module.eval()
        source_classification_module.eval()
        eval_source_model_being_pretrained(source_channel_resize, source_feature_extraction_module, source_classification_module,
                                           source_train_loader, epoch_num)
    torch.save({
        'channel_resize': source_channel_resize.state_dict(),
        'feature_extraction_state_dict': source_feature_extraction_module.state_dict(),
        'classification_state_dict': source_classification_module.state_dict(),
    }, "saved_models/source_pretrain.tar")
    print("------------target----------------------------------------------------")
    checkpoint_source = torch.load("saved_models/source_pretrain.tar")
    target_feature_extraction_module.load_state_dict(checkpoint_source['feature_extraction_state_dict'])
    #选择性更新参数以免报错
    new_model_dict = target_classification_module.state_dict()
    state_dict_tbloaded = {k:v for k,v in checkpoint_source['classification_state_dict'].items() if 'hidden' not in k}
    new_model_dict.update(state_dict_tbloaded)
    target_classification_module.load_state_dict(new_model_dict) #先试一下吧，不见得就能达到预期效果
    set_requires_grad(source_channel_resize, requires_grad=False)
    set_requires_grad(source_feature_extraction_module, requires_grad=False)
    set_requires_grad(source_classification_module, requires_grad=False)
    feature_discriminator = Discriminator_ATT(source_original_length,128,8,8,64).float().cuda()
    optimizer_feature_discriminator = torch.optim.Adam(feature_discriminator.parameters(), lr=0.002)
    for epoch_num in range(target_train_epoch):

        target_feature_extraction_module.train()
        target_classification_module.train()
        target_length_trans.train()
        target_data = list(enumerate(target_train_loader))
        source_data = list(enumerate(source_train_loader))
        rounds_per_epoch = min(len(target_data), len(source_data))
        for batch_idx in range(rounds_per_epoch):
            _, (target_train, target_label) = target_data[batch_idx]
            _, (source_train, source_label) = source_data[batch_idx]
            target_train = target_train.float().cuda()
            target_label = target_label.cuda()
            source_train = source_train.float().cuda()
            source_label = source_label.cuda()


            optimizer_target_feature_extraction.zero_grad()
            optimizer_target_length_trans.zero_grad()
            optimizer_target_classification.zero_grad()
            optimizer_feature_discriminator.zero_grad()
            # train discriminator #
            source_feature = source_feature_extraction_module(source_channel_resize(source_train))
            target_feature = target_feature_extraction_module(target_train)
            target_feature_changed = target_length_trans(target_feature)
            feat_concat = torch.cat((source_feature, target_feature_changed), dim=0)
            pred_concat = feature_discriminator(feat_concat.detach())
            label_src = torch.ones(source_feature.size(0)).cuda()
            label_tgt = torch.zeros(target_feature_changed.size(0)).cuda()
            label_concat = torch.cat((label_src, label_tgt), 0)
            # Discriminator Loss
            loss_disc = criterion_disc(pred_concat.squeeze(), label_concat.float())
            loss_disc.backward()
            # Update disciriminator optimizer
            optimizer_feature_discriminator.step()
            # train target modules#

            optimizer_target_feature_extraction.zero_grad()
            optimizer_target_length_trans.zero_grad()
            optimizer_feature_discriminator.zero_grad()

            pred_tgt = feature_discriminator(target_feature_changed)
            # prepare fake labels
            label_tgt = (torch.ones(target_feature_changed.size(0))).cuda()
            # compute loss for target encoder
            loss_tgt = criterion_disc(pred_tgt.squeeze(), label_tgt.float())
            prediction , _ = target_classification_module(target_feature)
            target_c_loss = criterion(prediction,target_label)
            total_loss = target_c_loss + loss_tgt
            total_loss.backward()
            print("target: epoch " + str(epoch_num) + " batch " + str(batch_idx) + " " + str(total_loss.data.cpu().numpy()) + " while" \
                  +" classification_loss " + (str(target_c_loss.data.cpu().numpy())) + " adaptation_loss " + (str(loss_tgt.data.cpu().numpy())))
            # optimize target encoder

            optimizer_target_feature_extraction.step()
            optimizer_target_length_trans.step()
            optimizer_target_classification.step()
        scheduler_target_length_trans.step()
        scheduler_target_feature_extraction.step()
        scheduler_target_classification.step()
        target_feature_extraction_module.eval()
        target_classification_module.eval()
        eval_target_model_being_pretrained(target_feature_extraction_module, target_classification_module, \
                                           target_train_loader, epoch_num)
        eval_target_model_being_pretrained(target_feature_extraction_module, target_classification_module, \
                                           target_test_loader, epoch_num, True)
        torch.save({
            'epoch': epoch_num,
            'feature_extraction_state_dict': target_feature_extraction_module.state_dict(),
            'classification_state_dict': target_classification_module.state_dict(),
        }, "saved_models/epoch_" + str(epoch_num) + ".tar")
