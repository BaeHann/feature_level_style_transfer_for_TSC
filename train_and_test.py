#写加载和保存模型参数等的代码
#在train的时候可以仿照OS_CNN那里吧训练集和测试集的准确度都算了，到时候方便调用模型。另外，记得保存某些中间层的Tensor并在最后可视化一下看一下
#是否混合充分,最后test一般是负责调保存过的模型或者是用在Multi-source融合里面
import numpy as np
import os
import torch
from torch import nn
from torch.utils.data import DataLoader

from DataSource import TrainData, TestData
from C_DAN import RandomLayer, CDAN
from utils import eval_model_traindata, eval_model_testdata, save_target_classification_modules, \
    eval_source_model_traindata, eval_source_model_testdata, save_source_classification_modules, \
    eval_target_model_being_pretrained, eval_source_model_being_pretrained
from widgets import DimensionUnification, ProbTransfer, NoiseTransfer, AdversarialNetworkforCDAN, \
    FeatureDiscriminatorforSource, wgan_loss
from OS_CNN.OS_CNN_Structure_build import generate_layer_parameter_list
from OS_CNN.OS_CNN import OS_CNN_res, OS_CNN, layer_parameter_list_input_change
from Simplified_NF_WaveGlow import WaveGlow, WaveGlowLoss
from Comparison.SLARDA.train import CPC

def train(target_train_dataset:TrainData,target_test_dataset:TestData,source_train_dataset:TrainData,source_test_dataset:TestData,\
          with_nvidia=False,epoch_num=720):
    with torch.autograd.set_detect_anomaly(True):
        # 获取原始信号的长度、通道数以便后续使用
        target_original_length = target_train_dataset.time_length
        target_original_channel = target_train_dataset.in_channel
        target_num_class = target_train_dataset.num_class
        source_original_length = source_train_dataset.time_length
        source_original_channel = source_train_dataset.in_channel
        source_num_class = source_train_dataset.num_class
        """
        if torch.cuda.is_available():
            with_nvidia = True
        else:
            with_nvidia = False
        """
        paramenter_number_of_layer_list1 = [8 * 128 * target_original_channel, 5 * 128 * 256 + 2 * 256 * 128]
        paramenter_number_of_layer_list2 = [8 * 128 * source_original_channel, 5 * 128 * 256 + 2 * 256 * 128]
        Max_kernel_size = 89 #set by the author of OS_CNN
        #创建target数据集的模块
        target_receptive_field_shape = min(int(target_original_length / 4), Max_kernel_size)
        target_layer_parameter_list = generate_layer_parameter_list(1,
                                                             target_receptive_field_shape,
                                                             paramenter_number_of_layer_list1,
                                                             target_original_channel)
        target_feature_extraction_module = OS_CNN_res(target_layer_parameter_list)
        new_target_input_channels = 0
        for final_layer_parameters in target_layer_parameter_list[-1]:
            new_target_input_channels = new_target_input_channels + final_layer_parameters[1]
        new_target_layer_parameter_list = layer_parameter_list_input_change(target_layer_parameter_list, \
                                                                            new_target_input_channels)
        target_classification_module = OS_CNN(new_target_layer_parameter_list,target_num_class)
        #创建source数据集的模块
        source_receptive_field_shape = min(int(source_original_length / 4), Max_kernel_size)
        source_layer_parameter_list = generate_layer_parameter_list(1,
                                                                    source_receptive_field_shape,
                                                                    paramenter_number_of_layer_list2,
                                                                    source_original_channel)
        source_feature_extraction_module = OS_CNN_res(source_layer_parameter_list)
        new_source_input_channels = 0
        for final_layer_parameters in source_layer_parameter_list[-1]:
            new_source_input_channels = new_source_input_channels + final_layer_parameters[1]
        #source到target的转换模块
        source_to_target_feature_trans = DimensionUnification(new_source_input_channels,new_target_input_channels,\
                                                              source_original_length,target_original_length)
        source_classification_module = OS_CNN(new_target_layer_parameter_list, source_num_class) #感受野不是严格的1搭配L，但是绝对够用了
        #probtrasfer implemented thorugh the transformation of features before Linear
        feature_transfer_between_t_s = ProbTransfer(source_classification_module.length_before_classification)
        #Simplified_NF is coming!
        nf_for_transfer = WaveGlow(3,new_target_input_channels,120) #一般new_target_input_channels在50左右或者要大一些
        noise_confusion_for_nf = NoiseTransfer(new_target_input_channels,target_original_length)
        nf_loss = WaveGlowLoss()
        #modules required by CDAN in the target side
        input_length_for_cdan = 1024
        random_layer_for_cdan = RandomLayer([new_target_input_channels*target_original_length,target_num_class])
        ad_net = AdversarialNetworkforCDAN(input_length_for_cdan,1024)
        #modules required by probability discrimination in the source side
        feature_discriminator_s = FeatureDiscriminatorforSource(source_classification_module.length_before_classification)
        classification_loss_module = nn.CrossEntropyLoss()
        #决定运算位置
        if with_nvidia:
            target_feature_extraction_module = target_feature_extraction_module.cuda()
            target_classification_module = target_classification_module.cuda()
            source_feature_extraction_module = source_feature_extraction_module.cuda()
            source_to_target_feature_trans = source_to_target_feature_trans.cuda()
            source_classification_module = source_classification_module.cuda()
            feature_transfer_between_t_s = feature_transfer_between_t_s.cuda()
            nf_for_transfer = nf_for_transfer.cuda()
            nf_loss = nf_loss.cuda()
            noise_confusion_for_nf = noise_confusion_for_nf.cuda()
            random_layer_for_cdan = random_layer_for_cdan.cuda() #无可学习参数，不用optimizer
            ad_net = ad_net.cuda()
            feature_discriminator_s = feature_discriminator_s.cuda()
            classification_loss_module = classification_loss_module.cuda()
        #optimizer and scehduler
        optimizer_target_feature_extraction = torch.optim.RMSprop(target_feature_extraction_module.parameters(),lr=0.001)
        optimizer_target_classification = torch.optim.RMSprop(target_classification_module.parameters(),lr=0.003)
        optimizer_source_feature_extraction = torch.optim.RMSprop(source_feature_extraction_module.parameters(),lr=0.001)
        optimizer_source_to_target_feature_trans = torch.optim.RMSprop(source_to_target_feature_trans.parameters(),lr=0.001)
        optimizer_source_classification = torch.optim.RMSprop(source_classification_module.parameters(),lr=0.003)
        optimizer_feature_transfer_between_t_s = torch.optim.RMSprop(feature_transfer_between_t_s.parameters(),lr=0.001)
        optimizer_nf_for_transfer = torch.optim.RMSprop(nf_for_transfer.parameters(),lr=0.001)
        optimizer_noise_confusion_for_nf = torch.optim.RMSprop(noise_confusion_for_nf.parameters(),lr=0.005)
        optimizer_ad_net = torch.optim.RMSprop(ad_net.parameters(),lr=0.001)
        optimizer_feature_discriminator_s = torch.optim.RMSprop(feature_discriminator_s.parameters(),lr=0.001)
        optimizer_list = []
        optimizer_list.append(optimizer_target_feature_extraction)
        optimizer_list.append(optimizer_target_classification)
        optimizer_list.append(optimizer_source_feature_extraction)
        optimizer_list.append(optimizer_source_to_target_feature_trans)
        optimizer_list.append(optimizer_source_classification)
        optimizer_list.append(optimizer_feature_transfer_between_t_s)
        optimizer_list.append(optimizer_nf_for_transfer)
        optimizer_list.append(optimizer_noise_confusion_for_nf)
        optimizer_list.append(optimizer_ad_net)
        optimizer_list.append(optimizer_feature_discriminator_s)
        scheduler_target_feature_extraction = torch.optim.lr_scheduler.StepLR(optimizer_target_feature_extraction, step_size=25, gamma=0.8)
        scheduler_target_classification = torch.optim.lr_scheduler.StepLR(optimizer_target_classification, step_size=25, gamma=0.8)
        scheduler_source_feature_extraction = torch.optim.lr_scheduler.StepLR(optimizer_source_feature_extraction, step_size=25, gamma=0.8)
        scheduler_source_to_target_feature_trans = torch.optim.lr_scheduler.StepLR(optimizer_source_to_target_feature_trans, step_size=25, gamma=0.8)
        scheduler_source_classification = torch.optim.lr_scheduler.StepLR(optimizer_source_classification, step_size=25, gamma=0.8)
        scheduler_feature_transfer_between_t_s = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_feature_transfer_between_t_s,\
                                                                                            'min',factor=0.7, min_lr=0.0001)
        scheduler_nf_for_transfer = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_nf_for_transfer,'min',factor=0.7, min_lr=0.0001)
        scheduler_noise_confusion_for_nf = torch.optim.lr_scheduler.StepLR(optimizer_noise_confusion_for_nf, step_size=55,gamma=0.6)
        scheduler_ad_net = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_ad_net,'min',factor=0.7, min_lr=0.0001)
        scheduler_feature_discriminator_s = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_feature_discriminator_s,\
                                                                                       'min',factor=0.7, min_lr=0.0001)

        SL_CPC = CPC(new_target_input_channels, 64, target_original_length // 2)
        SL_CPC = SL_CPC.cuda()
        optimizer_sl_cpc = torch.optim.Adam(SL_CPC.parameters(), lr=0.002)
        scheduler_sl_cpc = torch.optim.lr_scheduler.StepLR(optimizer_sl_cpc, step_size=25, gamma=0.7)
        target_train_loader = DataLoader(target_train_dataset, batch_size=20, shuffle=True)
        source_train_loader = DataLoader(source_train_dataset, batch_size=20, shuffle=True)
        target_test_loader = DataLoader(target_test_dataset, batch_size=20, shuffle=True)
        source_test_loader = DataLoader(source_test_dataset,batch_size=20,shuffle=True)

        #首先各自训练target和source的分类所需modules(训练多少个epoch最好先用test.py大概看一下)
        print("pretrain the target classification modules-----------------------------------------------------------")
        target_epoch_pretrain = 5
        for cur_epoch in range(3):
            target_feature_extraction_module.train()
            target_classification_module.train()
            target_data = list(enumerate(target_train_loader))
            rounds_per_epoch = len(target_data)
            for batch_idx in range(rounds_per_epoch):
                _, (target_train, target_label) = target_data[batch_idx]
                if with_nvidia:
                    target_train = target_train.float().cuda()
                    target_label = target_label.cuda()
                target_feature = target_feature_extraction_module(target_train)
                t_sl_loss = SL_CPC(target_feature)
                target_classification_result, target_before_last_linear = target_classification_module(target_feature)
                target_classification_loss = classification_loss_module(target_classification_result, target_label)
                if with_nvidia:
                    str_out = "Epoch:"+str(cur_epoch)+" batch_num:"+str(batch_idx)+" t_c_loss:"+str(target_classification_loss.data.cpu().numpy())\
                              + " t_sl_loss:"+str(t_sl_loss.data.cpu().numpy())
                else:
                    str_out = "Epoch:" + str(cur_epoch) + " batch_num:" + str(batch_idx) + " t_c_loss:" + str(target_classification_loss.data.numpy())\
                              + " t_sl_loss:"+str(t_sl_loss.data.numpy())
                t_total_Loss = target_classification_loss + t_sl_loss
                print(str_out)
                t_total_Loss.backward()
                optimizer_target_feature_extraction.step()
                optimizer_target_classification.step()
                optimizer_sl_cpc.step()
                optimizer_target_feature_extraction.zero_grad()
                optimizer_target_classification.zero_grad()
                optimizer_sl_cpc.zero_grad()
            scheduler_target_feature_extraction.step()
            scheduler_target_classification.step()
            scheduler_sl_cpc.step()
            target_feature_extraction_module.eval()
            target_classification_module.eval()
            eval_target_model_being_pretrained(target_feature_extraction_module,target_classification_module,target_train_loader,\
                                               cur_epoch)
            eval_target_model_being_pretrained(target_feature_extraction_module, target_classification_module,target_test_loader,\
                                               cur_epoch,True)
        print("pretrain the source classification modules-----------------------------------------------------------")
        source_epoch_pretrain = 70
        for cur_epoch in range(source_epoch_pretrain):
            source_feature_extraction_module.train()
            source_to_target_feature_trans.train()
            source_classification_module.train()
            source_data = list(enumerate(source_train_loader))
            rounds_per_epoch = len(source_data)
            for batch_idx in range(rounds_per_epoch):
                _, (source_train, source_label) = source_data[batch_idx]
                if with_nvidia:
                    source_train = source_train.float().cuda()
                    source_label = source_label.cuda()
                source_feature = source_feature_extraction_module(source_train)
                source_shape_changed_feature = source_to_target_feature_trans(source_feature)
                source_classification_result, source_before_last_linear = source_classification_module(source_shape_changed_feature)
                source_classification_loss = classification_loss_module(source_classification_result, source_label)
                if with_nvidia:
                    str_out = "Epoch:"+str(cur_epoch)+" batch_num:"+str(batch_idx)+" s_c_loss:"+str(source_classification_loss.data.cpu().numpy())
                else:
                    str_out = "Epoch:" + str(cur_epoch) + " batch_num:" + str(batch_idx) + " s_c_loss:" + str(source_classification_loss.data.numpy())

                print(str_out)
                source_classification_loss.backward()
                optimizer_source_feature_extraction.step()
                optimizer_source_to_target_feature_trans.step()
                optimizer_source_classification.step()
                optimizer_source_feature_extraction.zero_grad()
                optimizer_source_to_target_feature_trans.zero_grad()
                optimizer_source_classification.zero_grad()
            scheduler_source_feature_extraction.step()
            scheduler_source_to_target_feature_trans.step()
            scheduler_source_classification.step()
            source_feature_extraction_module.eval()
            source_to_target_feature_trans.eval()
            source_classification_module.eval()
            eval_source_model_being_pretrained(source_feature_extraction_module, source_to_target_feature_trans,source_classification_module,\
                                               source_train_loader, cur_epoch)
            eval_source_model_being_pretrained(source_feature_extraction_module, source_to_target_feature_trans,source_classification_module,\
                                               source_test_loader, cur_epoch, True)
        print("self-supervised learning for target and source dataset-----------------------------------------------")
        for cur_epoch in range(65*target_epoch_pretrain):
            target_feature_extraction_module.train()
            target_classification_module.train()
            source_feature_extraction_module.train()
            source_to_target_feature_trans.train()
            source_classification_module.train()
            SL_CPC.train()
            target_data = list(enumerate(target_train_loader))
            source_data = list(enumerate(source_train_loader))
            rounds_per_epoch = min(len(target_data), len(source_data))
            if cur_epoch % 50 == 0:
                for batch_idx in range(rounds_per_epoch):
                    _, (target_train, target_label) = target_data[batch_idx]
                    _, (source_train, source_label) = source_data[batch_idx]
                    if with_nvidia:
                        target_train = target_train.float().cuda()
                        target_label = target_label.cuda()
                        source_train = source_train.float().cuda()
                        source_label = source_label.cuda()  # 这些label别习惯性地加上.float()
                    target_feature = target_feature_extraction_module(target_train)
                    t_sl_loss = SL_CPC(target_feature)
                    target_classification_result, target_before_last_linear = target_classification_module(
                        target_feature)
                    target_classification_loss = classification_loss_module(target_classification_result, target_label)
                    source_feature = source_feature_extraction_module(source_train)
                    source_shape_changed_feature = source_to_target_feature_trans(source_feature)
                    s_sl_loss = SL_CPC(source_shape_changed_feature)
                    source_classification_result, source_before_last_linear = source_classification_module(
                        source_shape_changed_feature)
                    source_classification_loss = classification_loss_module(source_classification_result, source_label)
                    if with_nvidia:
                        str_out = "Epoch:"+str(cur_epoch)+" batch_num:"+str(batch_idx)+" t_c_loss:"+str(target_classification_loss.data.cpu().numpy())\
                                  + " t_sl_loss:"+str(t_sl_loss.data.cpu().numpy()) + " s_c_loss:"+str(source_classification_loss.data.cpu().numpy())\
                                  + " s_sl_loss:"+str(s_sl_loss.data.cpu().numpy())
                    else:
                        str_out = "Epoch:" + str(cur_epoch) + " batch_num:" + str(batch_idx) + " t_c_loss:" + str(target_classification_loss.data.numpy())\
                                  + " t_sl_loss:"+str(t_sl_loss.data.numpy()) + " s_c_loss:"+str(source_classification_loss.data.numpy())\
                                  + " s_sl_loss:"+str(s_sl_loss.data.numpy())
                    t_total_Loss = t_sl_loss + s_sl_loss + 0.8 * target_classification_loss + 1.2*source_classification_loss
                    print(str_out)
                    t_total_Loss.backward()
                    optimizer_target_feature_extraction.step()
                    optimizer_target_classification.step()
                    optimizer_sl_cpc.step()
                    optimizer_source_feature_extraction.step()
                    optimizer_source_to_target_feature_trans.step()
                    optimizer_source_classification.step()
                    optimizer_source_feature_extraction.zero_grad()
                    optimizer_source_to_target_feature_trans.zero_grad()
                    optimizer_source_classification.zero_grad()
                    optimizer_target_feature_extraction.zero_grad()
                    optimizer_target_classification.zero_grad()
                    optimizer_sl_cpc.zero_grad()
                scheduler_target_feature_extraction.step()
                scheduler_target_classification.step()
                scheduler_sl_cpc.step()
                scheduler_source_feature_extraction.step()
                scheduler_source_to_target_feature_trans.step()
                scheduler_source_classification.step()
                source_feature_extraction_module.eval()
                source_to_target_feature_trans.eval()
                source_classification_module.eval()
                target_feature_extraction_module.eval()
                target_classification_module.eval()
                eval_target_model_being_pretrained(target_feature_extraction_module,target_classification_module,target_train_loader,\
                                                   cur_epoch)
                eval_target_model_being_pretrained(target_feature_extraction_module, target_classification_module,target_test_loader,\
                                                   cur_epoch,True)
                eval_source_model_being_pretrained(source_feature_extraction_module, source_to_target_feature_trans,
                                                   source_classification_module, \
                                                   source_train_loader, cur_epoch)
                eval_source_model_being_pretrained(source_feature_extraction_module, source_to_target_feature_trans,
                                                   source_classification_module, \
                                                   source_test_loader, cur_epoch, True)
            else:
                for batch_idx in range(rounds_per_epoch):
                    _, (target_train, target_label) = target_data[batch_idx]
                    _, (source_train, source_label) = source_data[batch_idx]
                    if with_nvidia:
                        target_train = target_train.float().cuda()
                        target_label = target_label.cuda()
                        source_train = source_train.float().cuda()
                        source_label = source_label.cuda()  # 这些label别习惯性地加上.float()
                    target_feature = target_feature_extraction_module(target_train)
                    t_sl_loss = SL_CPC(target_feature)
                    target_classification_result, target_before_last_linear = target_classification_module(
                        target_feature)
                    target_classification_loss = classification_loss_module(target_classification_result, target_label)
                    source_feature = source_feature_extraction_module(source_train)
                    source_shape_changed_feature = source_to_target_feature_trans(source_feature)
                    s_sl_loss = SL_CPC(source_shape_changed_feature)
                    source_classification_result, source_before_last_linear = source_classification_module(
                        source_shape_changed_feature)
                    source_classification_loss = classification_loss_module(source_classification_result, source_label)
                    if with_nvidia:
                        str_out = "Epoch:" + str(cur_epoch) + " batch_num:" + str(batch_idx) + " t_c_loss:" + str(
                            target_classification_loss.data.cpu().numpy()) \
                                  + " t_sl_loss:" + str(t_sl_loss.data.cpu().numpy()) + " s_c_loss:" + str(
                            source_classification_loss.data.cpu().numpy()) \
                                  + " s_sl_loss:" + str(s_sl_loss.data.cpu().numpy())
                    else:
                        str_out = "Epoch:" + str(cur_epoch) + " batch_num:" + str(batch_idx) + " t_c_loss:" + str(
                            target_classification_loss.data.numpy()) \
                                  + " t_sl_loss:" + str(t_sl_loss.data.numpy()) + " s_c_loss:" + str(
                            source_classification_loss.data.numpy()) \
                                  + " s_sl_loss:" + str(s_sl_loss.data.numpy())
                    t_total_Loss = t_sl_loss + s_sl_loss #+ 1.2*target_classification_loss + 1.2*source_classification_loss
                    print(str_out)
                    t_total_Loss.backward()
                    optimizer_target_feature_extraction.step()
                    #optimizer_target_classification.step()
                    optimizer_sl_cpc.step()
                    optimizer_source_feature_extraction.step()
                    optimizer_source_to_target_feature_trans.step()
                    #optimizer_source_classification.step()
                    optimizer_source_feature_extraction.zero_grad()
                    optimizer_source_to_target_feature_trans.zero_grad()
                    #optimizer_source_classification.zero_grad()
                    optimizer_target_feature_extraction.zero_grad()
                    #optimizer_target_classification.zero_grad()
                    optimizer_sl_cpc.zero_grad()
                scheduler_target_feature_extraction.step()
                #scheduler_target_classification.step()
                scheduler_sl_cpc.step()
                scheduler_source_feature_extraction.step()
                scheduler_source_to_target_feature_trans.step()
                #scheduler_source_classification.step()
                source_feature_extraction_module.eval()
                source_to_target_feature_trans.eval()
                source_classification_module.eval()
                target_feature_extraction_module.eval()
                target_classification_module.eval()
                eval_target_model_being_pretrained(target_feature_extraction_module,target_classification_module,target_train_loader,\
                                                   cur_epoch)
                eval_target_model_being_pretrained(target_feature_extraction_module, target_classification_module,target_test_loader,\
                                                   cur_epoch,True)
                eval_source_model_being_pretrained(source_feature_extraction_module, source_to_target_feature_trans,
                                                   source_classification_module, \
                                                   source_train_loader, cur_epoch)
                eval_source_model_being_pretrained(source_feature_extraction_module, source_to_target_feature_trans,
                                                   source_classification_module, \
                                                   source_test_loader, cur_epoch, True)
        torch.save({
            'feature_extraction_state_dict': target_feature_extraction_module.state_dict(),
            'classification_state_dict': target_classification_module.state_dict(),
        }, "train_log/target_classifier_itself.tar")
        torch.save({
            'feature_extraction_state_dict': source_feature_extraction_module.state_dict(),
            'source_to_target_feature_trans': source_to_target_feature_trans.state_dict(),
            'classification_state_dict': source_classification_module.state_dict(),
        }, "train_log/source_classifier_itself.tar")

        print("pretrain the normalizing flow-relevant modules---------------------------------------------------------")
        pretrain_nf = 600
        for cur_epoch in range(pretrain_nf):
            target_feature_extraction_module.train()
            target_classification_module.train()
            source_feature_extraction_module.train()
            source_to_target_feature_trans.train()
            source_classification_module.train()
            nf_for_transfer.train()
            nf_loss.train()
            target_data = list(enumerate(target_train_loader))
            source_data = list(enumerate(source_train_loader))
            rounds_per_epoch = min(len(target_data), len(source_data))
            if cur_epoch % 75 ==0:
                for batch_idx in range(rounds_per_epoch):
                    _, (target_train, target_label) = target_data[batch_idx]
                    _, (source_train, source_label) = source_data[batch_idx]
                    if with_nvidia:
                        target_train = target_train.float().cuda()
                        target_label = target_label.cuda()
                        source_train = source_train.float().cuda()
                        source_label = source_label.cuda()  # 这些label别习惯性地加上.float()
                    target_feature = target_feature_extraction_module(target_train)
                    t_sl_loss = SL_CPC(target_feature)
                    target_classification_result, target_before_last_linear = target_classification_module(target_feature)
                    target_classification_loss = classification_loss_module(target_classification_result, target_label)
                    source_feature = source_feature_extraction_module(source_train)
                    source_shape_changed_feature = source_to_target_feature_trans(source_feature)
                    s_sl_loss = SL_CPC(source_shape_changed_feature)
                    source_classification_result, source_before_last_linear = source_classification_module(source_shape_changed_feature)
                    source_classification_loss = classification_loss_module(source_classification_result, source_label)
                    target_nf_out = nf_for_transfer(target_feature)
                    source_nf_out = nf_for_transfer(source_shape_changed_feature)
                    target_nf_loss = nf_loss(target_nf_out)
                    source_nf_loss = nf_loss(source_nf_out)
                    if with_nvidia:
                        str_out = "Epoch:" + str(cur_epoch) + " batch_num:" + str(batch_idx) + " t_c_loss:" + \
                                  str(target_classification_loss.data.cpu().numpy())+" t_sl_loss:"+str(t_sl_loss.data.cpu().numpy())+" s_c_loss:" + \
                                  str(source_classification_loss.data.cpu().numpy())+" s_sl_loss:"+str(s_sl_loss.data.cpu().numpy())+\
                                  " t_nf_loss:"+str(target_nf_loss.data.cpu().numpy())+" s_nf_loss:"+str(source_nf_loss.data.cpu().numpy())
                    else:
                        str_out = "Epoch:" + str(cur_epoch) + " batch_num:" + str(batch_idx) + " t_c_loss:" + \
                                  str(target_classification_loss.data.numpy()) + " t_sl_loss:"+str(t_sl_loss.data.numpy())+" s_c_loss:" + \
                                  str(source_classification_loss.data.numpy()) + " s_sl_loss:"+str(s_sl_loss.data.numpy())+\
                                  " t_nf_loss:" + str(target_nf_loss.data.numpy()) + " s_nf_loss:" + str(source_nf_loss.data.numpy())
                    print(str_out)
                    nf_loss_with_classification_loss = target_nf_loss+source_nf_loss+5*target_classification_loss+5*source_classification_loss+3*t_sl_loss+3*s_sl_loss
                    nf_loss_with_classification_loss.backward()
                    optimizer_target_feature_extraction.step()
                    optimizer_target_classification.step()
                    optimizer_source_feature_extraction.step()
                    optimizer_source_to_target_feature_trans.step()
                    optimizer_source_classification.step()
                    optimizer_nf_for_transfer.step()
                    optimizer_sl_cpc.step()
                    optimizer_target_feature_extraction.zero_grad()
                    optimizer_target_classification.zero_grad()
                    optimizer_source_feature_extraction.zero_grad()
                    optimizer_source_to_target_feature_trans.zero_grad()
                    optimizer_source_classification.zero_grad()
                    optimizer_nf_for_transfer.zero_grad()
                    optimizer_sl_cpc.zero_grad()
                scheduler_target_feature_extraction.step()
                scheduler_target_classification.step()
                scheduler_source_feature_extraction.step()
                scheduler_source_to_target_feature_trans.step()
                scheduler_source_classification.step()
                scheduler_sl_cpc.step()
                scheduler_nf_for_transfer.step(nf_loss_with_classification_loss)
                target_feature_extraction_module.eval()
                target_classification_module.eval()
                source_feature_extraction_module.eval()
                source_to_target_feature_trans.eval()
                source_classification_module.eval()
                eval_target_model_being_pretrained(target_feature_extraction_module, target_classification_module,\
                                                   target_train_loader, cur_epoch)
                eval_target_model_being_pretrained(target_feature_extraction_module, target_classification_module,\
                                                   target_test_loader, cur_epoch, True)
                eval_source_model_being_pretrained(source_feature_extraction_module, source_to_target_feature_trans,\
                                                   source_classification_module, source_train_loader, cur_epoch)
                eval_source_model_being_pretrained(source_feature_extraction_module, source_to_target_feature_trans,\
                                                   source_classification_module, source_test_loader, cur_epoch, True)
            else:
                for batch_idx in range(rounds_per_epoch):
                    _, (target_train, target_label) = target_data[batch_idx]
                    _, (source_train, source_label) = source_data[batch_idx]
                    if with_nvidia:
                        target_train = target_train.float().cuda()
                        target_label = target_label.cuda()
                        source_train = source_train.float().cuda()
                        source_label = source_label.cuda()  # 这些label别习惯性地加上.float()
                    target_feature = target_feature_extraction_module(target_train)
                    target_feature = target_feature.detach_()
                    source_feature = source_feature_extraction_module(source_train)
                    source_shape_changed_feature = source_to_target_feature_trans(source_feature)
                    source_shape_changed_feature = source_shape_changed_feature.detach_()
                    target_nf_out = nf_for_transfer(target_feature)
                    source_nf_out = nf_for_transfer(source_shape_changed_feature)
                    target_nf_loss = nf_loss(target_nf_out)
                    source_nf_loss = nf_loss(source_nf_out)
                    if with_nvidia:
                        str_out = "Epoch:" + str(cur_epoch) + " batch_num:" + str(batch_idx) + " t_nf_loss:"+str(target_nf_loss.data.cpu().numpy())+\
                                  " s_nf_loss:"+str(source_nf_loss.data.cpu().numpy())
                    else:
                        str_out = "Epoch:" + str(cur_epoch) + " batch_num:" + str(batch_idx) + " t_nf_loss:" + \
                                  str(target_nf_loss.data.numpy()) + " s_nf_loss:" + str(source_nf_loss.data.numpy())
                    print(str_out)
                    nf_loss_without_c_loss = target_nf_loss+source_nf_loss
                    nf_loss_without_c_loss.backward()
                    optimizer_target_feature_extraction.step()
                    optimizer_source_feature_extraction.step()
                    optimizer_source_to_target_feature_trans.step()
                    optimizer_nf_for_transfer.step()
                    optimizer_target_feature_extraction.zero_grad()
                    optimizer_source_feature_extraction.zero_grad()
                    optimizer_source_to_target_feature_trans.zero_grad()
                    optimizer_nf_for_transfer.zero_grad()
                scheduler_target_feature_extraction.step()
                scheduler_source_feature_extraction.step()
                scheduler_source_to_target_feature_trans.step()
                scheduler_nf_for_transfer.step(nf_loss_without_c_loss)
        print("jointly train in both target side and source side--------------------------------------")


        # 开始结合grad_norm进行整体训练
        #weights that grad_norm requires
        if with_nvidia:  #t和s各自来
            weights_grad_norm_t = nn.Parameter(torch.tensor([2,5]).float().cuda())
            weights_grad_norm_s = nn.Parameter(torch.tensor([2,2,4]).float().cuda())
        else:
            weights_grad_norm_t = nn.Parameter(torch.tensor([2,5]).float())
            weights_grad_norm_s = nn.Parameter(torch.tensor([2,2,4]).float())
        optimizer_weights_t = torch.optim.Adam([weights_grad_norm_t], lr=0.0002)
        optimizer_weights_s = torch.optim.Adam([weights_grad_norm_s], lr=0.001)  #暂时先不用调度器了

        initial_loss_t = None
        initial_loss_s = None
        alpha_for_grad_norm = 3
        #调整成训练状态
        for cur_epoch in range(epoch_num):
            target_feature_extraction_module.train()
            target_classification_module.train()
            source_feature_extraction_module.train()
            source_to_target_feature_trans.train()
            source_classification_module.train()
            feature_transfer_between_t_s.train()
            nf_for_transfer.train()
            nf_loss.train()
            noise_confusion_for_nf.train()
            random_layer_for_cdan.train()
            ad_net.train()
            feature_discriminator_s.train()

            #保存各训练阶段的feature以便后续t-SNE
            np_target_feature = None
            np_source_to_target_feature = None
            np_source_shape_changed_feature = None

            np_source_feature = None
            np_target_to_source_feature = None
            np_s2t2s_feature = None

            target_data = list(enumerate(target_train_loader))
            source_data = list(enumerate(source_train_loader))
            rounds_per_epoch = min(len(target_data), len(source_data))
            for batch_idx in range(rounds_per_epoch):
                _, (target_train,target_label) = target_data[batch_idx]
                _, (source_train,source_label) = source_data[batch_idx]
                if with_nvidia:
                    target_train = target_train.float().cuda()
                    target_label = target_label.cuda()
                    source_train = source_train.float().cuda()
                    source_label = source_label.cuda()  #这些label别习惯性地加上.float()
                target_feature = target_feature_extraction_module(target_train)
                t_sl_loss = SL_CPC(target_feature)
                source_feature = source_feature_extraction_module(source_train)
                source_shape_changed_feature = source_to_target_feature_trans(source_feature)
                s_sl_loss = SL_CPC(source_shape_changed_feature)
                target_nf_out = nf_for_transfer(target_feature)
                source_nf_out = nf_for_transfer(source_shape_changed_feature)
                target_nf_loss = nf_loss(target_nf_out)  #!!!
                #target_nf_loss = 20*target_nf_loss #123123123
                source_nf_loss = nf_loss(source_nf_out)#!!!
                #source_nf_loss = source_nf_loss*20 #123123123
                target_noise, _, _ = target_nf_out  #看一下可行与否
                source_noise, _, _ = source_nf_out
                source_to_target_noise = noise_confusion_for_nf(target_noise,source_noise)
                source_to_target_feature = nf_for_transfer.infer(source_to_target_noise)

                #保存中间feature以便后续可视化
                if np_source_to_target_feature is None:
                    if with_nvidia:
                        np_target_feature = target_feature.data.cpu().numpy()
                        np_source_to_target_feature = source_to_target_feature.data.cpu().numpy()
                        np_source_shape_changed_feature = source_shape_changed_feature.data.cpu().numpy()
                    else:
                        np_target_feature = target_feature.data.numpy()
                        np_source_to_target_feature = source_to_target_feature.data.numpy()
                        np_source_shape_changed_feature = source_shape_changed_feature.data.numpy()
                else:
                    if with_nvidia:
                        np_target_feature = np.concatenate((np_target_feature, target_feature.data.cpu().numpy()), axis=0)
                        np_source_to_target_feature = np.concatenate((np_source_to_target_feature, source_to_target_feature.data.cpu().numpy()), axis=0)
                        np_source_shape_changed_feature = np.concatenate((np_source_shape_changed_feature, source_shape_changed_feature.data.cpu().numpy()),axis=0)
                    else:
                        np_target_feature = np.concatenate((np_target_feature, target_feature.data.numpy()), axis=0)
                        np_source_to_target_feature = np.concatenate((np_source_to_target_feature, source_to_target_feature.data.numpy()), axis=0)
                        np_source_shape_changed_feature = np.concatenate((np_source_shape_changed_feature, source_shape_changed_feature.data.numpy()), axis=0)

                target_classification_result, target_before_last_linear = target_classification_module(target_feature)
                target_classification_module.eval()  #!!!
                source_to_target_classification_result , source_to_target_before_last_linear= target_classification_module(source_to_target_feature)
                target_classification_module.train()
                source_classification_result, source_before_last_linear = source_classification_module(source_shape_changed_feature)
                target_classification_loss = classification_loss_module(target_classification_result,target_label)   #!!!
                #target_classification_loss = 100 * target_classification_loss #123123123
                source_classification_loss = classification_loss_module(source_classification_result,source_label)#不会有影响    #!!!
                #source_classification_loss = 100 * source_classification_loss #123123123
                #涉及特征混合、迁移学习的损失函数
                cdan_loss = CDAN(target_feature,source_to_target_feature,target_classification_result,\
                                 source_to_target_classification_result,ad_net,random_layer_for_cdan)   #!!!
                #cdan_loss = 0.00001*cdan_loss   #123132131232132
                transformed_target_before_last_linear = feature_transfer_between_t_s(target_before_last_linear)
                transformed_source_to_target_before = feature_transfer_between_t_s(source_to_target_before_last_linear)
                classification_result_s2t2s = source_classification_module.hidden(transformed_source_to_target_before)
                s2t2s_classification_loss = classification_loss_module(classification_result_s2t2s,source_label)    #!!!
                #s2t2s_classification_loss = 50 * s2t2s_classification_loss  # 123123123
                feature_discriminator_s_loss = wgan_loss(feature_discriminator_s(transformed_target_before_last_linear),\
                                                       feature_discriminator_s(transformed_source_to_target_before),\
                                                       feature_discriminator_s(source_before_last_linear))    #!!!
                if np_target_to_source_feature is None:
                    if with_nvidia:
                        np_source_feature = source_before_last_linear.data.cpu().numpy()
                        np_target_to_source_feature = transformed_target_before_last_linear.data.cpu().numpy()
                        np_s2t2s_feature = transformed_source_to_target_before.data.cpu().numpy()
                    else:
                        np_source_feature = source_before_last_linear.data.numpy()
                        np_target_to_source_feature = transformed_target_before_last_linear.data.numpy()
                        np_s2t2s_feature = transformed_source_to_target_before.data.numpy()
                else:
                    if with_nvidia:
                        np_source_feature = np.concatenate((np_source_feature, source_before_last_linear.data.cpu().numpy()),axis=0)
                        np_target_to_source_feature = np.concatenate((np_target_to_source_feature,transformed_target_before_last_linear.data.cpu().numpy()),axis=0)
                        np_s2t2s_feature = np.concatenate((np_s2t2s_feature,transformed_source_to_target_before.data.cpu().numpy()),axis=0)
                    else:
                        np_source_feature = np.concatenate((np_source_feature, source_before_last_linear.data.numpy()), axis=0)
                        np_target_to_source_feature = np.concatenate((np_target_to_source_feature, transformed_target_before_last_linear.data.numpy()),axis=0)
                        np_s2t2s_feature = np.concatenate((np_s2t2s_feature, transformed_source_to_target_before.data.numpy()), axis=0)

                #output the loss of the current batch
                if with_nvidia:
                    str_out = "Epoch:"+str(cur_epoch)+" batch_num:"+str(batch_idx)+" t_nf_loss:"+str(target_nf_loss.data.cpu().numpy())\
                              +" s_nf_loss:"+str(source_nf_loss.data.cpu().numpy())+" t_c_loss:"+str(target_classification_loss.data.cpu().numpy())\
                              +" t_sl_loss"+str(t_sl_loss.data.cpu().numpy())+" s_c_loss:"+str(source_classification_loss.data.cpu().numpy())+\
                              " s_sl_loss:"+str(s_sl_loss.data.cpu().numpy())+" cdan_loss:"+str(cdan_loss.data.cpu().numpy())+" s2t2s_c_loss:"+\
                              str(s2t2s_classification_loss.data.cpu().numpy())+" feature_discriminator_s_loss:"+\
                              str(feature_discriminator_s_loss.data.cpu().numpy())+" weight_t"+\
                              str(weights_grad_norm_t.data.cpu().numpy())+" weight_s"+str(weights_grad_norm_s.data.cpu().numpy())
                    #不知道为什么不能在上方用target_nf_loss.data.cpu().numpy()[0]  这个[0]
                else:
                    str_out = "Epoch:" + str(cur_epoch) + " batch_num:" + str(batch_idx) + " t_nf_loss:" + str(target_nf_loss.data.numpy()[0])\
                              + " s_nf_loss:" + str(source_nf_loss.data.numpy()[0]) + " t_c_loss:" + str(target_classification_loss.data.numpy()[0]) \
                              + " t_sl_loss"+str(t_sl_loss.data.numpy())+" s_c_loss:" + str(source_classification_loss.data.numpy()[0]) + \
                              " s_sl_loss:"+str(s_sl_loss.data.numpy())+" cdan_loss:" + str(cdan_loss.data.numpy()[0]) +\
                              " s2t2s_c_loss:" + str(s2t2s_classification_loss.data.numpy()[0]) + " feature_discriminator_s_loss:"\
                              + str(feature_discriminator_s_loss.data.numpy()[0])+" weight_t"+\
                              str(weights_grad_norm_t.data.numpy())+" weight_s"+str(weights_grad_norm_s.data.numpy())
                print(str_out)
                with open("train_log/log.txt", "a", encoding='utf-8') as f:
                    f.write(str_out + "\n")
                    f.close()
                #进行grad_norm辅助下的优化
                loss_t = []
                loss_s = []
                loss_t.append(target_nf_loss)
                #loss_t.append(cdan_loss)   #对抗训练损失函数不定，先不参与grad_norm了
                loss_t.append(target_classification_loss)
                loss_s.append(source_nf_loss)
                loss_s.append(source_classification_loss)
                #loss_s.append(feature_discriminator_s_loss)
                loss_s.append(s2t2s_classification_loss)
                loss_t_stacked = torch.stack(loss_t)
                loss_s_stacked = torch.stack(loss_s)
                if initial_loss_t is None:
                    if with_nvidia:
                        initial_loss_t = 1/(1+np.exp(-loss_t_stacked.data.cpu().numpy()))
                        initial_loss_s = 1/(1+np.exp(-loss_s_stacked.data.cpu().numpy()))
                    else:
                        initial_loss_t = 1/(1+np.exp(-loss_t_stacked.data.numpy()))
                        initial_loss_s = 1/(1+np.exp(-loss_s_stacked.data.numpy()))
                loss_total_without_ad = torch.sum(torch.mul(weights_grad_norm_t,loss_t_stacked))+torch.sum(torch.mul(weights_grad_norm_s,loss_s_stacked))
                if cur_epoch < 12:
                    loss_total = loss_total_without_ad + 3 * cdan_loss + 3 * feature_discriminator_s_loss + 2 * t_sl_loss + 2 * s_sl_loss
                elif cur_epoch < 24:
                    loss_total = loss_total_without_ad + 2 * cdan_loss + 3 * feature_discriminator_s_loss + 1.8 * t_sl_loss + 1.5 * s_sl_loss
                elif cur_epoch < 50:
                    loss_total = loss_total_without_ad + 1.5 * cdan_loss + 2 * feature_discriminator_s_loss + 1.8 * t_sl_loss + 1.8 * s_sl_loss
                else:
                    loss_total = loss_total_without_ad + 1.5 * cdan_loss + 1.5 * feature_discriminator_s_loss + 2.5 * t_sl_loss + 2.5 * s_sl_loss
                for the_optim in optimizer_list:
                    the_optim.zero_grad()
                optimizer_sl_cpc.zero_grad()
                optimizer_weights_s.zero_grad()
                optimizer_weights_t.zero_grad()
                loss_total.backward(retain_graph=True)
                optimizer_weights_s.zero_grad()
                optimizer_weights_t.zero_grad()
                shared_t = target_feature_extraction_module.return_last_layer()
                shared_s = source_feature_extraction_module.return_last_layer()
                norms_t = []
                norms_s = []
                for i in range(len(loss_t_stacked)):
                    grad_this_time = torch.autograd.grad(loss_t_stacked[i],shared_t.parameters(),retain_graph=True)
                    norms_t.append(torch.cat([torch.norm(torch.mul(weights_grad_norm_t[i],g)).unsqueeze(0) for g in grad_this_time]).sum())
                for i in range(len(loss_s_stacked)):
                    grad_this_time = torch.autograd.grad(loss_s_stacked[i],shared_s.parameters(),retain_graph=True)
                    norms_s.append(torch.cat([torch.norm(torch.mul(weights_grad_norm_s[i], g)).unsqueeze(0) for g in grad_this_time]).sum())
                norms_t_stack = torch.stack(norms_t)
                norms_s_stack = torch.stack(norms_s)
                if with_nvidia:
                    loss_ratio_t = (1/(1+np.exp(-loss_t_stacked.data.cpu().numpy()))) / initial_loss_t
                    loss_ratio_s = (1/(1+np.exp(-loss_s_stacked.data.cpu().numpy()))) / initial_loss_s
                else:
                    loss_ratio_t = (1/(1+np.exp(-loss_t_stacked.data.numpy()))) / initial_loss_t
                    loss_ratio_s = (1/(1+np.exp(-loss_s_stacked.data.numpy()))) / initial_loss_s
                inverse_train_rate_t = loss_ratio_t / np.mean(loss_ratio_t)
                inverse_train_rate_s = loss_ratio_s / np.mean(loss_ratio_s)
                if with_nvidia:
                    mean_norm_t = np.mean(norms_t_stack.data.cpu().numpy())
                    mean_norm_s = np.mean(norms_s_stack.data.cpu().numpy())
                else:
                    mean_norm_t = np.mean(norms_t_stack.data.numpy())
                    mean_norm_s = np.mean(norms_s_stack.data.numpy())
                constant_term_t = torch.tensor(mean_norm_t * (inverse_train_rate_t ** alpha_for_grad_norm), requires_grad=False)
                constant_term_s = torch.tensor(mean_norm_s * (inverse_train_rate_s ** alpha_for_grad_norm), requires_grad=False)
                if with_nvidia:
                    constant_term_t = constant_term_t.cuda()
                    constant_term_s = constant_term_s.cuda()
                grad_norm_loss_t = torch.sum(torch.abs(norms_t_stack - constant_term_t))
                grad_norm_loss_s = torch.sum(torch.abs(norms_s_stack - constant_term_s))
                grad_for_weight_t = torch.autograd.grad(grad_norm_loss_t, weights_grad_norm_t)[0]
                grad_for_weight_s = torch.autograd.grad(grad_norm_loss_s, weights_grad_norm_s)[0]
                #optimizer_weights_t.step()
                #optimizer_weights_s.step()
                """
                if cur_epoch==0:  #第一个epoch不确定性太大，尽量那什么稳妥一些
                    if with_nvidia:
                        initial_loss_t = 1/(1+np.exp(-loss_t_stacked.data.cpu().numpy()))
                        initial_loss_s = 1/(1+np.exp(-loss_s_stacked.data.cpu().numpy()))
                    else:
                        initial_loss_t = 1/(1+np.exp(-loss_t_stacked.data.numpy()))
                        initial_loss_s = 1/(1+np.exp(-loss_s_stacked.data.numpy()))
                """
                if with_nvidia:
                    saved_weights_grad_norm_t = weights_grad_norm_t.data.cpu().numpy()
                    saved_weights_grad_norm_s = weights_grad_norm_s.data.cpu().numpy()
                else:
                    saved_weights_grad_norm_t = weights_grad_norm_t.data.numpy()
                    saved_weights_grad_norm_s = weights_grad_norm_s.data.numpy()
                #清空计算图
                loss_total.data = loss_total.data * 0.0
                weights_grad_norm_t.data = weights_grad_norm_t.data * 0.0
                weights_grad_norm_s.data = weights_grad_norm_s.data * 0.0
                loss_t_stacked.data = loss_t_stacked.data * 0.0
                loss_s_stacked.data = loss_s_stacked.data * 0.0
                cdan_loss.data = cdan_loss.data *0.0
                feature_discriminator_s_loss.data = feature_discriminator_s_loss.data *0.0
                loss_total.backward()
                if with_nvidia:
                    weights_grad_norm_t.data = torch.tensor(saved_weights_grad_norm_t).cuda()
                    weights_grad_norm_s.data = torch.tensor(saved_weights_grad_norm_s).cuda()
                else:
                    weights_grad_norm_t.data = torch.tensor(saved_weights_grad_norm_t)
                    weights_grad_norm_s.data = torch.tensor(saved_weights_grad_norm_s)
                weights_grad_norm_t.grad = grad_for_weight_t
                weights_grad_norm_s.grad = grad_for_weight_s   #这么直接.data  .grad换是可以的，而且原来如果是require_grad那么还是
                optimizer_weights_t.step()
                optimizer_weights_s.step()
                for the_optim in optimizer_list:
                    the_optim.step()
                #进行weight_grad_norm的归一化
                optimizer_sl_cpc.step()
                weights_grad_norm_t.data[:].clamp_(min=0.0)
                normalize_coeff_t = 7 / torch.sum(weights_grad_norm_t.data, dim=0)
                weights_grad_norm_t.data = weights_grad_norm_t.data * normalize_coeff_t
                weights_grad_norm_s.data[:].clamp_(min=0.0)
                normalize_coeff_s = 8 / torch.sum(weights_grad_norm_s.data, dim=0)
                weights_grad_norm_s.data = weights_grad_norm_s.data * normalize_coeff_s
                #进行WGAN判别器的参数大小截断
                for p in ad_net.parameters():
                    p.data.clamp_(-0.0005, 0.0005)
                for p in feature_discriminator_s.parameters():
                    p.data.clamp_(-0.01, 0.01)
            scheduler_target_feature_extraction.step()
            scheduler_target_classification.step()
            scheduler_sl_cpc.step()
            scheduler_source_feature_extraction.step()
            scheduler_source_to_target_feature_trans.step()
            scheduler_source_classification.step()
            scheduler_feature_transfer_between_t_s.step(s2t2s_classification_loss) #看一下科学不科学，是否会报错
            scheduler_nf_for_transfer.step(target_nf_loss)
            scheduler_noise_confusion_for_nf.step()
            scheduler_ad_net.step(cdan_loss)
            scheduler_feature_discriminator_s.step(feature_discriminator_s_loss)
            if cur_epoch%2 ==0:
                #保存模型
                save_target_classification_modules(target_feature_extraction_module,target_classification_module,cur_epoch)
                save_source_classification_modules(source_feature_extraction_module,source_to_target_feature_trans,source_classification_module,cur_epoch)
                #一定调成eval模式
                target_feature_extraction_module.eval()
                target_classification_module.eval()
                source_feature_extraction_module.eval()
                source_to_target_feature_trans.eval()
                source_classification_module.eval()
                eval_model_traindata(target_feature_extraction_module,target_classification_module,target_train_loader,cur_epoch,with_nvidia)
                eval_model_testdata(target_feature_extraction_module,target_classification_module,target_test_loader,cur_epoch,with_nvidia)
                eval_source_model_traindata(source_feature_extraction_module,source_to_target_feature_trans,source_classification_module,source_train_loader,cur_epoch,with_nvidia)
                eval_source_model_testdata(source_feature_extraction_module,source_to_target_feature_trans,source_classification_module,source_test_loader,cur_epoch,with_nvidia)
                np.save("numpy_saved_with_accuracy/feature_of_target_s2t/epoch_"+str(cur_epoch)+"target_feature.npy",np_target_feature)
                np.save("numpy_saved_with_accuracy/feature_of_target_s2t/epoch_" + str(cur_epoch) + "s2t_feature.npy",np_source_to_target_feature)
                np.save("numpy_saved_with_accuracy/feature_of_target_s2t/epoch_" + str(cur_epoch) + "source_feature.npy",np_source_shape_changed_feature)
                np.save("numpy_saved_with_accuracy/feature_of_source_t2s/epoch_"+str(cur_epoch)+"source_feature.npy",np_source_feature)
                np.save("numpy_saved_with_accuracy/feature_of_source_t2s/epoch_" + str(cur_epoch) + "target_feature.npy",np_target_to_source_feature)
                np.save("numpy_saved_with_accuracy/feature_of_source_t2s/epoch_" + str(cur_epoch) + "s2t2s_feature.npy",np_s2t2s_feature)
                #保存后的np文件是三维的，第一个维度类似于batch，后两维才是真正的每个样本经提取后的feature