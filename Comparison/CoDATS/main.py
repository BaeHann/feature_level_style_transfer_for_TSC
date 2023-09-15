import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from Comparison.CoDATS.discriminator import Discriminator_ATT
from DataSource import TrainData, TestData
from OS_CNN.OS_CNN import OS_CNN_res, layer_parameter_list_input_change, OS_CNN
from OS_CNN.OS_CNN_Structure_build import generate_layer_parameter_list
from utils import eval_target_model_being_pretrained

total_epoch = 600
if __name__== "__main__" :
    #在同通道数的数据集间进行仿照论文的DA
    target_label_dict = {}
    source_label_dict_1 = {}
    source_label_dict_2 = {}
    source_label_dict_3 = {}
    target_train_dataset = TrainData("../../Univariate_ts", "Haptics/Haptics_TRAIN.ts", target_label_dict)
    target_test_dataset = TestData("../../Univariate_ts", "Haptics/Haptics_TEST.ts", target_label_dict)
    source_train_dataset_1 = TrainData("../../Univariate_ts", "InlineSkate/InlineSkate_TRAIN.ts",source_label_dict_1)
    source_train_dataset_2 = TrainData("../../Univariate_ts", "Worms/Worms_TRAIN.ts", source_label_dict_2)
    source_train_dataset_3 = TrainData("../../Univariate_ts", "SemgHandMovementCh2/SemgHandMovementCh2_TRAIN.ts", source_label_dict_3)
    target_train_loader = DataLoader(target_train_dataset, batch_size=30, shuffle=True)
    source_train_loader_1 = DataLoader(source_train_dataset_1, batch_size=30, shuffle=True)
    source_train_loader_2 = DataLoader(source_train_dataset_2, batch_size=30, shuffle=True)
    source_train_loader_3 = DataLoader(source_train_dataset_3, batch_size=30, shuffle=True)
    target_test_loader = DataLoader(target_test_dataset, batch_size=30, shuffle=True)
    # 开始创建所需模块
    target_original_length = target_train_dataset.time_length
    target_original_channel = target_train_dataset.in_channel
    target_num_class = target_train_dataset.num_class
    source_original_length_1 = source_train_dataset_1.time_length
    source_original_channel_1 = source_train_dataset_1.in_channel
    source_num_class_1 = source_train_dataset_1.num_class
    source_original_length_2 = source_train_dataset_2.time_length
    source_original_channel_2 = source_train_dataset_2.in_channel
    source_num_class_2 = source_train_dataset_2.num_class
    source_original_length_3 = source_train_dataset_3.time_length
    source_original_channel_3 = source_train_dataset_3.in_channel
    source_num_class_3 = source_train_dataset_3.num_class
    #通道数变换
    source_channel_resize_1 = nn.Conv1d(source_original_channel_1,target_original_channel,1)
    source_channel_resize_2 = nn.Conv1d(source_original_channel_2, target_original_channel, 1)
    source_channel_resize_3 = nn.Conv1d(source_original_channel_3, target_original_channel, 1)
    paramenter_number_of_layer_list = [8 * 128 * target_original_channel, 5 * 128 * 256 + 2 * 256 * 128]
    Max_kernel_size = 89  # set by the author of OS_CNN
    #注意，由于不同数据集间相差各异，所以不能在分类时用一模一样的模型，只能共享一部分参数并在此基础上对齐
    target_receptive_field_shape = min(int(target_original_length / 4), Max_kernel_size)
    target_layer_parameter_list = generate_layer_parameter_list(1,
                                                                target_receptive_field_shape,
                                                                paramenter_number_of_layer_list,
                                                                target_original_channel)
    target_feature_extraction_module = OS_CNN_res(target_layer_parameter_list)
    new_target_input_channels = 0
    for final_layer_parameters in target_layer_parameter_list[-1]:
        new_target_input_channels = new_target_input_channels + final_layer_parameters[1]
    new_target_layer_parameter_list = layer_parameter_list_input_change(target_layer_parameter_list, \
                                                                        new_target_input_channels)
    source_classification_module_1 = OS_CNN(new_target_layer_parameter_list, source_num_class_1)
    source_classification_module_2 = OS_CNN(new_target_layer_parameter_list, source_num_class_2)
    source_classification_module_3 = OS_CNN(new_target_layer_parameter_list, source_num_class_3)
    target_classification_module = OS_CNN(new_target_layer_parameter_list, target_num_class)
    source_trans_1 = nn.Linear(source_original_length_1,target_original_length)
    source_trans_2 = nn.Linear(source_original_length_2, target_original_length)
    source_trans_3 = nn.Linear(source_original_length_3, target_original_length)
    #上cuda
    source_channel_resize_1 = source_channel_resize_1.cuda()
    source_channel_resize_2 = source_channel_resize_2.cuda()
    source_channel_resize_3 = source_channel_resize_3.cuda()
    target_feature_extraction_module = target_feature_extraction_module.cuda()
    source_trans_1 = source_trans_1.cuda()
    source_trans_2 = source_trans_2.cuda()
    source_trans_3 = source_trans_3.cuda()
    source_classification_module_1 = source_classification_module_1.cuda()
    source_classification_module_2 = source_classification_module_2.cuda()
    source_classification_module_3 = source_classification_module_3.cuda()
    target_classification_module = target_classification_module.cuda()
    optimizer_source_c_r_1 = torch.optim.Adam(source_channel_resize_1.parameters(), lr=0.002)
    optimizer_source_c_r_2 = torch.optim.Adam(source_channel_resize_2.parameters(), lr=0.002)
    optimizer_source_c_r_3 = torch.optim.Adam(source_channel_resize_3.parameters(), lr=0.002)
    optimizer_target_feature_extraction = torch.optim.Adam(target_feature_extraction_module.parameters(), lr=0.002)
    optimizer_trans_1 = torch.optim.Adam(source_trans_1.parameters(), lr=0.002)
    optimizer_trans_2 = torch.optim.Adam(source_trans_2.parameters(), lr=0.002)
    optimizer_trans_3 = torch.optim.Adam(source_trans_3.parameters(), lr=0.002)
    optimizer_s_c_1 = torch.optim.Adam(source_classification_module_1.parameters(), lr=0.002)
    optimizer_s_c_2 = torch.optim.Adam(source_classification_module_2.parameters(), lr=0.002)
    optimizer_s_c_3 = torch.optim.Adam(source_classification_module_3.parameters(), lr=0.002)
    optimizer_t_c = torch.optim.Adam(target_classification_module.parameters(), lr=0.002)
    scheduler_source_c_r_1 = torch.optim.lr_scheduler.StepLR(optimizer_source_c_r_1, step_size=25, gamma=0.5)
    scheduler_source_c_r_2 = torch.optim.lr_scheduler.StepLR(optimizer_source_c_r_2, step_size=25, gamma=0.5)
    scheduler_source_c_r_3 = torch.optim.lr_scheduler.StepLR(optimizer_source_c_r_3, step_size=25, gamma=0.5)
    scheduler_target_feature_extraction = torch.optim.lr_scheduler.StepLR(optimizer_target_feature_extraction,
                                                                          step_size=25, gamma=0.5)
    scheduler_trans_1 = torch.optim.lr_scheduler.StepLR(optimizer_trans_1,step_size=25, gamma=0.5)
    scheduler_trans_2 = torch.optim.lr_scheduler.StepLR(optimizer_trans_2, step_size=25,gamma=0.5)
    scheduler_trans_3 = torch.optim.lr_scheduler.StepLR(optimizer_trans_3, step_size=25, gamma=0.5)
    scheduler_s_c_1 = torch.optim.lr_scheduler.StepLR(optimizer_s_c_1, step_size=25, gamma=0.5)
    scheduler_s_c_2 = torch.optim.lr_scheduler.StepLR(optimizer_s_c_2, step_size=25, gamma=0.5)
    scheduler_s_c_3 = torch.optim.lr_scheduler.StepLR(optimizer_s_c_3, step_size=25, gamma=0.5)
    scheduler_t_c = torch.optim.lr_scheduler.StepLR(optimizer_t_c, step_size=25, gamma=0.5)
    feature_discriminator = Discriminator_ATT(target_original_length, 128, 8, 8, 64, 4).float().cuda()
    optimizer_feature_discriminator = torch.optim.Adam(feature_discriminator.parameters(), lr=0.002)
    criterion = nn.CrossEntropyLoss()
    for epoch_num in range(total_epoch):
        source_channel_resize_1.train()
        source_channel_resize_2.train()
        source_channel_resize_3.train()
        target_feature_extraction_module.train()
        source_trans_1.train()
        source_trans_2.train()
        source_trans_3.train()
        target_classification_module.train()
        source_classification_module_1.train()
        source_classification_module_2.train()
        source_classification_module_3.train()
        feature_discriminator.train()
        target_data = list(enumerate(target_train_loader))
        source_data_1 = list(enumerate(source_train_loader_1))
        source_data_2 = list(enumerate(source_train_loader_2))
        source_data_3 = list(enumerate(source_train_loader_3))
        rounds_per_epoch = min(len(target_data), len(source_data_1), len(source_data_2), len(source_data_3))
        for batch_idx in range(rounds_per_epoch):
            _, (target_train, target_label) = target_data[batch_idx]
            _, (source_train_1, source_label_1) = source_data_1[batch_idx]
            _, (source_train_2, source_label_2) = source_data_2[batch_idx]
            _, (source_train_3, source_label_3) = source_data_3[batch_idx]
            target_train = target_train.float().cuda()
            target_label = target_label.cuda()
            source_train_1 = source_train_1.float().cuda()
            source_label_1 = source_label_1.cuda()
            source_train_2 = source_train_2.float().cuda()
            source_label_2 = source_label_2.cuda()
            source_train_3 = source_train_3.float().cuda()
            source_label_3 = source_label_3.cuda()

            target_feature_extraction_module.train()
            optimizer_source_c_r_1.zero_grad()
            optimizer_source_c_r_2.zero_grad()
            optimizer_source_c_r_3.zero_grad()
            optimizer_target_feature_extraction.zero_grad()
            optimizer_trans_1.zero_grad()
            optimizer_trans_2.zero_grad()
            optimizer_trans_3.zero_grad()
            optimizer_s_c_1.zero_grad()
            optimizer_s_c_2.zero_grad()
            optimizer_s_c_3.zero_grad()
            optimizer_t_c.zero_grad()
            optimizer_feature_discriminator.zero_grad()
            label_t = torch.tensor(np.array([0]*target_label.size(0))).long().cuda()
            label_s_1 = torch.tensor(np.array([1] * source_label_1.size(0))).long().cuda()
            label_s_2 = torch.tensor(np.array([2] * source_label_2.size(0))).long().cuda()
            label_s_3 = torch.tensor(np.array([3] * source_label_3.size(0))).long().cuda()

            target_feature = target_feature_extraction_module(target_train)
            #print(target_feature_extraction_module.net_1.res.bn.running_mean.data)
            #print(target_feature_extraction_module.net_1.res.bn.running_var.data)
            target_feature_extraction_module.eval()
            s_1_feature = source_trans_1(target_feature_extraction_module(source_channel_resize_1(source_train_1)))
            #print(target_feature_extraction_module.net_1.res.bn.running_mean.data)
            #print(target_feature_extraction_module.net_1.res.bn.running_var.data)
            s_2_feature = source_trans_2(target_feature_extraction_module(source_channel_resize_2(source_train_2)))
            #print(target_feature_extraction_module.net_1.res.bn.running_mean.data)
            #print(target_feature_extraction_module.net_1.res.bn.running_var.data)
            s_3_feature = source_trans_3(target_feature_extraction_module(source_channel_resize_3(source_train_3)))
            #打印batch_norm的mean与var
            #print(target_feature_extraction_module.net_1.res.bn.running_mean.data)
            #print(target_feature_extraction_module.net_1.res.bn.running_var.data)
            #discriminator loss
            feat_concat = torch.cat((target_feature, s_1_feature, s_2_feature, s_3_feature), dim=0)
            label_concat = torch.cat((label_t, label_s_1, label_s_2, label_s_3), 0)
            pred_concat = feature_discriminator(feat_concat)
            loss_disc  = criterion(pred_concat,label_concat)

            prediction_t , _ = target_classification_module(target_feature)
            prediction_s1 , _ = source_classification_module_1(s_1_feature)
            prediction_s2, _ = source_classification_module_2(s_2_feature)
            prediction_s3, _ = source_classification_module_3(s_3_feature)

            loss_t = criterion(prediction_t, target_label)
            loss_s1 = criterion(prediction_s1, source_label_1)
            loss_s2 = criterion(prediction_s2, source_label_2)
            loss_s3 = criterion(prediction_s3, source_label_3)
            loss = loss_t + loss_s1 + loss_s2 + loss_s3 + loss_disc
            loss.backward()
            print("target: epoch " + str(epoch_num) + " batch " + str(batch_idx) + " " + str(loss.data.cpu().numpy()) +\
                  " while" + " loss_t " + (str(loss_t.data.cpu().numpy())) + " loss_s1 " +\
                  str(loss_s1.data.cpu().numpy())+ " loss_s2 "+str(loss_s2.data.cpu().numpy())+" loss_s3 "+\
                  str(loss_s3.data.cpu().numpy())+" loss_disc "+ str(loss_disc.data.cpu().numpy()))
            optimizer_source_c_r_1.step()
            optimizer_source_c_r_2.step()
            optimizer_source_c_r_3.step()
            optimizer_target_feature_extraction.step()
            optimizer_trans_1.step()
            optimizer_trans_2.step()
            optimizer_trans_3.step()
            optimizer_s_c_1.step()
            optimizer_s_c_2.step()
            optimizer_s_c_3.step()
            optimizer_t_c.step()
            optimizer_feature_discriminator.step()
        scheduler_source_c_r_1.step()
        scheduler_source_c_r_2.step()
        scheduler_source_c_r_3.step()
        scheduler_target_feature_extraction.step()
        scheduler_trans_1.step()
        scheduler_trans_2.step()
        scheduler_trans_3.step()
        scheduler_s_c_1.step()
        scheduler_s_c_2.step()
        scheduler_s_c_3.step()
        scheduler_t_c.step()
        target_feature_extraction_module.eval()
        target_classification_module.eval()
        #先看一下batch_norm的影响
        with torch.no_grad():
            target_data = list(enumerate(target_train_loader))
            rounds_per_epoch = len(target_data)
            total_target_traindata = None
            total_target_trainlabel = None
            for batch_idx in range(rounds_per_epoch):
                _, (target_train, target_label) = target_data[batch_idx]
                target_train = target_train.float().cuda()
                target_label = target_label.cuda()
                if total_target_traindata is None:
                    total_target_traindata = target_train
                    total_target_trainlabel = target_label
                else:
                    total_target_traindata = torch.cat((total_target_traindata,target_train),dim=0)
                    total_target_trainlabel = torch.cat((total_target_trainlabel,target_label),dim=0)
            prediction , _ = target_classification_module(target_feature_extraction_module(total_target_traindata))
            loss_t = criterion(prediction,total_target_trainlabel)
            #print(target_feature_extraction_module.net_1.res.bn.running_mean.data)
            #print(target_feature_extraction_module.net_1.res.bn.running_var.data)
            print("epoch "+str(epoch_num)+": the loss_t when being evaluated is "+str(loss_t.data.cpu().numpy()))

        eval_target_model_being_pretrained(target_feature_extraction_module, target_classification_module, target_train_loader, epoch_num)
        eval_target_model_being_pretrained(target_feature_extraction_module, target_classification_module, target_test_loader, epoch_num, True)
        torch.save({
            'epoch': epoch_num,
            'feature_extraction_state_dict': target_feature_extraction_module.state_dict(),
            'classification_state_dict': target_classification_module.state_dict(),
        }, "saved_models/epoch_" + str(epoch_num) + ".tar")


