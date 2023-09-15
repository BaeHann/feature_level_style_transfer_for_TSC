import os

import numpy as np
import torch.nn as nn
import torch
from sklearn.metrics import accuracy_score
from sktime.datasets import load_from_tsfile
from torch.utils.data import DataLoader
import torch.nn.functional as F
from DataSource import TrainData, TestData
from OS_CNN.OS_CNN import OS_CNN_res, layer_parameter_list_input_change, OS_CNN
from OS_CNN.OS_CNN_Structure_build import generate_layer_parameter_list
from PIL import Image
from scipy.stats import entropy
"""
#判断eval()模式下仍可进行反向求导与模型更新
a = torch.randn(2).cuda()

m1 = nn.Linear(2,3).cuda()
m2 = nn.Linear(3,4).cuda()

m1.eval()
m2.train()
b = m2(m1(a))
sum = torch.sum(b)
sum.backward()
print(b.data.grad)
"""
"""
#学习python条件分支语句
a = 1.5
if a<1:
    b =2
elif a>2:
    b =3
else:
    b =0
print(b)
"""

"""
#指定数据并初始化Tensor
a = torch.ones(2).float().cuda()
print(a)
a = torch.tensor([1,2]).float().cuda()
print(a)
"""

"""
#测试生成的模型
check_point = torch.load("epoch_212.tar")

target_label_dict = {}
#source_label_dict = {}
target_train_dataset = TrainData("Multivariate_ts", "HandMovementDirection/HandMovementDirection_TRAIN.ts",target_label_dict)
target_test_dataset = TestData("Multivariate_ts", "HandMovementDirection/HandMovementDirection_TEST.ts",target_label_dict)
target_original_length = target_train_dataset.time_length
target_original_channel = target_train_dataset.in_channel
target_num_class = target_train_dataset.num_class
paramenter_number_of_layer_list = [8 * 128 * target_original_channel, 5 * 128 * 256 + 2 * 256 * 128]
Max_kernel_size = 89
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
target_classification_module = OS_CNN(new_target_layer_parameter_list,target_num_class)
classification_loss_module = nn.CrossEntropyLoss()
target_feature_extraction_module.load_state_dict(check_point['feature_extraction_state_dict'])
target_classification_module.load_state_dict(check_point['classification_state_dict'])
target_feature_extraction_module = target_feature_extraction_module.cuda()
target_classification_module = target_classification_module.cuda()
classification_loss_module.cuda()
target_feature_extraction_module.eval()
target_classification_module.eval()
target_train_loader = DataLoader(target_train_dataset, batch_size=20, shuffle=False)
target_test_loader = DataLoader(target_test_dataset,batch_size=20,shuffle=False)
predict_list = np.array([])
label_list = np.array([])
softmax_list = []
for i, (x, y) in enumerate(target_test_loader):
        x = x.float().cuda()
        y_ = y.cuda()
        y_predict, _ = target_classification_module(target_feature_extraction_module(x))
        print(classification_loss_module(y_predict,y_).detach().cpu().numpy())
        softmax_list.append(F.softmax(y_predict,dim=1).detach().cpu().numpy())
        y_predict = y_predict.detach().cpu().numpy()
        y_predict = np.argmax(y_predict, axis=1)
        predict_list = np.concatenate((predict_list, y_predict), axis=0)
        label_list = np.concatenate((label_list, y.numpy()), axis=0)
acc = accuracy_score(predict_list, label_list)
#print(predict_list)
#print(softmax_list)
#print(label_list)
print(acc)
np.save("epoch_212.npy",predict_list)
"""

"""
#验证z-normalization
train_x, train_y = load_from_tsfile(
    "Univariate_ts/ACSF1/ACSF1_TRAIN.ts", return_data_type="numpy3d"
        )


mean_of_feature_cols_train = np.nanmean(train_x, axis=2, keepdims= True)
std_of_feature_cols_train = np.nanstd(train_x, axis=2, keepdims= True)
if np.nanmean(abs(mean_of_feature_cols_train)) < 1e-7 and abs(np.nanmean(std_of_feature_cols_train)-1) < 0.05 :
    print("true")
"""

"""
#仅用熵并通过Haptic判断是否能够使用多源域
target_label_dict = {}
target_train_dataset = TrainData("Univariate_ts", "Haptics/Haptics_TRAIN.ts",target_label_dict)
target_test_dataset = TestData("Univariate_ts", "Haptics/Haptics_TEST.ts",target_label_dict)
target_test_loader = DataLoader(target_test_dataset, batch_size=20, shuffle=False)
target_original_length = target_train_dataset.time_length
target_original_channel = target_train_dataset.in_channel
target_num_class = target_train_dataset.num_class

paramenter_number_of_layer_list1 = [8 * 128 * target_original_channel, 5 * 128 * 256 + 2 * 256 * 128]
Max_kernel_size = 89 #set by the author of OS_CNN
#创建target数据集的模块
target_receptive_field_shape = min(int(target_original_length / 4), Max_kernel_size)
target_layer_parameter_list = generate_layer_parameter_list(1,
                                                     target_receptive_field_shape,
                                                     paramenter_number_of_layer_list1,
                                                     target_original_channel)
target_feature_extraction_module1 = OS_CNN_res(target_layer_parameter_list)
target_feature_extraction_module2 = OS_CNN_res(target_layer_parameter_list)
target_feature_extraction_module3 = OS_CNN_res(target_layer_parameter_list)
new_target_input_channels = 0
for final_layer_parameters in target_layer_parameter_list[-1]:
    new_target_input_channels = new_target_input_channels + final_layer_parameters[1]
new_target_layer_parameter_list = layer_parameter_list_input_change(target_layer_parameter_list, \
                                                                    new_target_input_channels)
target_classification_module1 = OS_CNN(new_target_layer_parameter_list,target_num_class)
target_classification_module2 = OS_CNN(new_target_layer_parameter_list,target_num_class)
target_classification_module3 = OS_CNN(new_target_layer_parameter_list,target_num_class)
target_feature_extraction_module1 = target_feature_extraction_module1.cuda()
target_classification_module1 = target_classification_module1.cuda()
target_feature_extraction_module2 = target_feature_extraction_module2.cuda()
target_classification_module2 = target_classification_module2.cuda()
target_feature_extraction_module3 = target_feature_extraction_module3.cuda()
target_classification_module3 = target_classification_module3.cuda()
checkpoint1 = torch.load("epoch_232.tar")
checkpoint2 = torch.load("epoch_282.tar")
checkpoint3 = torch.load("epoch_446.tar")
target_feature_extraction_module1.load_state_dict(checkpoint1['feature_extraction_state_dict'])
target_classification_module1.load_state_dict(checkpoint1['classification_state_dict'])
target_feature_extraction_module1.eval()
target_classification_module1.eval()
target_feature_extraction_module2.load_state_dict(checkpoint2['feature_extraction_state_dict'])
target_classification_module2.load_state_dict(checkpoint2['classification_state_dict'])
target_feature_extraction_module2.eval()
target_classification_module2.eval()
target_feature_extraction_module3.load_state_dict(checkpoint3['feature_extraction_state_dict'])
target_classification_module3.load_state_dict(checkpoint3['classification_state_dict'])
target_feature_extraction_module3.eval()
target_classification_module3.eval()

with torch.no_grad():
    results_of_probs1 = None
    for i, (x, y) in enumerate(target_test_loader):
        x = x.float().cuda()
        y_predict, _ = target_classification_module1(target_feature_extraction_module1(x))
        y_predict = y_predict.detach().cpu().numpy()
        #y_predict = np.argmax(y_predict, axis=1)
        if results_of_probs1 is None:
            results_of_probs1 = y_predict
        else:
            results_of_probs1 = np.concatenate((results_of_probs1, y_predict),axis=0)
        #predict_list = np.concatenate((predict_list, y_predict), axis=0)
        #label_list = np.concatenate((label_list, y.numpy()), axis=0)
    results_of_probs2 = None
    for i, (x, y) in enumerate(target_test_loader):
        x = x.float().cuda()
        y_predict, _ = target_classification_module2(target_feature_extraction_module2(x))
        y_predict = y_predict.detach().cpu().numpy()
        # y_predict = np.argmax(y_predict, axis=1)
        if results_of_probs2 is None:
            results_of_probs2 = y_predict
        else:
            results_of_probs2 = np.concatenate((results_of_probs2, y_predict), axis=0)
        # predict_list = np.concatenate((predict_list, y_predict), axis=0)
        #label_list = np.concatenate((label_list, y.numpy()), axis=0)
    results_of_probs3 = None
    label_list = np.array([])
    for i, (x, y) in enumerate(target_test_loader):
        x = x.float().cuda()
        y_predict, _ = target_classification_module3(target_feature_extraction_module3(x))
        y_predict = y_predict.detach().cpu().numpy()
        # y_predict = np.argmax(y_predict, axis=1)
        if results_of_probs3 is None:
            results_of_probs3 = y_predict
        else:
            results_of_probs3 = np.concatenate((results_of_probs3, y_predict), axis=0)
        # predict_list = np.concatenate((predict_list, y_predict), axis=0)
        label_list = np.concatenate((label_list, y.numpy()), axis=0)
    #熵前面不见得要乘以2，可以调调看
    for i in range(len(results_of_probs1)):
        results_of_probs1[i] = np.exp(results_of_probs1[i]) / np.sum(np.exp(results_of_probs1[i]))
        the_entropy = entropy(results_of_probs1[i])
        results_of_probs1[i] = results_of_probs1[i] * (1+ 2*np.exp(-the_entropy))
    for i in range(len(results_of_probs2)):
        results_of_probs2[i] = np.exp(results_of_probs2[i]) / np.sum(np.exp(results_of_probs2[i]))
        the_entropy = entropy(results_of_probs2[i])
        results_of_probs2[i] = results_of_probs2[i] * (1+ 2*np.exp(-the_entropy))
    for i in range(len(results_of_probs3)):
        results_of_probs3[i] = np.exp(results_of_probs3[i]) / np.sum(np.exp(results_of_probs3[i]))
        the_entropy = entropy(results_of_probs3[i])
        results_of_probs3[i] = results_of_probs3[i] * (1+ 2*np.exp(-the_entropy))

result_final = results_of_probs1 + results_of_probs2 + results_of_probs3
predict_list = np.argmax(result_final, axis=1)
acc = accuracy_score(predict_list, label_list)
np.save("final_predict.npy",predict_list)
#np.save("446_test_true.npy",label_list)
str_out = " accuracy_for_test:" + str(acc)
print(str_out)
"""

"""
#结合target域上训练数据的辅助作用来判断多源域的效果
target_label_dict = {}
target_train_dataset = TrainData("Univariate_ts", "Haptics/Haptics_TRAIN.ts",target_label_dict)
target_test_dataset = TestData("Univariate_ts", "Haptics/Haptics_TEST.ts",target_label_dict)
target_train_loader = DataLoader(target_train_dataset, batch_size=20, shuffle=False)
target_test_loader = DataLoader(target_test_dataset, batch_size=20, shuffle=False)
target_original_length = target_train_dataset.time_length
target_original_channel = target_train_dataset.in_channel
target_num_class = target_train_dataset.num_class

paramenter_number_of_layer_list1 = [8 * 128 * target_original_channel, 5 * 128 * 256 + 2 * 256 * 128]
Max_kernel_size = 89 #set by the author of OS_CNN
#创建target数据集的模块
target_receptive_field_shape = min(int(target_original_length / 4), Max_kernel_size)
target_layer_parameter_list = generate_layer_parameter_list(1,
                                                     target_receptive_field_shape,
                                                     paramenter_number_of_layer_list1,
                                                     target_original_channel)
target_feature_extraction_module1 = OS_CNN_res(target_layer_parameter_list)
target_feature_extraction_module2 = OS_CNN_res(target_layer_parameter_list)
target_feature_extraction_module3 = OS_CNN_res(target_layer_parameter_list)
#target_feature_extraction_module4 = OS_CNN_res(target_layer_parameter_list)
#target_feature_extraction_module5 = OS_CNN_res(target_layer_parameter_list)
new_target_input_channels = 0
for final_layer_parameters in target_layer_parameter_list[-1]:
    new_target_input_channels = new_target_input_channels + final_layer_parameters[1]
new_target_layer_parameter_list = layer_parameter_list_input_change(target_layer_parameter_list, \
                                                                    new_target_input_channels)
target_classification_module1 = OS_CNN(new_target_layer_parameter_list,target_num_class)
target_classification_module2 = OS_CNN(new_target_layer_parameter_list,target_num_class)
target_classification_module3 = OS_CNN(new_target_layer_parameter_list,target_num_class)
#target_classification_module4 = OS_CNN(new_target_layer_parameter_list,target_num_class)
#target_classification_module5 = OS_CNN(new_target_layer_parameter_list,target_num_class)
target_feature_extraction_module1 = target_feature_extraction_module1.cuda()
target_classification_module1 = target_classification_module1.cuda()
target_feature_extraction_module2 = target_feature_extraction_module2.cuda()
target_classification_module2 = target_classification_module2.cuda()
target_feature_extraction_module3 = target_feature_extraction_module3.cuda()
target_classification_module3 = target_classification_module3.cuda()

checkpoint1 = torch.load("epoch_22.tar")
checkpoint2 = torch.load("epoch_490.tar")
checkpoint3 = torch.load("epoch_548.tar")
#checkpoint4 = torch.load("epoch_426.tar")
#checkpoint5 = torch.load("epoch_128.tar")
target_feature_extraction_module1.load_state_dict(checkpoint1['feature_extraction_state_dict'])
target_classification_module1.load_state_dict(checkpoint1['classification_state_dict'])
target_feature_extraction_module1.eval()
target_classification_module1.eval()
target_feature_extraction_module2.load_state_dict(checkpoint2['feature_extraction_state_dict'])
target_classification_module2.load_state_dict(checkpoint2['classification_state_dict'])
target_feature_extraction_module2.eval()
target_classification_module2.eval()
target_feature_extraction_module3.load_state_dict(checkpoint3['feature_extraction_state_dict'])
target_classification_module3.load_state_dict(checkpoint3['classification_state_dict'])
target_feature_extraction_module3.eval()
target_classification_module3.eval()

#通过训练数据集上的准确度来决定对
with torch.no_grad():
    label_list_train = np.array([])
    results_of_train1 = None
    for i, (x, y) in enumerate(target_train_loader):
        x = x.float().cuda()
        y_predict, _ = target_classification_module1(target_feature_extraction_module1(x))
        y_predict = y_predict.detach().cpu().numpy()
        #y_predict = np.argmax(y_predict, axis=1)
        if results_of_train1 is None:
            results_of_train1 = y_predict
        else:
            results_of_train1 = np.concatenate((results_of_train1, y_predict),axis=0)
        label_list_train = np.concatenate((label_list_train, y.numpy()), axis=0)
    results_of_train1 = np.argmax(results_of_train1, axis=1)
    weight_for_1 = []
    for i in range(target_num_class):
        num_for_this_label = 0
        correct_for_this_label = 0
        for n in range(len(label_list_train)):
            if label_list_train[n] == i:
                num_for_this_label += 1
                if label_list_train[n] == results_of_train1[n]:
                    correct_for_this_label += 1
        weight_for_1.append(correct_for_this_label / num_for_this_label)

    results_of_train2 = None
    for i, (x, y) in enumerate(target_train_loader):
        x = x.float().cuda()
        y_predict, _ = target_classification_module2(target_feature_extraction_module2(x))
        y_predict = y_predict.detach().cpu().numpy()
        # y_predict = np.argmax(y_predict, axis=1)
        if results_of_train2 is None:
            results_of_train2 = y_predict
        else:
            results_of_train2 = np.concatenate((results_of_train2, y_predict), axis=0)
        #label_list_train = np.concatenate((label_list_train, y.numpy()), axis=0)
    results_of_train2 = np.argmax(results_of_train2, axis=1)
    weight_for_2 = []
    for i in range(target_num_class):
        num_for_this_label = 0
        correct_for_this_label = 0
        for n in range(len(label_list_train)):
            if label_list_train[n] == i:
                num_for_this_label += 1
                if label_list_train[n] == results_of_train2[n]:
                    correct_for_this_label += 1
        weight_for_2.append(correct_for_this_label / num_for_this_label)

    results_of_train3 = None
    for i, (x, y) in enumerate(target_train_loader):
        x = x.float().cuda()
        y_predict, _ = target_classification_module3(target_feature_extraction_module3(x))
        y_predict = y_predict.detach().cpu().numpy()
        # y_predict = np.argmax(y_predict, axis=1)
        if results_of_train3 is None:
            results_of_train3 = y_predict
        else:
            results_of_train3 = np.concatenate((results_of_train3, y_predict), axis=0)
        #label_list_train = np.concatenate((label_list_train, y.numpy()), axis=0)
    results_of_train3 = np.argmax(results_of_train3, axis=1)
    weight_for_3 = []
    for i in range(target_num_class):
        num_for_this_label = 0
        correct_for_this_label = 0
        for n in range(len(label_list_train)):
            if label_list_train[n] == i:
                num_for_this_label += 1
                if label_list_train[n] == results_of_train3[n]:
                    correct_for_this_label += 1
        weight_for_3.append(correct_for_this_label / num_for_this_label)

weight_1 = np.array(weight_for_1)
weight_2 = np.array(weight_for_2)
weight_3 = np.array(weight_for_3)
#weight_4 = np.array(weight_for_4)
#weight_5 = np.array(weight_for_5)
weight_avg = (weight_1 + weight_2 + weight_3)/3
weight_1 = weight_1/weight_avg
weight_2 = weight_2/weight_avg
weight_3 = weight_3/weight_avg
#weight_4 = weight_4/weight_avg
#weight_5 = weight_5/weight_avg
weight_1 = np.nan_to_num(weight_1)
weight_2 = np.nan_to_num(weight_2)
weight_3 = np.nan_to_num(weight_3)
#weight_4 = np.nan_to_num(weight_4)
#weight_5 = np.nan_to_num(weight_5)
with torch.no_grad():
    results_of_probs1 = None
    for i, (x, y) in enumerate(target_test_loader):
        x = x.float().cuda()
        y_predict, _ = target_classification_module1(target_feature_extraction_module1(x))
        y_predict = y_predict.detach().cpu().numpy()
        #y_predict = np.argmax(y_predict, axis=1)
        if results_of_probs1 is None:
            results_of_probs1 = y_predict
        else:
            results_of_probs1 = np.concatenate((results_of_probs1, y_predict),axis=0)

    results_of_probs2 = None
    for i, (x, y) in enumerate(target_test_loader):
        x = x.float().cuda()
        y_predict, _ = target_classification_module2(target_feature_extraction_module2(x))
        y_predict = y_predict.detach().cpu().numpy()
        # y_predict = np.argmax(y_predict, axis=1)
        if results_of_probs2 is None:
            results_of_probs2 = y_predict
        else:
            results_of_probs2 = np.concatenate((results_of_probs2, y_predict), axis=0)
        # predict_list = np.concatenate((predict_list, y_predict), axis=0)
        #label_list = np.concatenate((label_list, y.numpy()), axis=0)
    results_of_probs3 = None
    label_list = np.array([])
    for i, (x, y) in enumerate(target_test_loader):
        x = x.float().cuda()
        y_predict, _ = target_classification_module3(target_feature_extraction_module3(x))
        y_predict = y_predict.detach().cpu().numpy()
        # y_predict = np.argmax(y_predict, axis=1)
        if results_of_probs3 is None:
            results_of_probs3 = y_predict
        else:
            results_of_probs3 = np.concatenate((results_of_probs3, y_predict), axis=0)
        # predict_list = np.concatenate((predict_list, y_predict), axis=0)
        label_list = np.concatenate((label_list, y.numpy()), axis=0)

#熵前面不见得要乘以2，不同模型的权重也不见得以8为底，可以调调看
for i in range(len(results_of_probs1)):
    results_of_probs1[i] = np.exp(results_of_probs1[i]) / np.sum(np.exp(results_of_probs1[i]))
    the_entropy = entropy(results_of_probs1[i])
    the_label = np.argmax(results_of_probs1[i])
    results_of_probs1[i] = results_of_probs1[i] * (1+ 2*np.exp(-the_entropy)) * np.power(12,weight_1[the_label])
for i in range(len(results_of_probs2)):
    results_of_probs2[i] = np.exp(results_of_probs2[i]) / np.sum(np.exp(results_of_probs2[i]))
    the_entropy = entropy(results_of_probs2[i])
    the_label = np.argmax(results_of_probs2[i])
    results_of_probs2[i] = results_of_probs2[i] * (1+ 2*np.exp(-the_entropy)) * np.power(12,weight_2[the_label])
for i in range(len(results_of_probs3)):
    results_of_probs3[i] = np.exp(results_of_probs3[i]) / np.sum(np.exp(results_of_probs3[i]))
    the_entropy = entropy(results_of_probs3[i])
    the_label = np.argmax(results_of_probs3[i])
    results_of_probs3[i] = results_of_probs3[i] * (1+ 2*np.exp(-the_entropy)) * np.power(12,weight_3[the_label])

result_final = results_of_probs1 + results_of_probs2 + results_of_probs3
predict_list = np.argmax(result_final, axis=1)
acc = accuracy_score(predict_list, label_list)
np.save("final_predict.npy",predict_list)
np.save("true_label.npy", label_list)
#np.save("446_test_true.npy",label_list)
str_out = " accuracy_for_test:" + str(acc)
print(str_out)
"""


#通过可视化输出来看不同源域辅助下的模型在输出上是否有差别
a = np.load("epoch_72.npy")
b = np.load("epoch_212.npy")
f = np.load("epoch_208.npy")

d = np.load("true_label.npy")

#print(a)
#print(b)
#print(accuracy_score(a,b))
#c = np.not_equal(a,b)
#print(c.dtype) #bool类型
#print(d.dtype)#float64类型
c = np.not_equal(a,d)
d1 = []
for i in range(0,len(c)):
    if c[i] == True:
        d1.append(0)
    else:
        d1.append(1)

c = np.not_equal(b,d)
d2 = []
for i in range(0,len(c)):
    if c[i] == True:
        d2.append(0)
    else:
        d2.append(1)
c = np.not_equal(f,d)
d3 = []
for i in range(0,len(c)):
    if c[i] == True:
        d3.append(0)
    else:
        d3.append(1)
colors = [
    (176, 46, 40),  # 红色 #绯红色
    (240, 145, 161)  #蓝色 #桃红色
]
palette = np.array(colors).reshape(-1).tolist()
im = Image.fromarray(np.array(d1).astype(np.uint8).reshape(2,-1))
im = im.convert('P')
im.putpalette(palette)
im.save('72_out.png')
im2 = Image.fromarray(np.array(d2).astype(np.uint8).reshape(2,-1))
im2 = im2.convert('P')
im2.putpalette(palette)
im2.save('212_out.png')
im3 = Image.fromarray(np.array(d3).astype(np.uint8).reshape(2,-1))
im3 = im3.convert('P')
im3.putpalette(palette)
im3.save('208_out.png')
#print(np.not_equal(a,d))
#print(np.not_equal(b,d))
#print(c)


print("````````````````````````")
#print(d)

a = np.load("final_predict.npy")
d = np.load("true_label.npy")
c = np.not_equal(a,d)
d1 = []
for i in range(0,len(c)):
    if c[i] == True:
        d1.append(0)
    else:
        d1.append(1)

colors = [
    (176, 46, 40),  # 红色 #绯红色
    (240, 145, 161)  # 蓝色 #桃红色
]
palette = np.array(colors).reshape(-1).tolist()
im = Image.fromarray(np.array(d1).astype(np.uint8).reshape(2,-1))
im = im.convert('P')
im.putpalette(palette)
im.save('final_out.png')
