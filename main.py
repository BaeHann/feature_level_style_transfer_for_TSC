
from DataSource import *
from train_and_test import train

target_label_dict = {}
source_label_dict = {}
target_train_dataset = TrainData("Multivariate_ts", "SelfRegulationSCP2/SelfRegulationSCP2_TRAIN.ts",target_label_dict)
target_test_dataset = TestData("Multivariate_ts", "SelfRegulationSCP2/SelfRegulationSCP2_TEST.ts",target_label_dict)
source_train_dataset = TrainData("Univariate_ts","EthanolLevel/EthanolLevel_TRAIN.ts",source_label_dict)
source_test_dataset = TestData("Univariate_ts", "EthanolLevel/EthanolLevel_TEST.ts",source_label_dict)
train(target_train_dataset,target_test_dataset,source_train_dataset,source_test_dataset,True)
#注意，对于PhonemeSpectra这种难分类的样本 ，调高联合训练时的权重为5
"""
weights_grad_norm_t = nn.Parameter(torch.ones(3).float().cuda())
a = weights_grad_norm_t.data.cpu().numpy()
#a = weights_grad_norm_t.cpu().numpy()
print("数组元素数据类型：",a.dtype)cd 
b = torch.tensor(a).cuda()
print(b.dtype)
"""
"""
a = torch.rand(1,requires_grad=True).cuda()
b = a**2
print(b)
print(b.size())
#b = b.data.numpy()[0]
print(str(b.data.cpu().numpy()[0]))
"""
"""
paramenter_number_of_layer_list = [8 * 128, 5 * 128 * 256 + 2 * 256 * 128]
Max_kernel_size = 89 #set by the author of OS_CNN
        #创建target数据集的模块
target_receptive_field_shape = min(int(60 / 4), Max_kernel_size)
target_layer_parameter_list = generate_layer_parameter_list(1,
                                                             target_receptive_field_shape,
                                                             paramenter_number_of_layer_list,
                                                             2)
target_feature_extraction_module = OS_CNN_res(target_layer_parameter_list)
c = torch.nn.Conv1d(3,3,1)
print(c.weight)
"""