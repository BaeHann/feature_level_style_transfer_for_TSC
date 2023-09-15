from torch.utils.data import DataLoader, Dataset
import torch
from sktime.datasets import load_from_tsfile
import os

#from OS_CNN.OS_CNN_Structure_build import generate_layer_parameter_list

#两个Dataset间的标签一定要有关联，不能各自初始化，那样训练和测试阶段的问题设定不一致，没有意义
class TrainData(Dataset):
    def __init__(self, file_path_begin, file_path_end,temp_dict):#前者指定在是uni还是multi，后者具体到特定数据集
        super(TrainData, self).__init__()
        train_x, train_y = load_from_tsfile(
            os.path.join(file_path_begin, file_path_end), return_data_type="numpy3d"
        )  #二者都是numpy数组，前者是浮点数构成，后者则是字符
        self.len = train_x.shape[0]#信号总个数，用于下方调用
        self.in_channel = train_x.shape[1]#信号通道数
        self.time_length = train_x.shape[-1] #信号长度
        self.train_x = torch.from_numpy(train_x)#三维Tensor
        label = []
        #temp_dict = {}
        class_label = 0
        for i in train_y:
            if temp_dict.__contains__(i):
                label.append(temp_dict.get(i))
            else:
                temp_dict[i] = class_label
                class_label += 1
                label.append(temp_dict.get(i))
        self.num_class = class_label
        self.train_y = torch.tensor(label).long()

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return self.train_x[idx,:,:], self.train_y[idx]

class TestData(Dataset):
    def __init__(self, file_path_begin, file_path_end,temp_dict):  # 前者指定在是uni还是multi，后者具体到特定数据集
        super(TestData, self).__init__()
        test_x, test_y = load_from_tsfile(
            os.path.join(file_path_begin, file_path_end), return_data_type="numpy3d"
        )  # 二者都是numpy数组，前者是浮点数构成，后者则是字符
        self.len = test_x.shape[0]  # 信号总个数，用于下方调用
        self.in_channel = test_x.shape[1]  # 信号通道数
        self.time_length = test_x.shape[-1]  # 信号长度
        self.test_x = torch.from_numpy(test_x)  # 三维Tensor
        label = []
        #temp_dict = {}
        class_label = 0
        for i in test_y:
            if temp_dict.__contains__(i):
                label.append(temp_dict.get(i))
            else:
                print("训练集与测试集出现label不一致的情况，请及时停止训练")
        self.num_class = class_label
        self.test_y = torch.tensor(label).long()

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return self.test_x[idx,:,:], self.test_y[idx]

"""
a = TrainData("Univariate_ts", "InlineSkate/InlineSkate_TRAIN.ts")
print(a.len, a.num_class, a.time_length,a.in_channel)
target_receptive_field_shape = min(int(a.time_length / 4), 89)
target_layer_parameter_list = generate_layer_parameter_list(1,
                                                            target_receptive_field_shape,
                                                            [8 * 128, 5 * 128 * 256 + 2 * 256 * 128],
                                                            a.in_channel)
new_target_input_channels = 0
for final_layer_parameters in target_layer_parameter_list[-1]:
    new_target_input_channels = new_target_input_channels + final_layer_parameters[1]
print(new_target_input_channels)
"""
"""
a = TrainData("Univariate_ts", "InlineSkate/InlineSkate_TRAIN.ts")
b = TrainData("Univariate_ts","Haptics/Haptics_TRAIN.ts")
train_dataloader_a = DataLoader(a, batch_size=20, shuffle=True)
train_dataloader_b = DataLoader(b, batch_size=20, shuffle=True)
for i, (x,y) in enumerate(train_dataloader_a):
    print(x)
    print(x.shape, x.type)
    print(y)
    print(y.shape, y.type)
    print(i)
    print(a.len, a.num_class, a.time_length)
"""


#加了一个main函数防止import时的不必要输出
if __name__== "__main__" :
    #原来有的从这开始
    a = TrainData("Univariate_ts", "Haptics/Haptics_TRAIN.ts",{})
    b = TrainData("Univariate_ts","Phoneme/Phoneme_TEST.ts",{})
    train_dataloader_a = DataLoader(a, batch_size=20, shuffle=False)
    train_dataloader_b = DataLoader(b, batch_size=20, shuffle=False)
    """
    #这个for循环和下面的二选一
    for i, (x,y) in enumerate(train_dataloader_b):
        print(x)
        print(x.shape, x.type)
        print(y)
        print(y.shape, y.type)
        print(i)
        print(b.len, b.num_class, b.time_length)
    """
    source = list(enumerate(train_dataloader_a))
        # print(source)
    target = list(enumerate(train_dataloader_b))
    print(source)
    """
    train_steps = min(len(source), len(target))
    for batch_idx in range(train_steps):
        rounds, (source_data, source_label) = source[batch_idx]
        _, (target_data, _) = target[batch_idx]  # unsupervised learning
        print(source_data,rounds)
        print(source_label)
    """


