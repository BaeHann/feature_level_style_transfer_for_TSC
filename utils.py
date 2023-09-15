#写把损失情况、模型参数等记录到硬盘中的代码 以及  打印当前训练状态等的代码
import torch
import numpy as np
from sklearn.metrics import accuracy_score
from OS_CNN.OS_CNN import OS_CNN_res, OS_CNN
from widgets import DimensionUnification


def save_target_classification_modules(target_feature_extraction_module:OS_CNN_res, target_classification_module:OS_CNN,\
                                       cur_epoch):
    torch.save({
        'epoch': cur_epoch,
        'feature_extraction_state_dict': target_feature_extraction_module.state_dict(),
        'classification_state_dict': target_classification_module.state_dict(),
    }, "train_log/epoch_"+str(cur_epoch)+".tar")


def save_source_classification_modules(source_feature_extraction_module:OS_CNN_res, source_to_target_feature_trans:DimensionUnification,\
                                       source_classification_module:OS_CNN,cur_epoch):
    torch.save({
        'epoch': cur_epoch,
        'feature_extraction_state_dict': source_feature_extraction_module.state_dict(),
        'source_to_target_feature_trans':source_to_target_feature_trans.state_dict(),
        'classification_state_dict': source_classification_module.state_dict(),
    }, "train_log/epoch_"+str(cur_epoch)+"_source.tar")

def eval_model_testdata(target_feature_extraction_module:OS_CNN_res, target_classification_module:OS_CNN,test_dataloader,\
                        cur_epoch,with_nvidia=True):
    predict_list = np.array([])
    label_list = np.array([])
    for i, (x, y) in enumerate(test_dataloader):
        if with_nvidia:
            x = x.float().cuda()
            y_predict, _ = target_classification_module(target_feature_extraction_module(x))
            y_predict = y_predict.detach().cpu().numpy()
            y_predict = np.argmax(y_predict, axis=1)
            predict_list = np.concatenate((predict_list, y_predict), axis=0)
            label_list = np.concatenate((label_list, y.numpy()), axis=0)
        else:
            y_predict, _ = target_classification_module(target_feature_extraction_module(x))
            y_predict = y_predict.detach().numpy()
            y_predict = np.argmax(y_predict, axis=1)
            predict_list = np.concatenate((predict_list, y_predict), axis=0)
            label_list = np.concatenate((label_list, y.numpy()), axis=0)
    acc = accuracy_score(predict_list, label_list)
    #np.save("numpy_saved_with_accuracy/predicted_and_true_label/epoch_"+str(cur_epoch)+"_test_predict.npy",predict_list)
    #np.save("numpy_saved_with_accuracy/predicted_and_true_label/epoch_" + str(cur_epoch) + "_test_true.npy",label_list)
    str_out = "epoch_num:" + str(cur_epoch) + " accuracy_for_test:" + str(acc)
    with open("numpy_saved_with_accuracy/the_log.txt", "a", encoding='utf-8') as f:
        f.write(str_out + "\n")
        f.close()
    print(str_out)

def eval_model_traindata(target_feature_extraction_module:OS_CNN_res, target_classification_module:OS_CNN,train_dataloader,\
                         cur_epoch,with_nvidia=True):
    predict_list = np.array([])
    label_list = np.array([])
    for i, (x, y) in enumerate(train_dataloader):
        if with_nvidia:
            x = x.float().cuda()
            y_predict, _ = target_classification_module(target_feature_extraction_module(x))
            y_predict = y_predict.detach().cpu().numpy()
            y_predict = np.argmax(y_predict, axis=1)
            predict_list = np.concatenate((predict_list, y_predict), axis=0)
            label_list = np.concatenate((label_list, y.numpy()), axis=0)
        else:
            y_predict, _ = target_classification_module(target_feature_extraction_module(x))
            y_predict = y_predict.detach().numpy()
            y_predict = np.argmax(y_predict, axis=1)
            predict_list = np.concatenate((predict_list, y_predict), axis=0)
            label_list = np.concatenate((label_list, y.numpy()), axis=0)
    acc = accuracy_score(predict_list, label_list)
    #np.save("numpy_saved_with_accuracy/predicted_and_true_label/epoch_" + str(cur_epoch) + "_train_predict.npy",predict_list)
    #np.save("numpy_saved_with_accuracy/predicted_and_true_label/epoch_" + str(cur_epoch) + "_train_true.npy", label_list)
    str_out = "epoch_num:"+str(cur_epoch)+" accuracy_for_train:"+str(acc)
    with open("numpy_saved_with_accuracy/the_log.txt", "a", encoding='utf-8') as f:
        f.write(str_out + "\n")
        f.close()
    print(str_out)

def eval_source_model_traindata(source_feature_extraction_module:OS_CNN_res, source_to_target_feature_trans:DimensionUnification,\
                                source_classification_module:OS_CNN,train_dataloader,cur_epoch,with_nvidia=True):
    predict_list = np.array([])
    label_list = np.array([])
    for i, (x, y) in enumerate(train_dataloader):
        if with_nvidia:
            x = x.float().cuda()
            y_predict, _ = source_classification_module(source_to_target_feature_trans(source_feature_extraction_module(x)))
            y_predict = y_predict.detach().cpu().numpy()
            y_predict = np.argmax(y_predict, axis=1)
            predict_list = np.concatenate((predict_list, y_predict), axis=0)
            label_list = np.concatenate((label_list, y.numpy()), axis=0)
        else:
            y_predict, _ = source_classification_module(source_to_target_feature_trans(source_feature_extraction_module(x)))
            y_predict = y_predict.detach().numpy()
            y_predict = np.argmax(y_predict, axis=1)
            predict_list = np.concatenate((predict_list, y_predict), axis=0)
            label_list = np.concatenate((label_list, y.numpy()), axis=0)
    acc = accuracy_score(predict_list, label_list)
    #np.save("numpy_saved_with_accuracy/source_predicted_and_true_label/epoch_" + str(cur_epoch) + "_train_predict.npy",predict_list)
    #np.save("numpy_saved_with_accuracy/source_predicted_and_true_label/epoch_" + str(cur_epoch) + "_train_true.npy", label_list)
    str_out = "epoch_num:"+str(cur_epoch)+" accuracy_for_source_train:"+str(acc)
    with open("numpy_saved_with_accuracy/the_log.txt", "a", encoding='utf-8') as f:
        f.write(str_out + "\n")
        f.close()
    print(str_out)

def eval_source_model_testdata(source_feature_extraction_module:OS_CNN_res, source_to_target_feature_trans:DimensionUnification,\
                               source_classification_module:OS_CNN,test_dataloader,cur_epoch,with_nvidia=True):
    predict_list = np.array([])
    label_list = np.array([])
    for i, (x, y) in enumerate(test_dataloader):
        if with_nvidia:
            x = x.float().cuda()
            y_predict, _ = source_classification_module(source_to_target_feature_trans(source_feature_extraction_module(x)))
            y_predict = y_predict.detach().cpu().numpy()
            y_predict = np.argmax(y_predict, axis=1)
            predict_list = np.concatenate((predict_list, y_predict), axis=0)
            label_list = np.concatenate((label_list, y.numpy()), axis=0)
        else:
            y_predict, _ = source_classification_module(source_to_target_feature_trans(source_feature_extraction_module(x)))
            y_predict = y_predict.detach().numpy()
            y_predict = np.argmax(y_predict, axis=1)
            predict_list = np.concatenate((predict_list, y_predict), axis=0)
            label_list = np.concatenate((label_list, y.numpy()), axis=0)
    acc = accuracy_score(predict_list, label_list)
    #np.save("numpy_saved_with_accuracy/source_predicted_and_true_label/epoch_"+str(cur_epoch)+"_test_predict.npy",predict_list)
    #np.save("numpy_saved_with_accuracy/source_predicted_and_true_label/epoch_" + str(cur_epoch) + "_test_true.npy",label_list)
    str_out = "epoch_num:" + str(cur_epoch) + " accuracy_for_source_test:" + str(acc)
    with open("numpy_saved_with_accuracy/the_log.txt", "a", encoding='utf-8') as f:
        f.write(str_out + "\n")
        f.close()
    print(str_out)

def eval_target_model_being_pretrained(target_feature_extraction_module:OS_CNN_res, target_classification_module:OS_CNN,\
                                       target_dataloader,cur_epoch,whether_test=False,with_nvidia=True):
    predict_list = np.array([])
    label_list = np.array([])
    for i, (x, y) in enumerate(target_dataloader):
        if with_nvidia:
            x = x.float().cuda()
            y_predict, _ = target_classification_module(target_feature_extraction_module(x))
            y_predict = y_predict.detach().cpu().numpy()
            y_predict = np.argmax(y_predict, axis=1)
            predict_list = np.concatenate((predict_list, y_predict), axis=0)
            label_list = np.concatenate((label_list, y.numpy()), axis=0)
        else:
            y_predict, _ = target_classification_module(target_feature_extraction_module(x))
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

def eval_source_model_being_pretrained(source_feature_extraction_module:OS_CNN_res, source_to_target_feature_trans:DimensionUnification,\
                               source_classification_module:OS_CNN,source_dataloader,cur_epoch,whether_test=False,with_nvidia=True):
    predict_list = np.array([])
    label_list = np.array([])
    for i, (x, y) in enumerate(source_dataloader):
        if with_nvidia:
            x = x.float().cuda()
            y_predict, _ = source_classification_module(source_to_target_feature_trans(source_feature_extraction_module(x)))
            y_predict = y_predict.detach().cpu().numpy()
            y_predict = np.argmax(y_predict, axis=1)
            predict_list = np.concatenate((predict_list, y_predict), axis=0)
            label_list = np.concatenate((label_list, y.numpy()), axis=0)
        else:
            y_predict, _ = source_classification_module(source_to_target_feature_trans(source_feature_extraction_module(x)))
            y_predict = y_predict.detach().numpy()
            y_predict = np.argmax(y_predict, axis=1)
            predict_list = np.concatenate((predict_list, y_predict), axis=0)
            label_list = np.concatenate((label_list, y.numpy()), axis=0)
    acc = accuracy_score(predict_list, label_list)
    if whether_test:
        str_out = "epoch_num:" + str(cur_epoch) + " accuracy_for_source_test:" + str(acc)
    else:
        str_out = "epoch_num:" + str(cur_epoch) + " accuracy_for_source_train:" + str(acc)
    print(str_out)