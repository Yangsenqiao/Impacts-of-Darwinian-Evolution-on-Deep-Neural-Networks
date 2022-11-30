import torch 
import numpy as np
from sklearn import model_selection
import sys
sys.path.append('/root/neuro_evolution/')
from models import * 
from dataloaders.all_data import process_dataset
from models import lenet1, mlp, lstm, mlp, resenet_imagenet, resnet_cifar
def get_data_loaders(kwargs):
    conf = kwargs["conf"]
    data_dir = kwargs["dir"]
    data_name = conf.Data.dataname
    trainset, testset = process_dataset(data_dir, data_name)
    trainset,valiset = model_selection.train_test_split(trainset, test_size = conf.Data.valisize)
    train_loader = torch.utils.data.DataLoader(dataset = trainset,
                                            batch_size = conf.Data.batch_size,
                                            shuffle = True)

    vali_loader = torch.utils.data.DataLoader(dataset = valiset,
                                            batch_size =  conf.Data.batch_size,
                                            shuffle = False)

    test_loader = torch.utils.data.DataLoader(dataset = testset,
                                            batch_size =  conf.Data.batch_size,
                                            shuffle = False)
    return train_loader, vali_loader, test_loader

def get_models(kwargs):
    conf = kwargs["conf"]
    model_name = kwargs['model_name']
    data_name = conf.Data.dataname
    if model_name == 'LeNet1':
        model = lenet1.LeNet1()
    elif model_name == 'LeNet5':
        model = lenet1.LeNet5()
    elif model_name == 'MLP':
        if data_name == 'CIFAR10':
            model = mlp.MLP(Input_Size = 32, Time_Step = 32, Out_Size = 10)
        elif data_name == 'CIFAR100':
            model = mlp.MLP(Input_Size = 32, Time_Step = 32, Out_Size = 100)
        elif data_name == 'MNIST':
            model = mlp.MLP(Input_Size = 28, Time_Step = 28, Out_Size = 10)  
    elif model_name == 'Lstm':
        if data_name == 'CIFAR10':
            model = lstm.RNN(Input_Size = 32, Time_Step = 32, Out_Size = 10)
        elif data_name == 'CIFAR100':
            model = lstm.RNN(Input_Size = 32, Time_Step = 32, Out_Size = 100)
        elif data_name == 'MNIST':
            model = lstm.RNN(Input_Size = 28, Time_Step = 28, Out_Size = 10)  
    elif model_name == 'Resnet':
        if data_name =='CIFAR100':
            model = resnet_cifar.resnet18(pretrained=False, **{
                "num_classes": 100,})
        else :
            model = resnet_imagenet.resnet18()
    return model
            


    

