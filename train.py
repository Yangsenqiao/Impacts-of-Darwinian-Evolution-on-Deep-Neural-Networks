import argparse
from fileinput import filename
import imp
from statistics import mode
import numpy as np
import yaml
import easydict
import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
from models.GA_model import *
from utils import *
from utils.defaults import *
from utils.evolution import *
import torch.nn as nn
import torch.nn.functional as F
import torch
import warnings
from utils.tools import *
from eval import predict
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='./configs/LeNet1_MNIST.yaml')
parser.add_argument('--Adam_epoch', type=int, default = 30)
parser.add_argument('--DE_epoch', type=int, default = 200)
parser.add_argument('--model_name', type = str, default = 'LeNet1')
parser.add_argument('--lr', type=float, default=3e-2)
parser.add_argument('--dir', type=str, default='./data')
parser.add_argument('--print_epoch', type=int, default=50, help='print frequence')
parser.add_argument('--DE_eval', type=int, default=5)

args = parser.parse_args()
config_file = args.config
conf = yaml.full_load(open(config_file))
conf = easydict.EasyDict(conf)

conf.Adam.lr = args.lr
conf.Adam.all_epoch = args.Adam_epoch
conf.DE.all_epoch = args.DE_epoch
conf.model_name = args.model_name


inputs = vars(args)
inputs["conf"] = conf


def score_func(solution1, solution2):
    model.eval()
    model_weights_dict = model_weights_as_dict(model=model,
                                               weights_vector = solution1)
    model.load_state_dict(model_weights_dict)
    model2.eval()
    model_weights_dict = model_weights_as_dict(model=model2,
                                               weights_vector = solution2)
    model2.load_state_dict(model_weights_dict)
    All_loss1 = 0
    All_loss2 = 0
    with torch.no_grad():
        for batch, (image, target) in enumerate(TrainLoader):
            image = image.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            output = model(image)
            output2 = model2(image)
            loss1 = criterion(output, target)
            loss2 = criterion(output2, target)
            All_loss1 += loss1.item() * image.size(0)
            All_loss2 += loss2.item() * image.size(0)
    return min(All_loss1, All_loss2),2 if All_loss1 > All_loss2 else 1

def Adam_train():
    model.train()
    for epoch in range(conf.Adam.all_epoch):
        if  earlystop.stop :
            return epoch
        for batch, (image, target) in enumerate(TrainLoader):
            image = image.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            output = model(image)
            loss = criterion(output, target)
            Adam_optimizer.zero_grad()
            loss.backward()
            Adam_optimizer.step()
        cur_acc, cur_loss = predict(model, Vali_Loader)
        earlystop(val_loss = cur_loss, model=model)
        if epoch % args.print_epoch ==0:
            print('{}: epoch: {}, loss : {}, vali acc D:{}'.format\
                ('Adam', epoch, loss.item(),  cur_acc))
    
    return epoch
        
            
def GA_Train(start_epoch):
    population = earlystop.population
    for epoch in range(start_epoch, start_epoch + conf.DE.all_epoch):
        solution, best_score, population, idx = GA_optimizer.minimize(population)
        model_weights_dict = model_weights_as_dict(model=model, weights_vector = solution)
        model.load_state_dict(model_weights_dict)    
        if epoch % args.DE_eval == 0:
            cur_acc, cur_loss = predict(model, Vali_Loader)
            print('{}: epoch: {}, vali loss : {}, vali acc:{}'.format\
                    ('GA', epoch, cur_loss,  cur_acc))
        
        
        
if __name__ == '__main__':
    criterion = torch.nn.CrossEntropyLoss()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    TrainLoader, Vali_Loader, TestLoader = get_data_loaders(inputs)
    model = get_models(inputs)
    model2 = get_models(inputs)
    Adam_optimizer = torch.optim.Adam(model.parameters(), lr = conf.Adam.lr, weight_decay = conf.Adam.decay)
    model.to(device)
    model2.to(device)
    criterion.to(device)
    earlystop = EarlyStopping(model, inputs)
    GA_optimizer = DE(bounds = conf.DE.bounds, popsize = conf.DE.popsize, mutate = conf.DE.mutate, recombination = conf.DE.recomb, cost_func = score_func)
    GA = TorchGA(model = model, num_solutions = conf.DE.popsize)
    population = GA.population_weights
    start_epoch = Adam_train()           
    GA_Train(start_epoch)

