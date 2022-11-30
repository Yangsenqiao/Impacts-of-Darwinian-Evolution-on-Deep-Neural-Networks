import numpy as np
import torch
import torch.nn as nn

def model_weights_as_vector(model):
    weights_vector = []

    for curr_weights in model.state_dict().values():
        curr_weights = curr_weights.detach().cpu().numpy()
        vector = np.reshape(curr_weights, newshape=(curr_weights.size))
        weights_vector.extend(vector)
    return np.array(weights_vector)

def model_weights_as_dict(model, weights_vector):
    weights_dict = model.state_dict()
    start = 0
    for key in weights_dict:
        w_matrix = weights_dict[key].detach().cpu().numpy()
        layer_weights_shape = w_matrix.shape
        layer_weights_size = w_matrix.size
        layer_weights_vector = weights_vector[start:start + layer_weights_size]
        layer_weights_matrix = np.reshape(layer_weights_vector, newshape=(layer_weights_shape))
        weights_dict[key] = torch.from_numpy(layer_weights_matrix)
        start = start + layer_weights_size
    return weights_dict

def Init_param(name,shape,start,end,mean_cur=0,std_cur=1):
    '''
    :param name: use which method to init the param
    :param shape: the params' shape
    :param start: the params' upper bound
    :param end:  the params' lower bound
    :param mean_cur: the params' mean( when use 'normal')
    :param std_cur: the params' std( when use 'std')
    :return: the Inited params
    '''
    data=torch.Tensor(shape[0],shape[1])
    if name=='uniform':
        return nn.init.uniform_(data,start,end)
    elif name=='normal':
        return nn.init.normal_(data,mean=mean_cur,std=std_cur)
    elif name=='eye':#use Identity matrix
        return nn.init.eye_(data)
    elif name=='sparse':
        return nn.init.sparse_(data, sparsity=0.1,std=std_cur)#没有设置外部接口，参数过多
    elif name=='orthogonal':
        return nn.init.orthogonal_(data,gain=1)#
    elif name=='kaiming_normal':
        return nn.init.kaiming_normal_(data, mode='fan_out', nonlinearity='relu')

    elif name=='kaiming_uniform':
        return nn.init.kaiming_uniform_(data,mode='fan_out')
    elif name=='xavier_uniform':
        return nn.init.xavier_uniform_(data, gain=nn.init.calculate_gain('relu'))
    elif name=='xavier_normal':
        return nn.init.xavier_normal_(data)
    else:
        return nn.init.uniform_(data,start,end)
    

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, model, kwargs):
        conf = kwargs["conf"]
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 20
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            threshold(float): The answer of the Loss must smaller than this threshold
                            Default:0.0
            delay(int): if the minmum loss is bigger than the threshold we will wait delay
        """
        self.patience = conf.Early.patience
        self.verbose = conf.Early.verbose
        self.counter = 0
        self.best_score = None
        self.stop = False
        self.delta = conf.Early.delta
        self.threshold = conf.Early.threshold
        self.delay = conf.Early.delay
        self.model_num = 0
        self.popsize = conf.DE.popsize
        self.population = np.zeros((self.popsize, len(model_weights_as_vector(model))))

    def __call__(self, val_loss, model): 
        score = val_loss
        if self.best_score is None:
            self.best_score = score
            self.population[self.model_num] = model_weights_as_vector(model).copy()
            self.model_num = (self.model_num+1)% self.popsize
            
        elif score < self.best_score - self.delta:
            self.counter += 1
            self.population[self.model_num] = model_weights_as_vector(model).copy()
            self.model_num = (self.model_num+1)% self.popsize
            if (self.counter >= self.patience and self.best_score > self.threshold) or (
                    self.counter >= self.patience + self.delay):
                self.stop = True
        else:
            self.best_score = score
            self.population[self.model_num] = model_weights_as_vector(model).copy()
            self.model_num = (self.model_num+1)% self.popsize
            self.counter = 0
