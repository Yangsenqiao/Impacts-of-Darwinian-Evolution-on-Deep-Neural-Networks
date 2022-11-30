from torch import nn
class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()#输入是batchsize*1*32*32
        self.conv1 = nn.Conv2d(1, 6, 5)#运行后变成batchsize*6*28*28
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)#运行后变成batchsize*6*14*14
        self.conv2 = nn.Conv2d(6, 16, 5)#运行后变成batchsize*16*10*10
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)#运行后变成batchsize*16*5*5
        self.conv3 = nn.Conv2d(16, 120, 5)#运行后变成batchsize*120*1*1
        self.relu3 = nn.ReLU()
        self.fc1 = nn.Linear(120, 84)#运行后变成batchsize*84
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(84, 10)#运行后变成batchsize*10
        self.relu5 = nn.ReLU()
        
    def forward(self, x):
        y = self.conv1(x)
        y = self.relu1(y)
        y = self.pool1(y)
        y = self.conv2(y)
        y = self.relu2(y)
        y = self.pool2(y)
        y = self.conv3(y)
        y = self.relu3(y)

        y = y.view(y.shape[0], -1)
        y = self.fc1(y)
        y = self.relu4(y)
        y = self.fc2(y)
        y = self.relu5(y)
        return y
    