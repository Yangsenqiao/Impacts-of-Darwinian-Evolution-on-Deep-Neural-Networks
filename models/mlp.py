import torch.nn as nn
class MLP(nn.Module):
    def __init__(self, Input_Size, Time_Step, Out_Size):
        super(MLP, self).__init__()
        self.input_size = Input_Size
        self.time_step = Time_Step
        self.mlp=nn.Sequential(
            nn.Linear(Input_Size * Time_Step, 64),
            nn.ReLU(),
            nn.Linear(64, Out_Size)
        )

    def forward(self, x):
        x = x.view(-1, self.input_size * self.time_step)
        x=self.mlp(x)
        return x
    