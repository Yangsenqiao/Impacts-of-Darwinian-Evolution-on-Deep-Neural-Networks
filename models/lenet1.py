import torch
import torch.nn as nn
class LeNet1(nn.Module):
    def __init__(self):
        super(LeNet1, self).__init__()

        # input is Nx1x28x28
        model_list = [
            # params: 4*(5*5*1 + 1) = 104
            # output is (28 - 5) + 1 = 24 => Nx4x24x24
            nn.Conv2d(1, 4, 5),
            nn.Tanh(),
            # output is 24/2 = 12 => Nx4x12x12
            nn.AvgPool2d(2),
            # params: (5*5*4 + 1) * 12 = 1212
            # output: 12 - 5 + 1 = 8 => Nx12x8x8
            nn.Conv2d(4, 12, 5),
            nn.Tanh(),
            # output: 8/2 = 4 => Nx12x4x4
            nn.AvgPool2d(2)
        ]

        self.model = nn.Sequential(*model_list)
        self.fc = nn.Linear(12 * 4 * 4, 10)

        # Total number of parameters = 104 + 1212 + 1930 = 3246

    def forward(self, x):
        out = self.model(x)
        out = out.view(x.size(0), -1)
        out = self.fc(out)
        return out