from torch import nn
class RNN(nn.Module):
    def __init__(self, Input_Size, Time_Step, Out_Size):
        super(RNN, self).__init__()
        self.input_size = Input_Size
        self.time_step = Time_Step

        self.rnn = nn.LSTM(         # if use nn.RNN(), it hardly learns
            input_size=Input_Size,
            hidden_size=64,         # rnn hidden unit
            num_layers=1,           # number of rnn layer
            batch_first=True,       # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        )

        self.out = nn.Linear(64, Out_Size)

    def forward(self, x):
        x=x.view(x.size()[0], self.time_step, self.input_size)
        r_out, (h_n, h_c) = self.rnn(x, None)   # None represents zero initial hidden state
        # choose r_out at the last time step
        out = self.out(r_out[:, -1, :])
        return out
