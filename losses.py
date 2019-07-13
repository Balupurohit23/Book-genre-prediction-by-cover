from torch import nn

class LossFn(nn.Module):
    def __init__(self):
        super(LossFn, self).__init__()
        self.l = nn.NLLLoss()

    def forward(self, *params):

        loss = self.l(params[0], params[1])
        return loss
