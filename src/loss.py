import torch

class RMSELoss(torch.nn.Module):
    def __init__(self):
        super(RMSELoss,self).__init__()
        self.mse = torch.nn.MSELoss()
        self.eps = 1e-7

    def forward(self,y,y_hat):
        return torch.sqrt(self.mse(y,y_hat) + self.eps)

class RMSLELoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = torch.nn.MSELoss()
        
    def forward(self, pred, actual):
        return torch.sqrt(self.mse(torch.log(pred + 1), torch.log(actual + 1)))