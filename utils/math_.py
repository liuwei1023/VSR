import torch

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

        if self.sum > 999999:
            self.sum = self.avg
            self.count = 1

def calc_psnr(x, y):
    diff = torch.abs(x - y)
    rmse = torch.sqrt(torch.mean(torch.pow(diff,2)))
    psnr = 20*torch.log10(1/rmse)
    return psnr