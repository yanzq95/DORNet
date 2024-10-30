import torch

def calc_rmse(a, b, minmax):
    a = a[6:-6, 6:-6]
    b = b[6:-6, 6:-6]
    
    a = a*(minmax[0]-minmax[1]) + minmax[1]
    b = b*(minmax[0]-minmax[1]) + minmax[1]
    a = a * 100
    b = b * 100
    
    return torch.sqrt(torch.mean(torch.pow(a-b,2)))
    

def rgbdd_calc_rmse(gt, out, minmax):
    gt = gt[6:-6, 6:-6]
    out = out[6:-6, 6:-6]

    out = out*(minmax[0]-minmax[1]) + minmax[1]
    gt = gt / 10.0
    out = out / 10.0
    
    return torch.sqrt(torch.mean(torch.pow(gt-out,2)))
    
def tofdsr_calc_rmse(gt, out, minmax):
    gt = gt[6:-6, 6:-6]
    out = out[6:-6, 6:-6]
    
    mask = (gt >= 100) & (gt <= 5000)
    gt = gt[mask]
    out = out[mask]

    out = out*(minmax[0]-minmax[1]) + minmax[1]
    gt = gt / 10.0
    out = out / 10.0
    
    return torch.sqrt(torch.mean(torch.pow(gt-out,2)))
