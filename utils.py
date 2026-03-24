import torch.nn as nn

def loss_fn(pred, target):
    cos = nn.CosineSimilarity(dim=1)
    
    cosine_loss = 1 - cos(pred, target).mean()
    l2_loss = nn.MSELoss()(pred, target)
    
    return cosine_loss + l2_loss