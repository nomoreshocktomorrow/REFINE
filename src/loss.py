import torch  
  
def loss_fn(pred: torch.Tensor, target: torch.Tensor, missing_mask: torch.Tensor) -> torch.Tensor:  
    N = target.shape[-1]  
    if pred.shape[-1] == N * 2:  
        mu = pred[..., :N]  
        log_sigma2 = pred[..., N:]  
        log_probs = torch.where(missing_mask, -torch.inf, mu).log_softmax(dim=-1).nan_to_num(0)  
        ce_term = -(log_probs * torch.where(missing_mask, 0, target))  
        inv_variance = torch.exp(-log_sigma2)  
        weighted_loss = 0.5 * ce_term * inv_variance + 0.5 * log_sigma2  
        weighted_loss = torch.where(missing_mask, torch.zeros_like(weighted_loss), weighted_loss)  
        return weighted_loss.sum(-1).sum(0)  
    else:  
        return -(torch.where(missing_mask, -torch.inf, pred).log_softmax(dim = -1).nan_to_num(0) * torch.where(missing_mask, 0, target)).sum(-1).sum(0) 
