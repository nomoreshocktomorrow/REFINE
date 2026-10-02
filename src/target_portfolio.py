from typing import List, Dict
import tqdm

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sophia import SophiaG


class MaxSharpeWeightCollection(nn.Module):
    def __init__(self, yearmons: List[str], num_item: int, device: torch.device):
        super().__init__()
        self.device = device

        self.yearmons = yearmons
        self.num_item = num_item
        self.log_weight = nn.ParameterDict(
            {yearmon:torch.zeros((self.num_item), dtype = torch.float, device = self.device)
             for yearmon in yearmons}
        ) 

    def objective(self, yearmon:str, mean: torch.Tensor, cov: torch.Tensor, target_data: torch.Tensor, mask: torch.Tensor, prev_sharpe: torch.float):
        weight = self.log_weight[yearmon][~mask].softmax(dim = 0)
        target_data = target_data[:, ~mask]
        seq_len = target_data.shape[0]

        if not torch.isfinite(weight).all():
            raise ValueError(f"Invalid weight at index {yearmon}")

        mu = (mean * weight).sum()
        w64 = weight.to(dtype=cov.dtype)
        q = (w64.reshape(1, -1) @ cov @ w64).squeeze()
        sigma = torch.sqrt(torch.clamp(q, min=1e-8)).to(dtype=weight.dtype)
        loss = - mu * seq_len + prev_sharpe * sigma * np.sqrt(seq_len)
        
        with torch.no_grad():
            backtest = (target_data.exp() * weight).sum(-1).log()
            var = backtest.var(unbiased=False)
            sharpe = (backtest.mean(0) * seq_len) / torch.sqrt(torch.clamp(var * seq_len, min=1e-12))
            entropy = -(weight * (weight + 1e-12).log()).sum(-1)

        return loss, sharpe, entropy

    def train_max_sharpe_weights(self, data: pd.DataFrame, missing_mask: Dict[str, torch.Tensor], num_epoch: int = 1000):
        self.missing_mask = missing_mask

        for yearmon in (pbar := tqdm.tqdm(self.yearmons)):
            target_data = torch.tensor(data['log_tr'].loc[yearmon].values, dtype = torch.float, device = self.device)

            mean = target_data.mean(0)[~missing_mask[yearmon]]
            td = target_data.to(dtype=torch.float64)
            cov = td.mT.cov()[:, ~missing_mask[yearmon]][~missing_mask[yearmon], :]
            cov = (cov + cov.mT) * 0.5
            min_eig = torch.linalg.eigvalsh(cov).min()
            shift = torch.clamp(torch.as_tensor(1e-8, dtype=cov.dtype, device=cov.device) - min_eig, min=0.0)
            if shift.item() > 0:
                cov = cov + shift * torch.eye(cov.shape[0], dtype=cov.dtype, device=cov.device)

            try:
                self.log_weight[yearmon].data.fill_(0)
                optimizer = SophiaG([self.log_weight[yearmon]], lr = 5e-1)
                
                sharpe = 0.0
                for epoch in range(num_epoch):
                    optimizer.zero_grad(set_to_none = True)
                    loss, sharpe, entropy = self.objective(yearmon, mean, cov, target_data, missing_mask[yearmon], sharpe)
                    loss.backward()
                    optimizer.step()

                    if not torch.isfinite(self.log_weight[yearmon]).all():
                        self.log_weight[yearmon].data.zero_()
                        break
    
                pbar.set_description(f"Sharpe = {sharpe:.6f}, Entropy = {entropy:.6f}")
            except Exception as e:
                print(e)
                self.log_weight[yearmon].data.zero_()
                

    def get_weights(self):
        return {yearmon:torch.where(self.missing_mask[yearmon], -np.inf, self.log_weight[yearmon].data).softmax(-1).cpu() for yearmon in self.yearmons}

class MaxSortinoWeightCollection(nn.Module):
    def __init__(self, yearmons: List[str], num_item: int, device: torch.device):
        super().__init__()
        self.device = device

        self.yearmons = yearmons
        self.num_item = num_item
        self.log_weight = nn.ParameterDict(
            {yearmon:torch.zeros((self.num_item), dtype = torch.float, device = self.device)
             for yearmon in yearmons}
        ) 

    def objective(self, yearmon:str, mean: torch.Tensor, cov: torch.Tensor, target_data: torch.Tensor, mask: torch.Tensor, prev_sortino: torch.float):
        weight = self.log_weight[yearmon][~mask].softmax(dim = 0)
        target_data = target_data[:, ~mask]
        seq_len = target_data.shape[0]

        if not torch.isfinite(weight).all():
            raise ValueError(f"Invalid weight at index {yearmon}")

        mu = (mean * weight).sum()
        sigma = (weight.reshape(1, -1) @ cov @ weight).sqrt()
        loss = - mu * seq_len + prev_sortino * sigma * np.sqrt(seq_len)
        
        with torch.no_grad():
            backtest = (target_data.exp() * weight).sum(-1).log()
            sortino = (backtest.mean(0) * seq_len) / (backtest.var() * seq_len).sqrt()
            entropy = -(weight * (weight + 1e-12).log()).sum(-1)

        return loss, sortino, entropy

    def train_max_sortino_weights(self, data: pd.DataFrame, missing_mask: Dict[str, torch.Tensor], num_epoch: int = 1000):
        self.missing_mask = missing_mask

        for yearmon in (pbar := tqdm.tqdm(self.yearmons)):
            target_data = torch.tensor(data['log_tr'].loc[yearmon].values, dtype = torch.float, device = self.device)

            mean = target_data.mean(0)[~missing_mask[yearmon]]
            cov = target_data.clip(max=0).mT.cov()[:, ~missing_mask[yearmon]][~missing_mask[yearmon], :]

            try:
                self.log_weight[yearmon].data.fill_(0)
                optimizer = SophiaG([self.log_weight[yearmon]], lr = 5e-1)

                sortino = 0.0
                for epoch in range(num_epoch):
                    optimizer.zero_grad(set_to_none = True)
                    loss, sortino, entropy = self.objective(yearmon, mean, cov, target_data, missing_mask[yearmon], sortino)
                    loss.backward()
                    optimizer.step()

                pbar.set_description(f"Sortino = {sortino:.6f}, Entropy = {entropy:.6f}")
            except Exception as e:
                print(e)
                self.log_weight[yearmon].data.fill_(float('nan'))

    def get_weights(self):
        return {yearmon:torch.where(self.missing_mask[yearmon], -np.inf, self.log_weight[yearmon].data).softmax(-1).cpu() for yearmon in self.yearmons}

class MaxLogUtilityWeightCollection(nn.Module):
    def __init__(self, yearmons: List[str], num_item: int, device: torch.device):
        super().__init__()
        self.device = device

        self.yearmons = yearmons
        self.num_item = num_item
        self.log_weight = nn.ParameterDict(
            {yearmon:torch.zeros((self.num_item), dtype = torch.float, device = self.device)
             for yearmon in yearmons}
        ) 

    def objective(self, yearmon:str, target_data: torch.Tensor, mask: torch.Tensor):
        weight = self.log_weight[yearmon][~mask].softmax(dim = 0)
        target_data = target_data[:, ~mask]

        if not torch.isfinite(weight).all():
            raise ValueError(f"Invalid weight at index {yearmon}")

        port_returns = (target_data.exp() * weight).sum(-1).log()
        
        loss = -port_returns.sum()
        
        with torch.no_grad():
            entropy = -(weight * (weight + 1e-12).log()).sum(-1)
            log_utility = port_returns.sum()

        return loss, log_utility, entropy

    def train_max_logutility_weights(self, data: pd.DataFrame, missing_mask: Dict[str, torch.Tensor], num_epoch: int = 1000):
        self.missing_mask = missing_mask

        for yearmon in (pbar := tqdm.tqdm(self.yearmons)):
            target_data = torch.tensor(data['log_tr'].loc[yearmon].values, dtype = torch.float, device = self.device)

            try:
                self.log_weight[yearmon].data.fill_(0)
                optimizer = SophiaG([self.log_weight[yearmon]], lr = 5e-1)
                
                log_utility = 0.0
                for epoch in range(num_epoch):
                    optimizer.zero_grad(set_to_none = True)
                    loss, log_utility, entropy = self.objective(yearmon, target_data, missing_mask[yearmon])
                    loss.backward()
                    optimizer.step()

                    if not torch.isfinite(self.log_weight[yearmon]).all():
                        self.log_weight[yearmon].data.zero_()
                        break
    
                pbar.set_description(f"LogUtil = {log_utility:.6f}, Entropy = {entropy:.6f}")
            except Exception as e:
                print(e)
                self.log_weight[yearmon].data.zero_()

    def get_weights(self):
        return {yearmon:torch.where(self.missing_mask[yearmon], -np.inf, self.log_weight[yearmon].data).softmax(-1).cpu() for yearmon in self.yearmons}



class MinCVaRWeightCollection:
    def __init__(self, yearmons: List[str], num_assets: int, device: torch.device):
        self.yearmons = yearmons
        self.device = device
        self.log_weight = {yearmon:torch.zeros(num_assets, requires_grad=True, device=device) for yearmon in yearmons}
        self.missing_mask = None
        self.alpha = 0.15

    def objective(self, yearmon: str, target_data: torch.Tensor, missing_mask: torch.Tensor):
        weight = torch.where(missing_mask, -np.inf, self.log_weight[yearmon]).softmax(-1)
        
        returns = target_data.exp() - 1
        port_returns = (returns * weight).sum(-1)
        
        k = max(1, int(port_returns.size(0) * self.alpha))
        worst_returns, _ = torch.topk(port_returns, k, largest=False)
        
        cvar = -worst_returns.mean()
        
        entropy = -(weight * (weight + 1e-12).log()).sum(-1)
        
        loss = cvar - 0.005 * entropy
        
        return loss, cvar.item(), entropy.item()

    def train_min_cvar_weights(self, data: pd.DataFrame, missing_mask: Dict[str, torch.Tensor], num_epoch: int = 1000):
        self.missing_mask = missing_mask

        for yearmon in (pbar := tqdm.tqdm(self.yearmons)):
            target_data = torch.tensor(data['log_tr'].loc[yearmon].values, dtype=torch.float, device=self.device)

            try:
                self.log_weight[yearmon].data.fill_(0)
                optimizer = SophiaG([self.log_weight[yearmon]], lr=5e-2)
                
                cvar_val = 0.0
                for epoch in range(num_epoch):
                    optimizer.zero_grad(set_to_none=True)
                    loss, cvar_val, entropy = self.objective(yearmon, target_data, missing_mask[yearmon])
                    loss.backward()
                    optimizer.step()

                    if not torch.isfinite(self.log_weight[yearmon]).all():
                        self.log_weight[yearmon].data.zero_()
                        break
    
                pbar.set_description(f"CVaR = {cvar_val:.6f}, Entropy = {entropy:.6f}")
            except Exception as e:
                print(e)
                self.log_weight[yearmon].data.zero_()

    def get_weights(self):
        return {yearmon:torch.where(self.missing_mask[yearmon], -np.inf, self.log_weight[yearmon].data).softmax(-1).cpu() for yearmon in self.yearmons}
