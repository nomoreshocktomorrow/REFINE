from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import torch

def get_yearmons(data: pd.DataFrame) -> List[str]:
    first_date = data.index.min()
    last_date = data.index.max()
    yearmons = []
    for year in range(first_date.year, last_date.year + 1):
        for month in range(12):
            if year == first_date.year and month + 1 < first_date.month:
                continue
    
            yearmon = f'{year}-{month + 1:02d}'
    
            if year == last_date.year and month + 1 == last_date.month:
                break
    
            yearmons.append(yearmon)
    return yearmons

def preprocess_data(file_name: str, start_date: str, device: torch.device, end_date: str = '') -> Tuple[List[str], pd.DataFrame, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
    data = pd.read_csv(file_name)
    data['date'] = pd.to_datetime(data['date'])
    data = data.sort_values(['ticker', 'date'])
    data = data.set_index(['ticker', 'date'])

    data['open'] = np.log(data['open'] / data['close'])
    data['high'] = np.log(data['high'] / data['close'])
    data['low'] = np.log(data['low'] / data['close'])
    data['volume'] = np.log(data['volume'].clip(lower=1.0))
    data['log_tr'] = np.log(data.groupby(level=0)['div_adj_close'].pct_change() + 1)
    data = data.drop(['close', 'div_adj_close'], axis = 1)
    data = data.unstack(level = 0)
    data = data[(~data['log_tr'].isna()).sum(1) > 1]
    data = data[(~(data['log_tr'].isna() | (data['log_tr'] == 0))).sum(1) > 0]
    data = data.loc[data.index.intersection(pd.bdate_range(data.index.min(), data.index.max()))]
    data = data.loc[start_date:]
    if str(end_date).strip():
        data = data.loc[:end_date]

    torch_data = torch.tensor(data.values, dtype = torch.float, device = device).nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)
    torch_target_data = torch_data[:, -data['log_tr'].columns.__len__():]

    yearmons = get_yearmons(data)

    missing_mask = {yearmon:torch.tensor(data['log_tr'].loc[yearmon].iloc[0].isna().values, dtype = torch.bool, device = device)
                    for yearmon in yearmons}

    return yearmons, data.fillna(0), torch_data, torch_target_data, missing_mask

def preprocess_backtest_data(file_name: str, device: torch.device) -> Tuple[List[str], pd.DataFrame]:
    data = pd.read_csv(file_name)
    data['date'] = pd.to_datetime(data['date'])
    data = data.sort_values(['ticker', 'date'])
    data = data.set_index(['ticker', 'date'])

    data['log_tr'] = np.log(data.groupby(level=0)['div_adj_close'].pct_change() + 1)
    data = data[['volume', 'close', 'log_tr']]
    data = data.unstack(level = 0)
    data = data[(~data['log_tr'].isna()).sum(1) > 0]
    data = data[(~(data['log_tr'].isna() | (data['log_tr'] == 0))).sum(1) > 0]
    data = data.loc[data.index.intersection(pd.bdate_range(data.index.min(), data.index.max()))]

    yearmons = get_yearmons(data)
    
    missing_mask = {yearmon:torch.tensor(data['log_tr'].loc[yearmon].iloc[0].isna().values, dtype = torch.bool, device = device)
                    for yearmon in yearmons}

    return yearmons, data, missing_mask