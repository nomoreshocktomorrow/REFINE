import argparse
import pickle

import torch

from src.utils import preprocess_data
from src.target_portfolio import MaxSharpeWeightCollection, MaxSortinoWeightCollection, MaxLogUtilityWeightCollection, MinCVaRWeightCollection

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file_name', default = 'universe1', type = str)
    parser.add_argument('-n', '--num_epoch', default = 1000, type = int)
    parser.add_argument('--end_date', default = '', type = str)
    parser.add_argument('--objective', default='all', type=str, choices=['all', 'maxsharpe', 'maxsortino', 'maxlogutility', 'mincvar'])
    args = parser.parse_args()

    yearmons, data, torch_data, torch_target_data, missing_mask = preprocess_data('data/' + args.file_name + '.csv', '2010-01', device, end_date=args.end_date)

    
    if args.objective in ['all', 'mincvar']:
        weight_collection = MinCVaRWeightCollection(yearmons, data['log_tr'].columns.__len__(), device)
        weight_collection.train_min_cvar_weights(data, missing_mask, num_epoch = args.num_epoch)
        weights = weight_collection.get_weights()
        with open(f'result/{args.file_name}_weights_mincvar.pkl', 'wb') as f:
            pickle.dump(weights, f)

    if args.objective in ['all', 'maxsharpe']:
        weight_collection = MaxSharpeWeightCollection(yearmons, data['log_tr'].columns.__len__(), device)
        weight_collection.train_max_sharpe_weights(data, missing_mask, num_epoch = args.num_epoch)
        weights = weight_collection.get_weights()
        with open(f'result/{args.file_name}_weights_maxsharpe.pkl', 'wb') as f:
            pickle.dump(weights, f)

    if args.objective in ['all', 'maxsortino']:
        weight_collection = MaxSortinoWeightCollection(yearmons, data['log_tr'].columns.__len__(), device)
        weight_collection.train_max_sortino_weights(data, missing_mask, num_epoch = args.num_epoch)
        weights = weight_collection.get_weights()
        with open(f'result/{args.file_name}_weights_maxsortino.pkl', 'wb') as f:
            pickle.dump(weights, f)

    if args.objective in ['all', 'maxlogutility']:
        weight_collection = MaxLogUtilityWeightCollection(yearmons, data['log_tr'].columns.__len__(), device)
        weight_collection.train_max_logutility_weights(data, missing_mask, num_epoch = args.num_epoch)
        weights = weight_collection.get_weights()
        with open(f'result/{args.file_name}_weights_maxlogutility.pkl', 'wb') as f:
            pickle.dump(weights, f)
