import os
import argparse
import os.path as osp
import time
import pandas as pd
import torch
import numpy as np

from models.syntme_core import SynTMEFramework
from utlis import (EarlyStopping, load_dataloader, load_infer_dataloader,
                   set_random_seed, train, validate, infer)


def parse_runtime_args():
    parser = argparse.ArgumentParser(description="SynTME Execution Protocol")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--patience', type=int, default=50)
    parser.add_argument('--resume_from', type=str)
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test', 'infer'])
    parser.add_argument('--omic', type=str, default='exp,mut,cn,eff,dep,met')
    parser.add_argument('--workdir', type=str, default=os.getcwd())
    parser.add_argument('--celldataset', type=int, default=2)
    parser.add_argument('--cellencoder', type=str, default='cellCNNTrans')
    parser.add_argument('--cv_splits', type=str, default='0')
    parser.add_argument('--saved_model', type=str, default='./experiment/pretrained/0_fold_early_stop.pth')
    parser.add_argument('--infer_path', type=str, default='./data/split/0_fold_test_items.npy')
    parser.add_argument('--output_attn', type=int, default=1)
    return parser.parse_args()


def execute_pipeline():
    args = parse_runtime_args()
    set_random_seed(args.seed)

    _timestamp = time.strftime('%Y%m%d_%H%M', time.localtime())
    _artifact_dir = osp.join('experiment/', f'syntme_{_timestamp}')
    os.makedirs(_artifact_dir, exist_ok=True)

    print('\n[SynTME Configuration Parameters]')
    for k, v in vars(args).items():
        print(f'{k.upper()}: {v}')
    print('\n')

    if args.mode == 'train':
        _folds = args.cv_splits.split(',')

        for _k_idx in _folds:
            model = SynTMEFramework(args=args).to(args.device)
            model.init_weights()

            _loss_fn = torch.nn.MSELoss(reduction='mean')
            _optim = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.)
            _epoch_ptr = 0

            if args.resume_from:
                _state_cache = torch.load(args.resume_from)
                _current_state = model.state_dict()
                _valid_state = {k: v for k, v in _state_cache.items() if k in _current_state}
                _current_state.update(_valid_state)
                model.load_state_dict(_current_state)
                _epoch_ptr = int(osp.basename(args.resume_from).split('_')[0]) + 1

            tr_stream, val_stream, test_stream = load_dataloader(n_fold=_k_idx, args=args)
            _t0 = time.time()

            _callback_stopper = EarlyStopping(mode='lower', metric='mse', patience=args.patience, n_fold=_k_idx,
                                              folder=_artifact_dir)

            _optimum_mse = float('inf')
            _optimum_epoch = -1
            _chkp_path = osp.join(_artifact_dir, f'syntme_{_k_idx}_best_test.pth')

            for epoch in range(_epoch_ptr, args.epochs):
                _tr_loss, _ = train(model, _loss_fn, _optim, tr_stream, args.device, args)
                _val_loss, *_ = validate(model, _loss_fn, val_stream, args.device, args)

                print(f'[Epoch {epoch}] Loss -> TR: {_tr_loss:.4f} | VAL: {_val_loss:.4f}')

                _cache_val_loss = _callback_stopper.best_score
                _halt_flag = _callback_stopper.step(_val_loss, model)

                if epoch >= 70 and _cache_val_loss is not None and _val_loss < _cache_val_loss:
                    _callback_stopper.load_checkpoint(model)
                    _test_mse, *_ = validate(model, _loss_fn, test_stream, args.device, args)

                    if _test_mse < _optimum_mse:
                        _optimum_mse = _test_mse
                        _optimum_epoch = epoch
                        torch.save(model.state_dict(), _chkp_path)

                if _halt_flag:
                    print('[SynTME Protocol] Convergence reached. Halting.')
                    break

            _callback_stopper.load_checkpoint(model)
            _tr_metrics = validate(model, _loss_fn, tr_stream, args.device, args)
            _val_metrics = validate(model, _loss_fn, val_stream, args.device, args)

            print(f'Train Core Metrics (MSE/RMSE/MAE/R2/P/S): {tuple(round(m, 4) for m in _tr_metrics[:6])}')
            print(f'Valid Core Metrics (MSE/RMSE/MAE/R2/P/S): {tuple(round(m, 4) for m in _val_metrics[:6])}')

            if _optimum_epoch != -1:
                model.load_state_dict(torch.load(_chkp_path))
                _ts_metrics = validate(model, _loss_fn, test_stream, args.device, args)
                print(f'Optimum Test Profile (Epoch {_optimum_epoch}): {tuple(round(m, 4) for m in _ts_metrics[:6])}')

    elif args.mode == 'test':
        model = SynTMEFramework(args=args).to(args.device)
        try:
            model.load_state_dict(torch.load(args.saved_model), strict=True)
        except RuntimeError:
            model.load_state_dict(torch.load(args.saved_model), strict=False)

        _loss_fn = torch.nn.MSELoss(reduction='mean')
        _k_idx = osp.basename(args.saved_model).split('_')[0]
        _, _, test_stream = load_dataloader(n_fold=_k_idx, args=args)

        _ts_metrics = validate(model, _loss_fn, test_stream, args.device, args)
        print(f'Test Core Metrics: {tuple(round(m, 4) for m in _ts_metrics[:6])}')

    elif args.mode == 'infer':
        model = SynTMEFramework(args=args).to(args.device)
        model.load_state_dict(torch.load(args.saved_model))

        infer_stream, infer_registry = load_infer_dataloader(args=args)
        _preds, _embeds, _attns = infer(model, infer_stream, args.device, args)

        _valid_idx = len(_preds)
        _aggregated_out = np.concatenate((infer_registry[:_valid_idx], _preds), axis=1)

        pd.DataFrame(_aggregated_out,
                     columns=['Mol_A', 'Mol_B', 'Context_ID', 'Ground_Truth', 'Syn_Score_Pred']).to_csv(
            f'experiment/{_timestamp}/syntme_inference.csv')

        if args.output_attn:
            np.save(f'experiment/{_timestamp}/syntme_ctx_embed.npy', _embeds)
            np.save(f'experiment/{_timestamp}/syntme_attn_map.npy', _attns)


if __name__ == '__main__':
    execute_pipeline()