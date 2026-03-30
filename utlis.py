import os
import os.path as osp
import random
import numpy as np
import torch
from mmcv.utils import collect_env as collect_base_env
from torch_geometric.loader import DataLoader
from dataset.syntme_dataset import SynTMETopologyDataset
from metrics import get_metrics


def set_random_seed(seed, deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class EarlyStopping():
    def __init__(self, mode='higher', patience=50, filename=None, metric=None, n_fold=None, folder=None):
        if filename is None:
            filename = os.path.join(folder, f'syntme_{n_fold}_convergence_point.pth')

        if metric is not None:
            assert metric in ['r2', 'mae', 'rmse', 'roc_auc_score', 'pr_auc_score', 'mse']
            if metric in ['r2', 'roc_auc_score', 'pr_auc_score']:
                mode = 'higher'
            if metric in ['mae', 'rmse', 'mse']:
                mode = 'lower'

        assert mode in ['higher', 'lower']
        self.mode = mode
        self._eval_criteria = self._check_higher if self.mode == 'higher' else self._check_lower

        self.patience = patience
        self._stagnation_ticks = 0
        self.filename = filename
        self.best_score = None
        self.early_stop = False

    def _check_higher(self, score, prev_best_score):
        return score > prev_best_score

    def _check_lower(self, score, prev_best_score):
        return score < prev_best_score

    def step(self, current_metric, active_model):
        if self.best_score is None:
            self.best_score = current_metric
            self.save_checkpoint(active_model)
        elif self._eval_criteria(current_metric, self.best_score):
            self.best_score = current_metric
            self.save_checkpoint(active_model)
            self._stagnation_ticks = 0
        else:
            self._stagnation_ticks += 1
            if self._stagnation_ticks >= self.patience:
                self.early_stop = True
        return self.early_stop

    def save_checkpoint(self, active_model):
        torch.save(active_model.state_dict(), self.filename)

    def load_checkpoint(self, active_model):
        active_model.load_state_dict(torch.load(self.filename))


def collect_env():
    return collect_base_env()


def load_dataloader(n_fold, args):
    _data_dir = osp.join(args.workdir, 'data')

    if args.celldataset == 1:
        _ctx_path = osp.join(_data_dir, '0_cell_data/18498g/985_cellGraphs_exp_mut_cn_18498_genes_norm.npy')
    elif args.celldataset == 2:
        _ctx_path = osp.join(_data_dir, '0_cell_data/4079g/985_cellGraphs_exp_mut_cn_eff_dep_met_4079_genes_norm.npy')
    elif args.celldataset == 3:
        _ctx_path = osp.join(_data_dir, '0_cell_data/963g/985_cellGraphs_exp_mut_cn_963_genes_norm.npy')

    _mol_path = osp.join(_data_dir, '1_drug_data/drugSmile_drugSubEmbed_2644.npy')
    _microenv_path = osp.join(_data_dir, '0_cell_data/cellTme.npy')

    _tr_idx = osp.join(_data_dir, f'split/{n_fold}_fold_tr_items.npy')
    _val_idx = osp.join(_data_dir, f'split/{n_fold}_fold_val_items.npy')
    _ts_idx = osp.join(_data_dir, f'split/{n_fold}_fold_test_items.npy')

    _tr_manifold = SynTMETopologyDataset(_data_dir, _tr_idx, _ctx_path, _mol_path, _microenv_path, args=args)
    _val_manifold = SynTMETopologyDataset(_data_dir, _val_idx, _ctx_path, _mol_path, _microenv_path, args=args)
    _ts_manifold = SynTMETopologyDataset(_data_dir, _ts_idx, _ctx_path, _mol_path, _microenv_path, args=args)

    _tr_stream = DataLoader(_tr_manifold, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=False)
    _val_stream = DataLoader(_val_manifold, batch_size=args.batch_size, shuffle=False, num_workers=4, drop_last=False)
    _ts_stream = DataLoader(_ts_manifold, batch_size=args.batch_size, shuffle=False, num_workers=4, drop_last=False)

    return _tr_stream, _val_stream, _ts_stream


def load_infer_dataloader(args):
    _data_dir = osp.join(args.workdir, 'data')

    if args.celldataset == 1:
        _ctx_path = osp.join(_data_dir, '0_cell_data/18498g/985_cellGraphs_exp_mut_cn_18498_genes_norm.npy')
    elif args.celldataset == 2:
        _ctx_path = osp.join(_data_dir, '0_cell_data/4079g/985_cellGraphs_exp_mut_cn_eff_dep_met_4079_genes_norm.npy')
    elif args.celldataset == 3:
        _ctx_path = osp.join(_data_dir, '0_cell_data/963g/985_cellGraphs_exp_mut_cn_963_genes_norm.npy')

    _mol_path = osp.join(_data_dir, '1_drug_data/drugSmile_drugSubEmbed_2644.npy')
    _microenv_path = osp.join(_data_dir, '0_cell_data/cellTme.npy')
    _infer_idx = args.infer_path

    _infer_manifold = SynTMETopologyDataset(_data_dir, _infer_idx, _ctx_path, _mol_path, _microenv_path, args=args)
    _infer_stream = DataLoader(_infer_manifold, batch_size=384, shuffle=False, num_workers=4, drop_last=False)

    _infer_registry = np.load(_infer_idx, allow_pickle=True)

    return _infer_stream, _infer_registry


def train(model, criterion, opt, dataloader, device, args=None):
    model.train()
    _cumulative_empirical_risk = 0
    _step_ptr = 0
    _optim_schedule = []

    for _payload in dataloader:
        _step_ptr += 1
        model.zero_grad()
        _stream_input = _payload.to(device)

        _syn_score_logits, _, _ = model(_stream_input)
        _ground_truth_tensor = _payload.y.unsqueeze(1).to(dtype=torch.float32, device=device)

        _loss_manifold = criterion(_syn_score_logits, _ground_truth_tensor)
        _cumulative_empirical_risk += _loss_manifold

        _loss_manifold.backward()
        opt.step()

    _cumulative_empirical_risk = _cumulative_empirical_risk.cpu().detach().numpy()
    _normalized_loss = _cumulative_empirical_risk / _step_ptr

    return _normalized_loss, _optim_schedule


def validate(model, criterion, dataloader, device, args=None):
    model.eval()
    _gt_registry = []
    _pred_registry = []

    with torch.no_grad():
        for _payload in dataloader:
            _stream_input = _payload.to(device)
            _gt_tensor = _payload.y.unsqueeze(1).to(device)

            _gt_registry.append(_gt_tensor.view(-1, 1))
            _syn_score_logits, _, _ = model(_stream_input)
            _pred_registry.append(_syn_score_logits)

    _gt_registry = torch.cat(_gt_registry, dim=0).cpu().detach().numpy()
    _pred_registry = torch.cat(_pred_registry, dim=0).cpu().detach().numpy()

    mse, rmse, mae, r2, pearson, spearman = get_metrics(_gt_registry, _pred_registry)
    return mse, rmse, mae, r2, pearson, spearman, None


def infer(model, dataloader, device, args=None):
    model.eval()
    _pred_registry = []
    _gt_registry = []
    _ctx_manifold_embeds = []
    _attn_topologies = []

    with torch.no_grad():
        for _payload in dataloader:
            _stream_input = _payload.to(device)
            _syn_score_logits, _latent_embed, _attn_map = model(_stream_input)

            if args.output_attn:
                _ctx_manifold_embeds.append(_latent_embed.cpu())
                _attn_topologies.append(_attn_map.cpu())

            _gt_tensor = _payload.y.unsqueeze(1).to(device)
            _gt_registry.append(_gt_tensor.view(-1, 1).cpu())
            _pred_registry.append(_syn_score_logits.cpu())

    _gt_registry = torch.cat(_gt_registry, dim=0).cpu().detach().numpy()
    _pred_registry = torch.cat(_pred_registry, dim=0).cpu().detach().numpy()

    mse, rmse, mae, r2, pearson, spearman = get_metrics(_gt_registry, _pred_registry)
    print(
        '[SynTME Inference Profile] MSE:{:.2f} RMSE:{:.2f} MAE:{:.2f} R2:{:.2f} Pearson:{:.2f} Spearman:{:.2f}'.format(
            mse, rmse, mae, r2, pearson, spearman))

    if _ctx_manifold_embeds:
        try:
            _ctx_manifold_embeds = torch.stack(_ctx_manifold_embeds, dim=0).cpu().detach().numpy()
        except RuntimeError:
            _ctx_manifold_embeds = torch.cat(_ctx_manifold_embeds, dim=0).cpu().detach().numpy()

    if _attn_topologies:
        try:
            _attn_topologies = torch.stack(_attn_topologies, dim=0).cpu().detach().numpy()
        except RuntimeError:
            _attn_topologies = torch.cat(_attn_topologies, dim=0).cpu().detach().numpy()

    return _pred_registry, _ctx_manifold_embeds, _attn_topologies