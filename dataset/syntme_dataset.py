import os
import os.path as osp
import numpy as np
import torch
from torch_geometric.data import Data
from tqdm import tqdm
from .base_InMemory_dataset import BaseInMemoryDataset


class SynTMETopologyDataset(BaseInMemoryDataset):
    def __init__(self,
                 data_root,
                 data_items,
                 celllines_data,
                 drugs_data,
                 celltme_data,
                 dgi_data=None,
                 transform=None,
                 pre_transform=None,
                 args=None,
                 max_node_num=155):

        super(SynTMETopologyDataset, self).__init__(root=data_root, transform=transform, pre_transform=pre_transform)

        _suffix = ''
        if args.celldataset == 1:
            _suffix = '18498g'
        elif args.celldataset == 2:
            _suffix = '4079g'
        elif args.celldataset == 3:
            _suffix = '963g'

        self.name = f"syntme_{osp.basename(data_items).split('items')[0]}{_suffix}_manifold"

        if args.mode == 'infer':
            self.name = osp.basename(data_items).split('items')[0]

        self.args = args
        self._registry = np.load(data_items, allow_pickle=True)
        self._ctx_bank = np.load(celllines_data, allow_pickle=True).item()
        self._mol_bank = np.load(drugs_data, allow_pickle=True).item()
        self._microenv_bank = np.load(celltme_data, allow_pickle=True).item()

        self._aux_graph = np.load(dgi_data, allow_pickle=True).item() if dgi_data else {}
        self.max_node_num = max_node_num

        if os.path.isfile(self.processed_paths[0]):
            self.data, self.slices = torch.load(self.processed_paths[0])
        else:
            self.process()
            self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return [self.name + '.pt']

    def download(self):
        pass

    def _download(self):
        pass

    def _process(self):
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

    def process(self):
        _graph_list = []
        _registry_len = len(self._registry)

        for i in tqdm(range(_registry_len), desc="Constructing SynTME Manifold"):
            m_a, m_b, ctx_idx, target_val = self._registry[i]

            _ctx_vector = self._ctx_bank[ctx_idx]
            _microenv_vector = self._microenv_bank[ctx_idx]

            _aux_a = self._aux_graph.get(m_a, np.ones(_ctx_vector.shape[0]))
            _aux_b = self._aux_graph.get(m_b, np.ones(_ctx_vector.shape[0]))

            _mol_a_feat = self._mol_bank[m_a]
            _mol_b_feat = self._mol_bank[m_b]

            _manifold_obj = Data()
            _manifold_obj.drugA = torch.Tensor(np.array([_mol_a_feat])).to(dtype=torch.float16)
            _manifold_obj.drugB = torch.Tensor(np.array([_mol_b_feat])).to(dtype=torch.float16)
            _manifold_obj.x_cell = torch.as_tensor(_ctx_vector).to(dtype=torch.float16)
            _manifold_obj.celltem = torch.as_tensor(_microenv_vector).to(dtype=torch.float16)
            _manifold_obj.y = torch.Tensor([float(target_val)]).to(dtype=torch.float16)

            _manifold_obj.dgiA = torch.Tensor(_aux_a).to(dtype=torch.float16)
            _manifold_obj.dgiB = torch.Tensor(_aux_b).to(dtype=torch.float16)

            _graph_list.append(_manifold_obj)

        if self.pre_filter is not None:
            _graph_list = [g for g in _graph_list if self.pre_filter(g)]

        if self.pre_transform is not None:
            _graph_list = [self.pre_transform(g) for g in _graph_list]

        _data, _slices = self.collate(_graph_list)
        torch.save((_data, _slices), self.processed_paths[0])