import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .head import TMESynergyPredictor
from .model_utlis import TokenEmbeddingSpace, TMEAttentionalBlock, EncoderCA, EncoderCell, EncoderCellCA, EncoderD2C, \
    EncoderSSA, EncoderCellSSA


class OmicsFeatureExtractor(nn.Module):
    def __init__(self, in_channel=3, feat_dim=None, args=None):
        super(OmicsFeatureExtractor, self).__init__()

        _pool_cfg = [2, 2, 6]
        _drop = 0.2
        _k_size = [16, 16, 16]

        if in_channel == 3:
            _in_dim = [3, 8, 16]
            _out_dim = [8, 16, 32]
        elif in_channel == 1:
            _in_dim = [1, 8, 16]
            _out_dim = [8, 16, 32]
        elif in_channel == 6:
            _in_dim = [6, 16, 32]
            _out_dim = [16, 32, 64]

        self._ctx_stream = nn.Sequential(
            nn.Conv1d(in_channels=_in_dim[0], out_channels=_out_dim[0], kernel_size=_k_size[0]),
            nn.ReLU(),
            nn.Dropout(p=_drop),
            nn.MaxPool1d(_pool_cfg[0]),
            nn.Conv1d(in_channels=_in_dim[1], out_channels=_out_dim[1], kernel_size=_k_size[1]),
            nn.ReLU(),
            nn.Dropout(p=_drop),
            nn.MaxPool1d(_pool_cfg[1]),
            nn.Conv1d(in_channels=_in_dim[2], out_channels=_out_dim[2], kernel_size=_k_size[2]),
            nn.ReLU(),
            nn.MaxPool1d(_pool_cfg[2]),
        )
        self._proj_layer = nn.Linear(_out_dim[2], feat_dim)

    def forward(self, x_tensor):
        x_tensor = x_tensor.transpose(1, 2)
        _encoded_stream = self._ctx_stream(x_tensor)
        _encoded_stream = _encoded_stream.transpose(1, 2)
        return self._proj_layer(_encoded_stream)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv1d)):
                nn.init.xavier_normal_(m.weight)


def _align_molecular_tokens(raw_codes, dev, p_size, seq_limit):
    _idx_tensor = raw_codes[:, 0].to(torch.long).to(dev)
    _m_tensor = raw_codes[:, 1].to(torch.long).to(dev)

    if p_size > seq_limit:
        _pad_matrix = torch.zeros((_idx_tensor.size(0), p_size - seq_limit), dtype=torch.long, device=dev)
        _idx_tensor = torch.cat((_idx_tensor, _pad_matrix), dim=1)
        _m_tensor = torch.cat((_m_tensor, _pad_matrix), dim=1)

    _attn_penalty = (1.0 - _m_tensor.view(-1, 1, 1, _m_tensor.size(-1))) * -1e4
    return _idx_tensor, _attn_penalty.to(torch.float32)


class SynTMEFramework(torch.nn.Module):
    def __init__(self,
                 num_attention_heads=8,
                 attention_probs_dropout_prob=0.1,
                 hidden_dropout_prob=0.1,
                 max_length=50,
                 input_dim_drug=2586,
                 output_dim=2560,
                 hidden_channels=128,
                 num_mfic_layers=2,
                 args=None):
        super(SynTMEFramework, self).__init__()

        self.args = args
        self.include_omic = args.omic.split(',')
        self._omic_mapping = {'exp': 0, 'mut': 1, 'cn': 2, 'eff': 3, 'dep': 4, 'met': 5}
        self.in_channel = len(self.include_omic)
        self._seq_limit = max_length
        self.num_layer = 1

        if args.celldataset == 0:
            self._g_dim = 697
        elif args.celldataset == 1:
            self._g_dim = 18498
        elif args.celldataset == 2:
            self._g_dim = 4079

        if self.args.cellencoder == 'cellTrans':
            self._p_size = 50
            if self.in_channel == 3:
                _f_dim, _h_size = 243, 256
            elif self.in_channel == 6:
                _f_dim, _h_size = 486, 512
            self._ctx_proj = nn.Linear(_f_dim, _h_size)

        elif self.args.cellencoder == 'cellCNNTrans':
            self._p_size = 165
            _h_size = 128 if self.in_channel == 6 else 64
            self._ctx_extractor = OmicsFeatureExtractor(in_channel=self.in_channel, feat_dim=_h_size, args=args)

        self._tme_scale_proj = nn.Linear(22, _h_size, bias=False)
        self._tme_shift_proj = nn.Linear(22, _h_size, bias=False)

        self._microenv_ffn = nn.Sequential(
            nn.Linear(150, 300),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(300, _h_size),
            nn.Dropout(0.2)
        )
        self._microenv_norm = nn.LayerNorm(_h_size)

        _inter_size = _h_size * 2
        self.mol_embed_space = TokenEmbeddingSpace(input_dim_drug, _h_size, self._p_size, hidden_dropout_prob)

        self.mol_sa_block = TMEAttentionalBlock(_h_size, _inter_size, num_attention_heads, attention_probs_dropout_prob,
                                                hidden_dropout_prob)
        self.ctx_sa_block = EncoderCell(_h_size, _inter_size, num_attention_heads, attention_probs_dropout_prob,
                                        hidden_dropout_prob)
        self.cross_modal_block = EncoderD2C(_h_size, _inter_size, num_attention_heads, attention_probs_dropout_prob,
                                            hidden_dropout_prob)

        self.predictor_head = TMESynergyPredictor()

        self._ctx_bottleneck = nn.Sequential(
            nn.Linear(self._p_size * _h_size, output_dim),
            nn.ReLU(),
            nn.Dropout(p=0.2),
        )
        self._mol_bottleneck = nn.Sequential(
            nn.Linear(self._p_size * _h_size, output_dim),
            nn.ReLU(),
            nn.Dropout(p=0.2),
        )

    def forward(self, payload):
        b_dim = payload.x_cell.size(0) // self._g_dim

        mol_alpha_raw, mol_beta_raw = payload.drugA, payload.drugB
        mol_alpha, alpha_mask = _align_molecular_tokens(mol_alpha_raw, self.args.device, self._p_size, self._seq_limit)
        mol_beta, beta_mask = _align_molecular_tokens(mol_beta_raw, self.args.device, self._p_size, self._seq_limit)

        mol_alpha = self.mol_embed_space(mol_alpha).to(torch.float32)
        mol_beta = self.mol_embed_space(mol_beta).to(torch.float32)

        x_ctx = payload.x_cell.to(torch.float32)
        _active_omics = [self._omic_mapping[i] for i in self.include_omic]
        x_ctx = x_ctx[:, _active_omics]

        ctx_matrix_a = x_ctx.view(b_dim, self._g_dim, -1)
        ctx_matrix_b = ctx_matrix_a.clone()

        microenv_state = payload.celltem.to(torch.float32).view(b_dim, -1)
        microenv_expanded = microenv_state.unsqueeze(1).expand(-1, 165, -1)

        if self.args.cellencoder == 'cellTrans':
            _g_len = 4050
            ctx_alpha = self._ctx_proj(ctx_matrix_a[:, :_g_len, :].view(b_dim, self._p_size, -1))
            ctx_beta = self._ctx_proj(ctx_matrix_b[:, :_g_len, :].view(b_dim, self._p_size, -1))
        elif self.args.cellencoder == 'cellCNNTrans':
            ctx_alpha = self._ctx_extractor(ctx_matrix_a)
            ctx_beta = self._ctx_extractor(ctx_matrix_b)

        _fusion_alpha = torch.cat((ctx_alpha, microenv_expanded), dim=-1)
        _fusion_beta = torch.cat((ctx_beta, microenv_expanded), dim=-1)

        ctx_alpha_tme = self._microenv_norm(self._microenv_ffn(_fusion_alpha).add(ctx_alpha))
        ctx_beta_tme = self._microenv_norm(self._microenv_ffn(_fusion_beta).add(ctx_beta))

        ctx_alpha, attn_a_sa = self.ctx_sa_block(ctx_alpha_tme, None)
        ctx_beta, attn_b_sa = self.ctx_sa_block(ctx_beta_tme, None)

        ctx_alpha_cache = ctx_alpha

        ctx_alpha, mol_alpha, attn_c2m_a, attn_m2c_a = self.cross_modal_block(ctx_alpha, mol_alpha, alpha_mask, None)
        ctx_beta, mol_beta, attn_c2m_b, attn_m2c_b = self.cross_modal_block(ctx_beta, mol_beta, beta_mask, None)

        ctx_alpha_state = ctx_alpha
        ctx_beta_state = ctx_beta

        _scale_factor = self._tme_scale_proj(microenv_expanded[:, 0, :]).unsqueeze(1)
        _shift_factor = self._tme_shift_proj(microenv_expanded[:, 0, :]).unsqueeze(1)

        ctx_alpha = ctx_alpha.mul(1.0 + _scale_factor).add(_shift_factor)
        ctx_beta = ctx_beta.mul(1.0 + _scale_factor).add(_shift_factor)

        mol_alpha, attn_mol_a = self.mol_sa_block(mol_alpha, alpha_mask)
        mol_beta, attn_mol_b = self.mol_sa_block(mol_beta, beta_mask)

        mol_alpha_repr = self._mol_bottleneck(mol_alpha.view(-1, mol_alpha.size(1) * mol_alpha.size(2)))
        mol_beta_repr = self._mol_bottleneck(mol_beta.view(-1, mol_beta.size(1) * mol_beta.size(2)))
        ctx_alpha_repr = self._ctx_bottleneck(ctx_alpha.view(-1, ctx_alpha.size(1) * ctx_alpha.size(2)))
        ctx_beta_repr = self._ctx_bottleneck(ctx_beta.view(-1, ctx_beta.size(1) * ctx_beta.size(2)))

        _latent_fusion = torch.cat((ctx_alpha_repr, ctx_beta_repr, mol_alpha_repr, mol_beta_repr), dim=1)
        synergy_score = self.predictor_head(_latent_fusion)

        _embed_registry = None
        _attn_registry = None

        if self.args.output_attn:
            _embed_registry = torch.stack((
                ctx_alpha_state.mean(dim=1).flatten(),
                ctx_beta_state.mean(dim=1).flatten(),
                ctx_alpha.mean(dim=1).flatten(),
                ctx_beta.mean(dim=1).flatten(),
                ctx_alpha_cache.mean(dim=1).flatten()
            ), dim=0)
            _attn_registry = torch.cat(
                (attn_c2m_a, attn_m2c_a, attn_c2m_b, attn_m2c_b, attn_mol_a, attn_mol_b, attn_a_sa, attn_b_sa), dim=0)

        return synergy_score, _embed_registry, _attn_registry

    def init_weights(self):
        self.predictor_head.init_weights()
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
        if self.args.cellencoder == 'cellCNNTrans':
            self._ctx_extractor.init_weights()