import torch
from torch import nn
import torch.nn.functional as F


class _ManifoldLayerNorm(nn.Module):
    def __init__(self, hidden_size, variance_epsilon=1e-12):
        super(_ManifoldLayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(hidden_size))
        self.beta = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = variance_epsilon

    def forward(self, x_tensor):
        _mean_val = x_tensor.mean(-1, keepdim=True)
        _var_val = (x_tensor - _mean_val).pow(2).mean(-1, keepdim=True)
        _normed_state = (x_tensor - _mean_val) / torch.sqrt(_var_val + self.variance_epsilon)
        return self.gamma * _normed_state + self.beta


class TokenEmbeddingSpace(nn.Module):
    def __init__(self, vocab_size, hidden_size, max_position_size, dropout_rate):
        super(TokenEmbeddingSpace, self).__init__()
        self._semantic_embed = nn.Embedding(vocab_size, hidden_size)
        self._spatial_embed = nn.Embedding(max_position_size, hidden_size)
        self._norm_operator = _ManifoldLayerNorm(hidden_size)
        self._stochastic_drop = nn.Dropout(dropout_rate)

    def forward(self, token_idx):
        _seq_dim = token_idx.size(1)
        _pos_idx = torch.arange(_seq_dim, dtype=torch.long, device=token_idx.device)
        _pos_idx = _pos_idx.unsqueeze(0).expand_as(token_idx)

        _fusion_state = self._semantic_embed(token_idx) + self._spatial_embed(_pos_idx)
        return self._stochastic_drop(self._norm_operator(_fusion_state))


class _CanonicalSelfAttentionOperator(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, attention_probs_dropout_prob):
        super(_CanonicalSelfAttentionOperator, self).__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(f"Dimensionality conflict: {hidden_size} // {num_attention_heads} != 0")

        self._core_attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_attention_heads,
            dropout=attention_probs_dropout_prob,
            batch_first=True
        )

    def forward(self, hidden_states, attention_mask=None):
        _pad_mask = None
        if attention_mask is not None:
            if attention_mask.dim() == 4:
                attention_mask = attention_mask.squeeze(1).squeeze(1)
            elif attention_mask.dim() == 3:
                attention_mask = attention_mask.squeeze(1)

            if attention_mask.dtype == torch.float:
                _pad_mask = attention_mask == -10000.0
            else:
                _pad_mask = attention_mask

        return self._core_attn(
            query=hidden_states, key=hidden_states, value=hidden_states,
            key_padding_mask=_pad_mask, need_weights=True
        )


class _CanonicalCrossAttentionOperator(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, attention_probs_dropout_prob):
        super(_CanonicalCrossAttentionOperator, self).__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(f"Dimensionality conflict: {hidden_size} // {num_attention_heads} != 0")

        self._core_attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_attention_heads,
            dropout=attention_probs_dropout_prob,
            batch_first=True
        )

    def forward(self, q_tensor, kv_tensor, query_mask=None, kv_mask=None):
        _pad_mask = None
        if query_mask is not None:
            if query_mask.dim() == 4:
                query_mask = query_mask.squeeze(1).squeeze(1)
            elif query_mask.dim() == 3:
                query_mask = query_mask.squeeze(1)

            if query_mask.dtype == torch.float:
                _pad_mask = query_mask == -10000.0
            else:
                _pad_mask = query_mask

        return self._core_attn(
            query=q_tensor, key=kv_tensor, value=kv_tensor,
            key_padding_mask=_pad_mask, need_weights=True
        )


class _AttentionProjectionSpace(nn.Module):
    def __init__(self, hidden_size, hidden_dropout_prob):
        super(_AttentionProjectionSpace, self).__init__()
        self._proj = nn.Linear(hidden_size, hidden_size)
        self._norm = _ManifoldLayerNorm(hidden_size)
        self._drop = nn.Dropout(hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        _projected = self._drop(self._proj(hidden_states))
        return self._norm(_projected + input_tensor)


class _MultiHeadAttentionOperator(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, attention_probs_dropout_prob, hidden_dropout_prob):
        super(_MultiHeadAttentionOperator, self).__init__()
        self._sa_node = _CanonicalSelfAttentionOperator(hidden_size, num_attention_heads, attention_probs_dropout_prob)
        self._out_node = _AttentionProjectionSpace(hidden_size, hidden_dropout_prob)

    def forward(self, input_tensor, attention_mask):
        _sa_state, _attn_map = self._sa_node(input_tensor, attention_mask)
        return self._out_node(_sa_state, input_tensor), _attn_map


class _CrossModalAttentionOperator(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, attention_probs_dropout_prob, hidden_dropout_prob):
        super(_CrossModalAttentionOperator, self).__init__()
        self._ca_node = _CanonicalCrossAttentionOperator(hidden_size, num_attention_heads, attention_probs_dropout_prob)
        self._out_node = _AttentionProjectionSpace(hidden_size, hidden_dropout_prob)

    def forward(self, entity_a, entity_b, mask_a, mask_b):
        _ca_state_a, _map_a = self._ca_node(entity_a, entity_b, mask_a, mask_b)
        _final_a = self._out_node(_ca_state_a, entity_a)

        _ca_state_b, _map_b = self._ca_node(entity_b, entity_a, mask_b, mask_a)
        _final_b = self._out_node(_ca_state_b, entity_b)

        return _final_a, _final_b, _map_a, _map_b


class _DirectedAttentionOperator(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, attention_probs_dropout_prob, hidden_dropout_prob):
        super(_DirectedAttentionOperator, self).__init__()
        self._ca_node = _CanonicalCrossAttentionOperator(hidden_size, num_attention_heads, attention_probs_dropout_prob)
        self._out_node = _AttentionProjectionSpace(hidden_size, hidden_dropout_prob)

    def forward(self, src_entity, tgt_entity, src_mask):
        _ca_state, _map = self._ca_node(src_entity, tgt_entity, src_mask, None)
        return self._out_node(_ca_state, src_entity), _map


class _LatentExpansionPhase(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super(_LatentExpansionPhase, self).__init__()
        self._expansion_proj = nn.Linear(hidden_size, intermediate_size)

    def forward(self, hidden_states):
        return F.relu(self._expansion_proj(hidden_states))


class _LatentProjectionPhase(nn.Module):
    def __init__(self, intermediate_size, hidden_size, hidden_dropout_prob):
        super(_LatentProjectionPhase, self).__init__()
        self._collapse_proj = nn.Linear(intermediate_size, hidden_size)
        self._norm = _ManifoldLayerNorm(hidden_size)
        self._drop = nn.Dropout(hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        _collapsed = self._drop(self._collapse_proj(hidden_states))
        return self._norm(_collapsed + input_tensor)


class TMEAttentionalBlock(nn.Module):
    def __init__(self, hidden_size, intermediate_size, num_attention_heads, attention_probs_dropout_prob,
                 hidden_dropout_prob):
        super(TMEAttentionalBlock, self).__init__()
        self._attention_phase = _MultiHeadAttentionOperator(hidden_size, num_attention_heads,
                                                            attention_probs_dropout_prob, hidden_dropout_prob)
        self._expansion_phase = _LatentExpansionPhase(hidden_size, intermediate_size)
        self._projection_phase = _LatentProjectionPhase(intermediate_size, hidden_size, hidden_dropout_prob)

    def forward(self, hidden_states, attention_mask):
        _attn_out, _attn_prob = self._attention_phase(hidden_states, attention_mask)
        _expanded_state = self._expansion_phase(_attn_out)
        return self._projection_phase(_expanded_state, _attn_out), _attn_prob


class EncoderCell(nn.Module):
    def __init__(self, hidden_size, intermediate_size, num_attention_heads, attention_probs_dropout_prob,
                 hidden_dropout_prob):
        super(EncoderCell, self).__init__()
        self._norm_layer = _ManifoldLayerNorm(hidden_size)
        self._attention_phase = _MultiHeadAttentionOperator(hidden_size, num_attention_heads,
                                                            attention_probs_dropout_prob, hidden_dropout_prob)
        self._ffn_bottleneck = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size),
            nn.ReLU(),
            nn.Dropout(hidden_dropout_prob),
            nn.Linear(intermediate_size, hidden_size)
        )

    def forward(self, hidden_states, attention_mask):
        _normed_in = self._norm_layer(hidden_states)
        _attn_out, _attn_prob = self._attention_phase(_normed_in, attention_mask)
        _residual_state = _normed_in + _attn_out

        _ffn_out = self._ffn_bottleneck(self._norm_layer(_residual_state))
        return _residual_state + _ffn_out, _attn_prob


class EncoderCA(nn.Module):
    def __init__(self, hidden_size, intermediate_size, num_attention_heads, attention_probs_dropout_prob,
                 hidden_dropout_prob):
        super(EncoderCA, self).__init__()
        self._cross_phase = _CrossModalAttentionOperator(hidden_size, num_attention_heads, attention_probs_dropout_prob,
                                                         hidden_dropout_prob)
        self._expansion = _LatentExpansionPhase(hidden_size, intermediate_size)
        self._projection = _LatentProjectionPhase(intermediate_size, hidden_size, hidden_dropout_prob)

    def forward(self, entity_a, entity_b, mask_a, mask_b):
        _ca_out_a, _ca_out_b, _prob_a, _prob_b = self._cross_phase(entity_a, entity_b, mask_a, mask_b)

        _layer_out_a = self._projection(self._expansion(_ca_out_a), _ca_out_a)
        _layer_out_b = self._projection(self._expansion(_ca_out_b), _ca_out_b)

        return _layer_out_a, _layer_out_b, _prob_a, _prob_b


class EncoderCellCA(nn.Module):
    def __init__(self, hidden_size, intermediate_size, num_attention_heads, attention_probs_dropout_prob,
                 hidden_dropout_prob):
        super(EncoderCellCA, self).__init__()
        self._norm = _ManifoldLayerNorm(hidden_size)
        self._cross_phase = _CrossModalAttentionOperator(hidden_size, num_attention_heads, attention_probs_dropout_prob,
                                                         hidden_dropout_prob)
        self._ffn_bottleneck = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size),
            nn.ReLU(),
            nn.Dropout(hidden_dropout_prob),
            nn.Linear(intermediate_size, hidden_size)
        )

    def forward(self, cell_a, cell_b, mask_a=None, mask_b=None):
        _norm_a, _norm_b = self._norm(cell_a), self._norm(cell_b)
        _ca_a, _ca_b, _prob_a, _prob_b = self._cross_phase(cell_a, cell_b, mask_a, mask_b)

        _res_a = _norm_a + _ca_a
        _out_a = _res_a + self._ffn_bottleneck(self._norm(_res_a))

        _res_b = _norm_b + _ca_b
        _out_b = _res_b + self._ffn_bottleneck(self._norm(_res_b))

        return _out_a, _out_b, _prob_a, _prob_b


class EncoderD2C(nn.Module):
    def __init__(self, hidden_size, intermediate_size, num_attention_heads, attention_probs_dropout_prob,
                 hidden_dropout_prob):
        super(EncoderD2C, self).__init__()
        self._norm = _ManifoldLayerNorm(hidden_size)
        self._cross_phase = _CrossModalAttentionOperator(hidden_size, num_attention_heads, attention_probs_dropout_prob,
                                                         hidden_dropout_prob)
        self._expansion = _LatentExpansionPhase(hidden_size, intermediate_size)
        self._projection = _LatentProjectionPhase(intermediate_size, hidden_size, hidden_dropout_prob)
        self._ffn_bottleneck = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size),
            nn.ReLU(),
            nn.Dropout(hidden_dropout_prob),
            nn.Linear(intermediate_size, hidden_size)
        )

    def forward(self, cell_entity, mol_entity, mol_mask, cell_mask=None):
        _norm_cell = self._norm(cell_entity)
        _ca_cell, _ca_mol, _prob_cell, _prob_mol = self._cross_phase(_norm_cell, mol_entity, cell_mask, mol_mask)

        _res_cell = _norm_cell + _ca_cell
        _cell_layer_out = _res_cell + self._ffn_bottleneck(self._norm(_res_cell))

        _mol_layer_out = self._projection(self._expansion(_ca_mol), _ca_mol)

        return _cell_layer_out, _mol_layer_out, _prob_cell, _prob_mol


class EncoderSSA(nn.Module):
    def __init__(self, hidden_size, intermediate_size, num_attention_heads, attention_probs_dropout_prob,
                 hidden_dropout_prob):
        super(EncoderSSA, self).__init__()
        self._directed_phase = _DirectedAttentionOperator(hidden_size, num_attention_heads,
                                                          attention_probs_dropout_prob, hidden_dropout_prob)
        self._expansion = _LatentExpansionPhase(hidden_size, intermediate_size)
        self._projection = _LatentProjectionPhase(intermediate_size, hidden_size, hidden_dropout_prob)

    def forward(self, src_entity, tgt_entity, src_mask):
        _dir_out, _prob = self._directed_phase(src_entity, tgt_entity, src_mask)
        _layer_out = self._projection(self._expansion(_dir_out), _dir_out)
        return _layer_out, _prob


class EncoderCellSSA(nn.Module):
    def __init__(self, hidden_size, intermediate_size, num_attention_heads, attention_probs_dropout_prob,
                 hidden_dropout_prob):
        super(EncoderCellSSA, self).__init__()
        self._norm = _ManifoldLayerNorm(hidden_size)
        self._directed_phase = _DirectedAttentionOperator(hidden_size, num_attention_heads,
                                                          attention_probs_dropout_prob, hidden_dropout_prob)
        self._ffn_bottleneck = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size),
            nn.ReLU(),
            nn.Dropout(hidden_dropout_prob),
            nn.Linear(intermediate_size, hidden_size)
        )

    def forward(self, src_entity, tgt_entity, src_mask=None):
        _norm_src, _norm_tgt = self._norm(src_entity), self._norm(tgt_entity)
        _dir_out, _prob = self._directed_phase(src_entity, tgt_entity, src_mask)

        _res_src = _norm_src + _dir_out
        _layer_out = _res_src + self._ffn_bottleneck(self._norm(_res_src))

        return _layer_out, _prob