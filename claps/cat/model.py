"""
Cross-modal Aligned Transformer (CAT): cognitive symptoms + time series -> PEPQ-R.
- Scenario path (flat): whole-session time series -> shared Transformer -> global embedding
- Scene path (hierarchical): per-scene time series -> shared Transformer -> Cross-Scene Transformer
- Text: the per-scene cognitive symptom (cognitive symptoms of anxiety) from the Generator LLM
- Output: 9 PEPQ-R item scores
"""

import os
import torch
from torch import nn
from claps.cat.layers.Transformer_EncDec import Encoder, EncoderLayer
from claps.cat.layers.SelfAttention_Family import FullAttention, AttentionLayer
from claps.cat.layers.Embed import PatchEmbedding
from transformers import AutoModel

# Text encoder for cognitive symptoms (KURE-v1, fine-tuned on BGE-M3). Swap for any
# Hugging Face encoder id if needed; hidden size is read from its config.
TEXT_ENCODER = 'nlpai-lab/KURE-v1'


class _PaddingMask:
    def __init__(self, m):
        self._mask = m

    @property
    def mask(self):
        return self._mask


class DualPathEncoder(nn.Module):
    def __init__(self, configs, device):
        super(DualPathEncoder, self).__init__()
        self.seq_len = configs.seq_len
        self.enc_in = configs.enc_in

        patch_len = configs.patch_len
        stride = configs.stride
        self.stride = stride
        padding = stride

        self.device = device
        self.d_model = configs.d_model
        self.ablation = getattr(configs, 'ablation', 'none')
        d_scene = configs.d_scene
        self.d_scene = d_scene

        # ============================================================
        # Shared Encoder (both paths)
        # ============================================================
        self.patch_embedding = PatchEmbedding(configs.d_model, patch_len, stride, padding, configs.dropout)

        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, attention_dropout=configs.dropout,
                                      output_attention=False), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for _ in range(configs.e_layers)
            ],
            norm_layer=nn.LayerNorm(configs.d_model)
        )

        self.dropout = nn.Dropout(configs.dropout)
        self.num_items = 9  # PEPQ-R

        proj_input_dim = configs.enc_in * configs.d_model

        # ============================================================
        # Flat path
        # ============================================================
        self.flat_projection = nn.Sequential(
            nn.Linear(proj_input_dim, d_scene),
            nn.ReLU(),
            nn.Dropout(configs.dropout),
        )

        # ============================================================
        # Scene path + cross-scene transformer
        # ============================================================
        self.scene_projection = nn.Sequential(
            nn.Linear(proj_input_dim, d_scene),
            nn.ReLU(),
            nn.Dropout(configs.dropout),
        )

        self.scene_pos_embedding = nn.Parameter(torch.randn(1, 10, d_scene) * 0.02)

        self.cross_scene_encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, attention_dropout=configs.dropout,
                                      output_attention=False), d_scene, configs.cross_scene_heads),
                    d_scene,
                    d_scene * 2,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for _ in range(configs.cross_scene_layers)
            ],
            norm_layer=nn.LayerNorm(d_scene)
        )

        # ============================================================
        # Prediction heads
        # ============================================================
        self.dual_head = nn.Sequential(
            nn.Linear(d_scene * 2, d_scene),
            nn.ReLU(),
            nn.Dropout(configs.dropout),
            nn.Linear(d_scene, self.num_items),
        )

        self.single_head = nn.Sequential(
            nn.Linear(d_scene, self.num_items),
        )

        # ============================================================
        # Text encoder for the cognitive symptom text: KURE-v1
        # (fine-tuned on BGE-M3), as used in the paper.
        # ============================================================
        os.environ['TOKENIZERS_PARALLELISM'] = 'True'
        self.lm_model = AutoModel.from_pretrained(TEXT_ENCODER)
        # Freeze the text encoder (keeps it out of the optimizer; saves AdamW state memory)
        for p in self.lm_model.parameters():
            p.requires_grad = False

        self.layer_text = nn.Linear(self.lm_model.config.hidden_size, configs.d_model)

    def _encode_patches(self, x_time, real_lens, text_emb_per_unit, bs_units):
        x_time = x_time.permute(0, 2, 1)
        enc_out, n_vars = self.patch_embedding(x_time)
        num_time_patches = enc_out.shape[1]

        text_expanded = text_emb_per_unit.unsqueeze(1).repeat(1, n_vars, 1)
        text_expanded = text_expanded.view(bs_units * n_vars, 1, self.d_model)
        enc_out = torch.cat((enc_out, text_expanded), dim=1)

        total_len = enc_out.shape[1]

        attn_mask = None
        pool_mask = None
        if real_lens is not None:
            real_lens_dev = real_lens.to(self.device)
            num_real_patches = torch.ceil(real_lens_dev.float() / self.stride).long()
            num_real_patches = torch.clamp(num_real_patches, max=num_time_patches)

            mask_2d = torch.zeros(bs_units, total_len, dtype=torch.bool, device=self.device)
            time_pos = torch.arange(num_time_patches, device=self.device)
            mask_2d[:, :num_time_patches] = time_pos.unsqueeze(0) >= num_real_patches.unsqueeze(1)

            mask_expanded = mask_2d.unsqueeze(1).expand(-1, n_vars, -1).reshape(bs_units * n_vars, total_len)
            attn_mask = _PaddingMask(mask_expanded.unsqueeze(1).unsqueeze(1))
            pool_mask = (~mask_2d).float()

        enc_out, _ = self.encoder(enc_out, attn_mask=attn_mask)
        enc_out = enc_out.view(bs_units, n_vars, total_len, self.d_model)
        enc_out = enc_out.permute(0, 1, 3, 2)

        if pool_mask is not None:
            pm = pool_mask.unsqueeze(1).unsqueeze(1)
            enc_out = (enc_out * pm).sum(dim=-1) / pm.sum(dim=-1).clamp(min=1)
        else:
            enc_out = enc_out.mean(dim=-1)
        enc_out = self.dropout(enc_out)

        return enc_out.reshape(bs_units, -1)

    def forward(self, x_enc_time_flat, x_enc_time_scene, x_enc_text,
                flat_real_len=None, scene_real_lens=None, num_scenes_ts=None):
        bs = x_enc_time_flat.shape[0]
        max_scenes = x_enc_time_scene.shape[1]
        max_scene_len = x_enc_time_scene.shape[2]

        if num_scenes_ts is not None:
            num_scenes_ts = num_scenes_ts.to(self.device)

        # ============================================================
        # Encode the cognitive symptom text
        # ============================================================
        tok_len = x_enc_text['input_ids'].shape[2]
        num_scenes_text = x_enc_text['num_scenes'].to(self.device)

        flat_ids = x_enc_text['input_ids'].view(-1, tok_len).to(self.device)
        flat_mask = x_enc_text['attention_mask'].view(-1, tok_len).to(self.device)

        with torch.no_grad():
            outputs = self.lm_model(
                input_ids=flat_ids,
                attention_mask=flat_mask,
                output_hidden_states=True
            )

        cls_embs = outputs['hidden_states'][-1][:, 0, :]
        cls_embs = cls_embs.view(bs, max_scenes, -1)
        scene_text_embs = self.layer_text(cls_embs)
        scene_text_embs = self.dropout(scene_text_embs)

        # Text embedding for the flat (scenario) path:
        # - long-context models (flat_input_ids present): whole-scenario concat -> single CLS
        # - otherwise: average per-scene CLS over the real number of scenes (lossy)
        if 'flat_input_ids' in x_enc_text:
            flat_full_ids = x_enc_text['flat_input_ids'].to(self.device)
            flat_full_mask = x_enc_text['flat_attention_mask'].to(self.device)
            with torch.no_grad():
                flat_outputs = self.lm_model(
                    input_ids=flat_full_ids,
                    attention_mask=flat_full_mask,
                    output_hidden_states=True,
                )
            flat_cls = flat_outputs['hidden_states'][-1][:, 0, :]
            flat_text_emb = self.layer_text(flat_cls)
            flat_text_emb = self.dropout(flat_text_emb)
        else:
            scene_mask = torch.arange(max_scenes, device=self.device).unsqueeze(0) < num_scenes_text.unsqueeze(1)
            scene_mask_f = scene_mask.unsqueeze(2).float()
            flat_text_emb = (scene_text_embs * scene_mask_f).sum(dim=1) / num_scenes_text.unsqueeze(1).float()

        # ============================================================
        # Session-level normalization
        # ============================================================
        x_flat = x_enc_time_flat.float().to(self.device)
        means_flat = x_flat.mean(1, keepdim=True).detach()
        stdev_flat = torch.sqrt(torch.var(x_flat, dim=1, keepdim=True, unbiased=False) + 1e-5)

        # ============================================================
        # Path 1: Flat
        # ============================================================
        flat_emb = None
        if self.ablation not in ('path2_only', 'path2_no_cross_scene'):
            x_flat_normed = (x_flat - means_flat) / stdev_flat
            flat_emb_raw = self._encode_patches(x_flat_normed, flat_real_len, flat_text_emb, bs)
            flat_emb = self.flat_projection(flat_emb_raw)

        # ============================================================
        # Path 2: Scene (hierarchical)
        # ============================================================
        session_emb = None
        if self.ablation != 'path1_only':
            x_scene = x_enc_time_scene.float().to(self.device)
            scene_means = x_scene.mean(dim=2, keepdim=True).detach()
            scene_stdev = torch.sqrt(x_scene.var(dim=2, keepdim=True, unbiased=False) + 1e-5).detach()
            x_scene = (x_scene - scene_means) / scene_stdev

            x_scene = x_scene.view(bs * max_scenes, max_scene_len, -1)
            scene_text_flat = scene_text_embs.view(bs * max_scenes, -1)
            scene_rl = scene_real_lens.view(-1) if scene_real_lens is not None else None

            scene_emb_raw = self._encode_patches(x_scene, scene_rl, scene_text_flat, bs * max_scenes)
            scene_embs = self.scene_projection(scene_emb_raw)
            scene_embs = scene_embs.view(bs, max_scenes, self.d_scene)

            if self.ablation in ('no_cross_scene', 'path2_no_cross_scene'):
                if num_scenes_ts is not None:
                    scene_pos = torch.arange(max_scenes, device=self.device)
                    scene_mask_2d = scene_pos.unsqueeze(0) >= num_scenes_ts.unsqueeze(1)
                    scene_valid = (~scene_mask_2d).float().unsqueeze(2)
                    session_emb = (scene_embs * scene_valid).sum(dim=1) / scene_valid.sum(dim=1).clamp(min=1)
                else:
                    session_emb = scene_embs.mean(dim=1)
            else:
                scene_embs = scene_embs + self.scene_pos_embedding[:, :max_scenes, :]

                cross_attn_mask = None
                if num_scenes_ts is not None:
                    scene_pos = torch.arange(max_scenes, device=self.device)
                    scene_mask_2d = scene_pos.unsqueeze(0) >= num_scenes_ts.unsqueeze(1)
                    cross_attn_mask = _PaddingMask(scene_mask_2d.unsqueeze(1).unsqueeze(1))

                cross_out, _ = self.cross_scene_encoder(scene_embs, attn_mask=cross_attn_mask)

                if num_scenes_ts is not None:
                    scene_valid = (~scene_mask_2d).float().unsqueeze(2)
                    session_emb = (cross_out * scene_valid).sum(dim=1) / scene_valid.sum(dim=1).clamp(min=1)
                else:
                    session_emb = cross_out.mean(dim=1)

        # ============================================================
        # Combine
        # ============================================================
        if self.ablation == 'path1_only':
            output = self.single_head(flat_emb)
            return output, flat_emb
        elif self.ablation in ('path2_only', 'path2_no_cross_scene'):
            output = self.single_head(session_emb)
            return output, session_emb
        else:
            combined = torch.cat([flat_emb, session_emb], dim=1)
            output = self.dual_head(combined)
            return output, combined
