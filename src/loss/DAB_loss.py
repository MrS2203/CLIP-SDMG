import math

import torch
from torch import nn

from .common import (
    DAB_LAYERS,
    TEACHER_TEXT_DIM,
    TEACHER_VISION_DIM,
    MultiLayerLinearProjection,
    extract_valid_tokens,
    get_student_dims,
)

# 视觉/文本模仿损失权重
DAB_ALPHA_VISION = 0.00003
DAB_ALPHA_TEXT = 0.0003
# 注意力阈值系数（均值 + alpha * 标准差）
DAB_MASK_ALPHA = 0.5


class DABLoss(nn.Module):
    def __init__(self, args):
        super(DABLoss, self).__init__()
        self.model_name = args.model

        # ---- 超参集中化：优先用 args 里的值，否则 fallback 到 common.py 默认 ----
        self.dab_layers = getattr(args, 'dab_layers', DAB_LAYERS)
        self.mask_alpha = getattr(args, 'dab_mask_alpha', DAB_MASK_ALPHA)

        dims = get_student_dims(args.model)
        vision_dim, text_dim = dims if dims is not None else (None, None)

        # 视觉/文本维度对齐
        self.v_proj = MultiLayerLinearProjection(vision_dim, TEACHER_VISION_DIM, len(self.dab_layers))
        self.t_proj = MultiLayerLinearProjection(text_dim, TEACHER_TEXT_DIM, len(self.dab_layers))
        # 总轮次
        self.epochs = args.epochs

        self.mse_sum = nn.MSELoss(reduction='sum')
        self.mse_none = nn.MSELoss(reduction='none')

    def forward(self, intermediate_features, t_intermediate_features,
                txt_intermediate_features, t_txt_intermediate_features,
                attention_weights, t_attention_weights, now_epoch, texts):
        """
          对于注意力分数，我们针对图像patch采用注意力分数的形式进行选择对齐，对于文本token我们只针对end_token之前的所有进行对齐

          Shape convention:
            - intermediate_features / t_intermediate_features: 每层 [B, L, D] (NLD, 已 permute)
            - attention_weights[0][layer] / t_attention_weights[0][layer]: [B*H, L, L]
              (PyTorch nn.MultiheadAttention 默认输出格式，head 维被 flatten 进 batch)
          """
        # 教师的注意权重分数：随训练轮次由学生主导渐变到教师主导
        weight = 0.5 * (1 + math.cos(now_epoch * math.pi / self.epochs))

        # 获取学生和教师注意力分数
        v_at = attention_weights[0]      # list[ [B*H, L, L], ... ] per layer
        v_t_at = t_attention_weights[0]  # list[ [B*H, L, L], ... ] per layer

        # 视觉部分：按注意力阈值筛选需要对齐的 patch
        # kept_count_total 统计跨 batch、跨 layer 的"被保留 patch 总数"，
        # 用于后续将 mse_sum 归一化为"每被保留 patch、每维度的平均 MSE"。
        kept_count_total = 0
        masked_s_features = []
        masked_t_features = []

        B = intermediate_features[self.dab_layers[0]].shape[0]
        L_full = intermediate_features[self.dab_layers[0]].shape[1]  # 含 cls，通常 196
        L_wo_cls = L_full - 1

        for layer in self.dab_layers:
            # ---- 把 [B*H, L, L] 还原成 [B, H, L, L] ----
            attn_shape = v_at[layer].shape
            assert attn_shape[1] == attn_shape[2] == L_full, \
                f"Attention shape mismatch: got {attn_shape}, features L={L_full}"
            H = attn_shape[0] // B
            s_attn = v_at[layer].view(B, H, L_full, L_full)
            t_attn = v_t_at[layer].view(B, H, L_full, L_full)

            # ---- 取 cls_token query 对各 patch 的注意力（去掉 cls 自身的 key）----
            # s_attn[:, :, 0, 1:] 即 [B, H, L_wo_cls]，再对 head 维求平均得到 [B, L_wo_cls]
            blend_at = (weight * t_attn[:, :, 0, 1:] + (1 - weight) * s_attn[:, :, 0, 1:]).mean(dim=1)

            # ---- 每个样本独立阈值：mean + alpha * std ----
            threshold = blend_at.mean(dim=1, keepdim=True) + self.mask_alpha * blend_at.std(dim=1, keepdim=True)
            # mask=True 表示该 patch 被保留参与对齐，shape [B, L_wo_cls]
            mask = blend_at >= threshold
            kept_count_total += mask.sum()

            # ---- 仅保留 mask=True 的 patch 做对齐 ----
            mask_expanded = mask.unsqueeze(-1)  # [B, L_wo_cls, 1]
            masked_s_features.append(intermediate_features[layer][:, 1:, :] * mask_expanded)
            masked_t_features.append(t_intermediate_features[layer][:, 1:, :] * mask_expanded)

        # 维度对齐
        s_vision_features = self.v_proj(torch.stack(masked_s_features, dim=1))  # [B, 3, N, D]
        t_vision_features = torch.stack(masked_t_features, dim=1)               # [B, 3, N, D]

        # ---- 归一化：mse_sum 是"被保留 patch × 所有维度"上的平方误差总和 ----
        # 除以 (保留总数 × 维度) 即得到"每被保留 patch、每维度的平均 MSE"，
        # 语义稳定（不随 mask 保留比例波动），量级与 MSELoss(reduction='mean') 一致。
        kept_count_total = kept_count_total.clamp_min(1)  # 全被 mask 掉时防止除零
        loss_dab_vision = self.mse_sum(s_vision_features, t_vision_features) / (
            kept_count_total * TEACHER_VISION_DIM) * DAB_ALPHA_VISION

        # 文本部分
        """
             现在的文本组成token的形式是 
             [sot_token] + _tokenizer.encode(text) + [eot_token]
             [eot_token]作为文本最后的总结token了。
             而进行文本模仿的应该是_tokenizer.encode(text)两个token中间的部分。
        """
        # 获取每个文本 token 的有效长度（去 sot/eot 后的真实 token 数）
        txt_token_length = texts.argmax(dim=-1) - 1  # [B]
        total_txt_length = txt_token_length.sum().clamp_min(1)

        s_txt_features = torch.stack(
            [extract_valid_tokens(texts, txt_intermediate_features[layer]) for layer in self.dab_layers],
            dim=1)  # [B, num_layers, max_tokens, D]
        s_txt_features = self.t_proj(s_txt_features)

        t_txt_features = torch.stack(
            [extract_valid_tokens(texts, t_txt_intermediate_features[layer]) for layer in self.dab_layers],
            dim=1)  # [B, num_layers, max_tokens, D]

        # ---- 归一化到"每有效 token、每维度的平均 MSE" ----
        # extract_valid_tokens 会用 0 填充不足的位置，需用 txt_token_length 屏蔽 padding 部分
        B_txt = s_txt_features.shape[0]
        L_txt = s_txt_features.shape[1]  # len(DAB_LAYERS)
        D_txt = s_txt_features.shape[-1]  # TEACHER_TEXT_DIM (投影后)

        # 构造 per-token 的有效 mask: [B, 1, max_tokens, 1]
        max_tokens = s_txt_features.shape[2]
        token_idx = torch.arange(max_tokens, device=s_txt_features.device).unsqueeze(0)
        valid_mask = (token_idx < txt_token_length.unsqueeze(1)).float()  # [B, max_tokens]
        valid_mask = valid_mask.unsqueeze(1).unsqueeze(-1)  # [B, 1, max_tokens, 1]

        # 逐元素平方误差，padding 位置置零
        err_none = self.mse_none(s_txt_features, t_txt_features)  # [B, 3, max_tokens, D]
        err_weighted = (err_none * valid_mask).sum(dim=(1, 2, 3))  # [B] per_sample sum (不含 padding)

        # 归一化因子: per_sample 的 num_valid_tokens × num_layers × 维度
        per_sample_denom = (txt_token_length * L_txt * D_txt).clamp_min(1)  # [B]
        loss_dab_txt = (err_weighted / per_sample_denom).mean() * DAB_ALPHA_TEXT

        return loss_dab_vision + loss_dab_txt
