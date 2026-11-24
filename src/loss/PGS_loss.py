"""PGS 蒸馏损失：方向 + L2 逐层对齐。

结构说明：
  - 维度对齐层（VisualAligner / TextAligner）把学生特征映射到教师特征空间；
  - LayerAlignmentLoss 按层计算 angle_loss + distance_loss 的加权组合；
  - PGSLoss 用 nn.ModuleDict 管理多层、双模态的子模块，避免 setattr/getattr 的动态注册。
"""

import torch
from torch import nn
from torch.nn import functional as F

from .common import (
    DISTILL_LAYERS,
    TEACHER_TEXT_DIM,
    TEACHER_VISION_DIM,
    TOTAL_LAYERS,
    get_student_dims,
)


class _InitWeightsMixin:
    """VisualAligner / TextAligner 共用的权重初始化：LayerNorm 置 1/0，Linear 偏置置 0。"""

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.LayerNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()


class VisualAligner(_InitWeightsMixin, nn.Module):
    """视觉特征到教师维度的对齐（LayerNorm + Dropout + Linear）。

    使用 LayerNorm 而非 BatchNorm：
      - 蒸馏任务中每个样本独立对齐，逐样本的 LayerNorm 比跨样本的 BatchNorm 更合适；
      - 避免小 batch（甚至 batch_size=1）下 BN 统计量跳变导致的不稳定；
      - 训练/推理行为一致，不需要额外的 running_mean/var。
    """

    def __init__(self, args):
        super().__init__()
        dims = get_student_dims(args.model)
        if dims is None:
            self.fc1 = None
            return

        s_emb = dims[0]
        self.fc1 = nn.Sequential(
            nn.LayerNorm(s_emb),
            nn.Dropout(0.5),
            nn.Linear(s_emb, TEACHER_VISION_DIM),
        )
        self._init_weights()

    def forward(self, x):
        return self.fc1(x) if self.fc1 is not None else x


class TextAligner(_InitWeightsMixin, nn.Module):
    """文本特征到教师文本维度的对齐（LayerNorm + Dropout [+ Linear]）。"""

    def __init__(self, args):
        super().__init__()
        dims = get_student_dims(args.model)
        if dims is None:
            self.fc1 = None
            return

        s_emb = dims[1]
        layers = [nn.LayerNorm(s_emb), nn.Dropout(0.5)]
        if s_emb != TEACHER_TEXT_DIM:
            layers.append(nn.Linear(s_emb, TEACHER_TEXT_DIM))
        self.fc1 = nn.Sequential(*layers)
        self._init_weights()

    def forward(self, x):
        return self.fc1(x) if self.fc1 is not None else x


class LayerAlignmentLoss(nn.Module):
    """单层特征的方向 + L2 蒸馏损失，浅层更侧重方向一致性。"""

    def __init__(self, layer=0):
        super().__init__()
        self.layer = layer
        # 线性衰减：越靠近输入层，方向损失的权重越大
        self.distance_weight = 1.0 - layer / (TOTAL_LAYERS - 1)

    def forward(self, s_emb, t_emb):
        assert s_emb.size() == t_emb.size()
        s_norm = F.normalize(s_emb, p=2, dim=-1)
        t_norm = F.normalize(t_emb, p=2, dim=-1)
        angle_loss = 1 - (s_norm * t_norm).sum(dim=-1)
        distance_loss = F.mse_loss(s_emb, t_emb, reduction='none').mean(dim=-1)
        return (angle_loss * (1 - self.distance_weight) +
                distance_loss * self.distance_weight).mean()


class PGSLoss(nn.Module):
    """全局语义渐进式蒸馏：对 distill_layers 的 cls/end token 做逐层对齐。"""

    def __init__(self, args):
        super().__init__()
        self.model_name = args.model

        # ---- 超参集中化 ----
        self.distill_layers = getattr(args, 'distill_layers', DISTILL_LAYERS)

        # ---- 用 nn.ModuleDict 注册多层子模块（比 setattr/getattr 更清晰）----
        # 注意：key 必须是字符串；state_dict 的 key 会是 "visual_projs.<layer>.fc1.xxx"
        # 与之前 setattr 版本的 "visual_proj_<layer>.fc1.xxx" 不同，如需 checkpoint 兼容，
        # 可在 load_state_dict 时做一次 key 重映射（此处先优先结构清晰）。
        self.visual_projs = nn.ModuleDict({
            str(layer): VisualAligner(args) for layer in self.distill_layers
        })
        self.text_projs = nn.ModuleDict({
            str(layer): TextAligner(args) for layer in self.distill_layers
        })
        self.v_losses = nn.ModuleDict({
            str(layer): LayerAlignmentLoss(layer) for layer in self.distill_layers
        })
        self.t_losses = nn.ModuleDict({
            str(layer): LayerAlignmentLoss(layer) for layer in self.distill_layers
        })

        # 保持与原始 forward 中 list 访问方式一致的迭代顺序
        self._layer_keys = [str(layer) for layer in self.distill_layers]

    def forward(self, intermediate_features, t_intermediate_features,
                txt_intermediate_features, t_txt_intermediate_features, texts):
        # 每段文本的 <end_of_text> 位置
        eot_index = texts.argmax(dim=-1)
        batch_idx = torch.arange(txt_intermediate_features[0].shape[0])

        cls_tokens = [self.visual_projs[k](intermediate_features[layer][:, 0, :])
                      for k, layer in zip(self._layer_keys, self.distill_layers)]
        t_cls_tokens = [t_intermediate_features[layer][:, 0, :]
                        for layer in self.distill_layers]
        end_tokens = [self.text_projs[k](txt_intermediate_features[layer][batch_idx, eot_index])
                      for k, layer in zip(self._layer_keys, self.distill_layers)]
        t_end_tokens = [t_txt_intermediate_features[layer][batch_idx, eot_index]
                        for layer in self.distill_layers]

        vision_loss = sum(self.v_losses[k](s, t) for k, s, t in zip(self._layer_keys, cls_tokens, t_cls_tokens))
        text_loss = sum(self.t_losses[k](s, t) for k, s, t in zip(self._layer_keys, end_tokens, t_end_tokens))
        return vision_loss + text_loss
