"""蒸馏损失共用的维度表、层号与投影模块。"""

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence


# ---- 教师模型（ViT-B-16）的视觉/文本特征宽度 ----
TEACHER_VISION_DIM = 768
TEACHER_TEXT_DIM = 512

# ---- Transformer 的编码层总数 ----
TOTAL_LAYERS = 12

# ---- 蒸馏使用的默认层号 ----
DISTILL_LAYERS = (0, 1, 6, 11)   # PGS 全局语义渐进蒸馏
DAB_LAYERS = (0, 1, 6)           # DAB 注意力浅层模仿

# ---- 学生模型各模态的宽度：{模型名: (视觉宽度, 文本宽度)} ----
MODEL_DIMS = {
    'ViT-M-16': (512, 512),
    'ViT-S-16': (384, 384),
    'ViT-T-16': (192, 384),
}


def get_student_dims(model_name):
    """返回学生模型的 (视觉宽度, 文本宽度)；不支持的模型返回 None。"""
    for name, dims in MODEL_DIMS.items():
        if name in model_name:
            return dims
    return None


# ---------------------------------------------------------------------------
# 文本 token 工具
# ---------------------------------------------------------------------------

def extract_valid_tokens(text, feature):
    """提取每个文本中 eot_token 之前的有效 token 特征。

    输入 token 格式: [sot] + encode(text) + [eot] + padding...
    输出: 去掉 sot/eot 的 encode(text) 部分，用 0 padding 到 batch 内最长长度。

    Args:
        text:    [B, T] 文本 token 序列（LongTensor）
        feature: [B, T, D] 对应位置的 encoder 输出特征

    Returns:
        [B, max_valid_tokens, D] 有效 token 特征，不足处补 0。
    """
    eot_positions = text.argmax(dim=-1)
    valid_tokens = [feature[i, 1:eot_positions[i]] for i in range(feature.shape[0])]
    return pad_sequence(valid_tokens, batch_first=True, padding_value=0)


# ---------------------------------------------------------------------------
# 多层投影模块
# ---------------------------------------------------------------------------

class MultiLayerLinearProjection(nn.Module):
    """把多个中间层特征分别线性投影到教师维度。

    forward 输入: [B, num_layers, N, D_in]
    输出:        [B, num_layers, N, D_out]
    当输入维度未知（不支持的模型）时退化为恒等映射。
    """

    def __init__(self, in_dim, out_dim, num_layers):
        super().__init__()
        if in_dim is None:
            self.align2 = None
        else:
            self.align2 = nn.ModuleList([
                nn.Linear(in_dim, out_dim, bias=True)
                for _ in range(num_layers)
            ])

    def forward(self, preds):
        if self.align2 is None:
            return preds
        return torch.stack([proj(preds[:, i]) for i, proj in enumerate(self.align2)], dim=1)
