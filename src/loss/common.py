"""蒸馏损失共用的维度表、层号与投影模块。"""

import torch
import torch.nn as nn


# 教师模型（ViT-B-16）的视觉/文本特征宽度
TEACHER_VISION_DIM = 768
TEACHER_TEXT_DIM = 512

# Transformer 的编码层总数
TOTAL_LAYERS = 12

# 全局语义渐进式蒸馏（PGS）使用的层号
DISTILL_LAYERS = (0, 1, 6, 11)
# 注意力浅层模仿（DAB）使用的层号
DAB_LAYERS = (0, 1, 6)

# 学生模型各模态的宽度：{模型名: (视觉宽度, 文本宽度)}
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


class MultiLayerLinearProjection(nn.Module):
    """把多个中间层特征分别线性投影到教师维度。

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
