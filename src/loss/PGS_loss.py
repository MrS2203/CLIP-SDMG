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


class pgs_loss(nn.Module):
    def __init__(self, args):
        super(pgs_loss, self).__init__()
        self.model_name = args.model

        for layer in DISTILL_LAYERS:
            # 视觉/文本的维度对齐层
            setattr(self, f'visual_proj_{layer}', EmbTrans(args))
            setattr(self, f'text_proj_{layer}', txt_EmbTrans(args))
            # 每个层各一组视觉/文本蒸馏损失
            setattr(self, f'v_loss_{layer}', loss(layer))
            setattr(self, f't_loss_{layer}', loss(layer))

        # 子模块以原属性名注册，保持 checkpoint 兼容；这里只建立便于遍历的引用
        self.visual_projs = [getattr(self, f'visual_proj_{layer}') for layer in DISTILL_LAYERS]
        self.text_projs = [getattr(self, f'text_proj_{layer}') for layer in DISTILL_LAYERS]
        self.v_losses = [getattr(self, f'v_loss_{layer}') for layer in DISTILL_LAYERS]
        self.t_losses = [getattr(self, f't_loss_{layer}') for layer in DISTILL_LAYERS]

    def forward(self, intermediate_features, t_intermediate_features,
                txt_intermediate_features, t_txt_intermediate_features, texts):
        # 1. 维度对齐
        eot_index = texts.argmax(dim=-1)  # 每段文本的 <end_of_text> 位置
        indexs = torch.arange(txt_intermediate_features[0].shape[0])

        cls_tokens = [proj(intermediate_features[layer][:, 0, :])
                      for proj, layer in zip(self.visual_projs, DISTILL_LAYERS)]
        t_cls_tokens = [t_intermediate_features[layer][:, 0, :] for layer in DISTILL_LAYERS]
        end_tokens = [proj(txt_intermediate_features[layer][indexs, eot_index])
                      for proj, layer in zip(self.text_projs, DISTILL_LAYERS)]
        t_end_tokens = [t_txt_intermediate_features[layer][indexs, eot_index] for layer in DISTILL_LAYERS]

        # 2. 计算损失
        vision_loss = sum(loss_fn(s, t) for loss_fn, s, t in zip(self.v_losses, cls_tokens, t_cls_tokens))
        text_loss = sum(loss_fn(s, t) for loss_fn, s, t in zip(self.t_losses, end_tokens, t_end_tokens))
        return vision_loss + text_loss


class EmbTrans(nn.Module):
    """视觉特征到教师维度的对齐（BatchNorm + Dropout + Linear）。"""

    def __init__(self, args):
        super(EmbTrans, self).__init__()
        self.model_name = args.model

        dims = get_student_dims(args.model)
        if dims is None:
            self.fc1 = None
            return

        s_emb = dims[0]
        self.fc1 = nn.Sequential(
            nn.BatchNorm1d(s_emb),
            nn.Dropout(0.5),
            nn.Linear(s_emb, TEACHER_VISION_DIM)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        return self.fc1(x) if self.fc1 is not None else x


class txt_EmbTrans(nn.Module):
    """文本特征到教师文本维度的对齐（LayerNorm + Dropout [+ Linear]）。"""

    def __init__(self, args):
        super(txt_EmbTrans, self).__init__()
        self.model_name = args.model

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

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.LayerNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        return self.fc1(x) if self.fc1 is not None else x


class loss(nn.Module):
    """单层特征的方向 + L2 蒸馏损失，浅层更侧重方向一致性。"""

    def __init__(self, layer=0):  # 层数默认为0
        super(loss, self).__init__()
        self.layer = layer
        # 线性衰减：越靠近输入层，角度损失的权重越大
        self.distance_weight = 1.0 - layer / (TOTAL_LAYERS - 1)

    def project(self, s_emb, t_emb):
        assert s_emb.size() == t_emb.size()
        angle_loss = 0.0
        distance_loss = 0.0

        for s, t in zip(s_emb, t_emb):
            # 归一化学生和教师特征
            s_norm = s / s.norm(p=2)
            t_norm = t / t.norm(p=2)
            # 计算方向相似性损失
            angle_loss += 1 - torch.dot(s_norm, t_norm)
            # 计算L2距离损失
            distance_loss += F.mse_loss(s, t)

        # 组合损失，按权重加权
        return (angle_loss * (1 - self.distance_weight) +
                distance_loss * self.distance_weight)

    def forward(self, s_emb, t_emb):
        nd_loss = self.project(s_emb=s_emb, t_emb=t_emb)
        return nd_loss / len(s_emb)  # 返回平均损失
