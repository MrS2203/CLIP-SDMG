import torch
import math
from torch import nn
from torch.nn import functional as F


class pgs_loss(nn.Module):
    def __init__(self, args):
        super(pgs_loss, self).__init__()
        self.model_name = args.model

        # 视觉维度对齐
        self.visual_proj_0 = EmbTrans(args)
        self.visual_proj_1 = EmbTrans(args)
        self.visual_proj_6 = EmbTrans(args)
        self.visual_proj_11 = EmbTrans(args)
        # 文本维度对齐
        self.text_proj_0 = txt_EmbTrans(args)
        self.text_proj_1 = txt_EmbTrans(args)
        self.text_proj_6 = txt_EmbTrans(args)
        self.text_proj_11 = txt_EmbTrans(args)

        # 损失函数
        self.v_loss_0 = loss(0)
        self.v_loss_1 = loss(1)
        self.v_loss_6 = loss(6)
        self.v_loss_11 = loss(11)

        self.t_loss_0 = loss(0)
        self.t_loss_1 = loss(1)
        self.t_loss_6 = loss(6)
        self.t_loss_11 = loss(11)

    def forward(self, intermediate_features, t_intermediate_features,
                txt_intermediate_features, t_txt_intermediate_features, texts):
        pgs_loss = torch.tensor(0.).cuda()
        # 1. 维度对齐
        # 视觉
        cls_tokens = [
            self.visual_proj_0(intermediate_features[0][:, 0, :]),
            self.visual_proj_1(intermediate_features[1][:, 0, :]),
            self.visual_proj_6(intermediate_features[6][:, 0, :]),
            self.visual_proj_11(intermediate_features[11][:, 0, :])]
        t_cls_tokens = [t_intermediate_features[0][:, 0, :],
                        t_intermediate_features[1][:, 0, :],
                        t_intermediate_features[6][:, 0, :],
                        t_intermediate_features[11][:, 0, :]]

        # 文本
        indexs = torch.arange(txt_intermediate_features[0].shape[0])
        end_tokens = [txt_intermediate_features[0][indexs, texts.argmax(dim=-1)],
                      txt_intermediate_features[1][indexs, texts.argmax(dim=-1)],
                      txt_intermediate_features[6][indexs, texts.argmax(dim=-1)],
                      txt_intermediate_features[11][indexs, texts.argmax(dim=-1)]]
        # 隐射层
        end_tokens = [self.text_proj_0(end_tokens[0]),
                      self.text_proj_1(end_tokens[1]),
                      self.text_proj_6(end_tokens[2]),
                      self.text_proj_11(end_tokens[3])]
        t_end_tokens = [
            t_txt_intermediate_features[0][indexs, texts.argmax(dim=-1)],
            t_txt_intermediate_features[1][indexs, texts.argmax(dim=-1)],
            t_txt_intermediate_features[6][indexs, texts.argmax(dim=-1)],
            t_txt_intermediate_features[11][indexs, texts.argmax(dim=-1)]
        ]
        # 计算损失
        # 视觉
        pgs_loss = self.v_loss_0(cls_tokens[0], t_cls_tokens[0]) + \
                   self.v_loss_1(cls_tokens[1], t_cls_tokens[1]) + \
                   self.v_loss_6(cls_tokens[2], t_cls_tokens[2]) + \
                   self.v_loss_11(cls_tokens[3], t_cls_tokens[3]) + \
                   self.t_loss_0(end_tokens[0], t_end_tokens[0]) + \
                   self.t_loss_1(end_tokens[1], t_end_tokens[1]) + \
                   self.t_loss_6(end_tokens[2], t_end_tokens[2]) + \
                   self.t_loss_11(end_tokens[3], t_end_tokens[3])

        return pgs_loss


class EmbTrans(nn.Module):
    def __init__(self, args):
        super(EmbTrans, self).__init__()
        self.model_name = args.model

        if 'ViT-M-16' in self.model_name:
            s_emb = 512
            t_emb = 768
            self.fc1 = nn.Sequential(
                nn.BatchNorm1d(s_emb),
                nn.Dropout(0.5),
                nn.Linear(s_emb, t_emb)
            )

        elif 'ViT-S-16' in self.model_name:
            s_emb = 384
            t_emb = 768
            self.fc1 = nn.Sequential(
                nn.BatchNorm1d(s_emb),
                nn.Dropout(0.5),
                nn.Linear(s_emb, t_emb)
            )

        elif 'ViT-T-16' in self.model_name:
            s_emb = 192
            t_emb = 768
            self.fc1 = nn.Sequential(
                nn.BatchNorm1d(s_emb),
                nn.Dropout(0.5),
                nn.Linear(s_emb, t_emb)
            )
        else:
            self.fc1 = None

        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        if self.model_name in 'ViT-B-16':
            return x
        else:
            if self.fc1 is not None:
                cls_token = self.fc1(x)
                return cls_token
            else:
                return x


class txt_EmbTrans(nn.Module):

    def __init__(self, args):
        super(txt_EmbTrans, self).__init__()
        self.model_name = args.model

        if 'ViT-M-16' in self.model_name:
            s_emb = 512
            self.fc1 = nn.Sequential(
                nn.LayerNorm(s_emb),
                nn.Dropout(0.5),
            )

        elif 'ViT-S-16' in self.model_name:
            s_emb = 384
            t_emb = 512
            self.fc1 = nn.Sequential(
                nn.LayerNorm(s_emb),
                nn.Dropout(0.5),
                nn.Linear(s_emb, t_emb)
            )
        elif 'ViT-T-16' in self.model_name:
            s_emb = 384
            t_emb = 512
            self.fc1 = nn.Sequential(
                nn.LayerNorm(s_emb),
                nn.Dropout(0.5),
                nn.Linear(s_emb, t_emb)
            )
        else:
            self.fc1 = None

        for m in self.modules():
            if isinstance(m, nn.LayerNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        if self.model_name in 'ViT-B-16':
            return x
        else:
            if self.fc1 is not None:
                cls_token = self.fc1(x)
                return cls_token
            else:
                return x


class loss(nn.Module):
    def __init__(self, layer=0):  # 层数默认为0
        super(loss, self).__init__()
        self.total_layers = 12
        self.layer = layer
        self.kl_loss = DistillKL(T=1)

        # 线性衰减
        self.distance_weight = 1.0 - layer / (self.total_layers - 1)

    def project(self, s_emb, t_emb):
        assert s_emb.size() == t_emb.size()
        feat_dim = s_emb.shape[1]
        angle_loss = 0.0
        distance_loss = 0.0
        kld_loss = 0.0

        sqrt_dim = math.sqrt(feat_dim)

        for s, t in zip(s_emb, t_emb):
            # 归一化学生和教师特征
            s_norm = s / s.norm(p=2)
            t_norm = t / t.norm(p=2)
            # 计算方向相似性损失
            angle_loss += 1 - torch.dot(s_norm, t_norm)
            # 计算L2距离损失，除以特征维度的平方根进行缩放
            distance_loss += F.mse_loss(s, t)
            # # kld
            # s_k = s_emb.unsqueeze(0)
            # t_k = t_emb.unsqueeze(0)
            # kld_loss += self.kl_loss(s_k, t_k) / sqrt_dim

        # 组合损失，按权重加权
        total_loss = ((angle_loss + kld_loss) * (1 - self.distance_weight) +
                      distance_loss * self.distance_weight)
        return total_loss

    def forward(self, s_emb, t_emb):
        nd_loss = self.project(s_emb=s_emb, t_emb=t_emb)
        return nd_loss / len(s_emb)  # 返回平均损失


class DistillKL(nn.Module):
    """Distilling the Knowledge in a Neural Network"""

    def __init__(self, T):
        super(DistillKL, self).__init__()
        self.T = T

    def forward(self, y_s, y_t):
        p_s = F.log_softmax(y_s / self.T, dim=1)
        p_t = F.softmax(y_t / self.T, dim=1)
        loss = F.kl_div(p_s, p_t, reduction='batchmean') * (self.T ** 2)
        return loss
