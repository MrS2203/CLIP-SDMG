import math

import torch
from torch import nn


class dab_loss(nn.Module):
    def __init__(self, args):
        super(dab_loss, self).__init__()
        self.model_name = args.model

        # 视觉维度对齐
        self.v_proj = dab_EmbTrans(args)
        # 文本维度对齐
        self.t_proj = dab_txt_EmbTrans(args)
        # 总轮次
        self.epochs = args.epochs
        # 视觉损失权重
        self.alpha_v = 0.00003
        # 文本损失权重
        self.alpha_t = 0.0003

    def forward(self, intermediate_features, t_intermediate_features,
                txt_intermediate_features, t_txt_intermediate_features,
                attention_weights, t_attention_weights, now_epoch, texts):
        """
          对于注意力分数，我们针对图像patch采用注意力分数的形式进行选择对齐，对于文本token我们只针对end_token之前的所有进行对齐
          """
        # 教师的注意权重分数
        weight = 0.5 * (1 + math.cos(now_epoch * math.pi / self.epochs))
        # 调节参数
        alpha = 0.5
        # 视觉掩码的数量
        mask_number = 0
        # 获取学生和教师注意力分数
        v_at = attention_weights[0]
        v_t_at = t_attention_weights[0]

        # 0层注意分数
        v_at0 = v_at[0][:, 0, 1:]
        v_t_at0 = v_t_at[0][:, 0, 1:]
        blend_at0 = weight * v_t_at0 + (1 - weight) * v_at0
        # 获取注意力分数阈值
        mean = blend_at0.mean(dim=1, keepdim=True)
        std = blend_at0.std(dim=1, keepdim=True)
        threshold = mean + alpha * std
        mask = blend_at0 >= threshold
        mask_number += (~mask).sum(dim=1).float().mean()
        mask_expanded = mask.unsqueeze(-1)
        # 0层特征
        v_features0 = intermediate_features[0][:, 1:, :]
        v_t_features0 = t_intermediate_features[0][:, 1:, :]
        v_features0 = v_features0 * mask_expanded
        v_t_features0 = v_t_features0 * mask_expanded

        # 1层注意分数
        v_at1 = v_at[1][:, 0, 1:]
        v_t_at1 = v_t_at[1][:, 0, 1:]
        blend_at1 = weight * v_t_at1 + (1 - weight) * v_at1
        # 获取注意力分数阈值
        mean = blend_at1.mean(dim=1, keepdim=True)
        std = blend_at1.std(dim=1, keepdim=True)
        threshold = mean + alpha * std
        mask = blend_at1 >= threshold
        mask_number += (~mask).sum(dim=1).float().mean()
        mask_expanded = mask.unsqueeze(-1)
        # 1层特征
        v_features1 = intermediate_features[1][:, 1:, :]
        v_t_features1 = t_intermediate_features[1][:, 1:, :]
        v_features1 = v_features1 * mask_expanded
        v_t_features1 = v_t_features1 * mask_expanded

        # 6层注意分数
        v_at6 = v_at[6][:, 0, 1:]
        v_t_at6 = v_t_at[6][:, 0, 1:]
        blend_at6 = weight * v_t_at6 + (1 - weight) * v_at6
        # 获取注意力分数阈值
        mean = blend_at6.mean(dim=1, keepdim=True)
        std = blend_at6.std(dim=1, keepdim=True)
        threshold = mean + alpha * std
        mask = blend_at6 >= threshold
        mask_number += (~mask).sum(dim=1).float().mean()
        mask_expanded = mask.unsqueeze(-1)
        # 6层特征
        v_features6 = intermediate_features[6][:, 1:, :]
        v_t_features6 = t_intermediate_features[6][:, 1:, :]
        v_features6 = v_features6 * mask_expanded
        v_t_features6 = v_t_features6 * mask_expanded

        # 维度对齐
        s_vision_features = torch.stack([v_features0, v_features1, v_features6], dim=1)  # B * 3 * N * D
        s_vision_features = self.v_proj(s_vision_features)

        # 计算损失
        t_vision_features = torch.stack([v_t_features0, v_t_features1, v_t_features6], dim=1)  # B * 3 * N * D
        B = intermediate_features[0].shape[0]
        patchs = intermediate_features[0].shape[1]
        loss_mse = nn.MSELoss(reduction='sum')
        loss_dab_vision = loss_mse(s_vision_features, t_vision_features) / B * self.alpha_v / (mask_number / 3 / patchs)

        # 文本部分
        loss_mse1 = nn.MSELoss(reduction='none')
        """
             现在的文本组成token的形式是 
             [sot_token] + _tokenizer.encode(text) + [eot_token]
             [eot_token]作为文本最后的总结token了。
             而进行文本模仿的应该是_tokenizer.encode(text)两个token中间的部分。
        """
        # 获取每个文本token的长度
        txt_token_length = texts.argmax(dim=-1) - 1
        # 计算文本长度的总和
        total_txt_length = txt_token_length.sum()
        s_txt_features = torch.stack([extract_valid_tokens(texts, txt_intermediate_features[0]),
                                      extract_valid_tokens(texts, txt_intermediate_features[1]),
                                      extract_valid_tokens(texts, txt_intermediate_features[6])],
                                     dim=1)  # B * 3 * N * D
        s_txt_features = self.t_proj(s_txt_features)

        t_txt_features = torch.stack([extract_valid_tokens(texts, t_txt_intermediate_features[0]),
                                      extract_valid_tokens(texts, t_txt_intermediate_features[1]),
                                      extract_valid_tokens(texts, t_txt_intermediate_features[6])],
                                     dim=1)  # B * 3 * N * D

        # 计算每个样本的文本token长度比例，作为加权因子
        txt_weight = txt_token_length / total_txt_length  # 计算每个样本的权重，使得总权重为1
        loss_dab_txt = loss_mse1(s_txt_features, t_txt_features).sum(dim=(1, 2, 3)) * self.alpha_t
        # 计算每个样本的总损失，并根据文本长度加权
        loss_dab_txt = (loss_dab_txt * txt_weight).sum()

        return loss_dab_vision + loss_dab_txt


# VIT部分0、1、6部分模仿
class dab_EmbTrans(nn.Module):
    def __init__(self, args):
        super(dab_EmbTrans, self).__init__()
        self.model_name = args.model

        if 'ViT-M-16' in args.model:
            student_dims = 512
            teacher_dims = 768
            self.align2 = nn.ModuleList([
                nn.Linear(student_dims, teacher_dims, bias=True)
                for i in range(3)])
        elif 'ViT-S-16' in args.model:
            student_dims = 384
            teacher_dims = 768
            self.align2 = nn.ModuleList([
                nn.Linear(student_dims, teacher_dims, bias=True)
                for i in range(3)])
        elif 'ViT-T-16' in args.model:
            student_dims = 192
            teacher_dims = 768
            self.align2 = nn.ModuleList([
                nn.Linear(student_dims, teacher_dims, bias=True)
                for i in range(3)])
        else:
            self.align2 = None

    def forward(self,
                preds):
        """Forward function.
        Args:
            preds_S(List): [B*3*N*D, B*N*D], student's feature map
            preds_T(List): [B*3*N*D, B*N*D], teacher's feature map
        """
        low_s = preds

        if self.align2 is not None:
            for i in range(3):
                if i == 0:
                    xc = self.align2[i](low_s[:, i]).unsqueeze(1)
                else:
                    xc = torch.cat((xc, self.align2[i](low_s[:, i]).unsqueeze(1)), dim=1)
        else:
            xc = low_s
        return xc


# 只获取有效的文本token
def extract_valid_tokens(text, feature):
    """
    提取每个文本中 eot_token 之前的有效 token，去除 sot_token 和 eot_token。

    参数:
    text (torch.LongTensor): 输入的文本 token 序列，形状为 [batch_size, sequence_length]。
    sot_token (int): <start_of_text> token 的索引。
    eot_token (int): <end_of_text> token 的索引。

    返回:
    torch.LongTensor: 每个文本中 eot_token 之前的有效 token（不包括 eot_token 和 sot_token）。
                      张量形状为 [batch_size, max_valid_tokens]。
    """
    # 获取每个文本中 eot_token 的位置
    eot_positions = text.argmax(dim=-1)
    valid_tokens = []
    for i in range(feature.shape[0]):
        tokens = feature[i]
        eot_pos = eot_positions[i]

        # 提取从 sot_token 后开始到 eot_token 前的所有 token，排除 sot_token 和 eot_token
        valid_tokens_i = tokens[1:eot_pos]  # 去除 sot_token 和 eot_token
        valid_tokens.append(valid_tokens_i)

    # 使用 pad_sequence 填充每个序列，保证所有序列有相同的长度
    valid_tokens = torch.nn.utils.rnn.pad_sequence([torch.tensor(v) for v in valid_tokens],
                                                   batch_first=True,
                                                   padding_value=0)
    return valid_tokens


# 文本部分0、1、6部分模仿
class dab_txt_EmbTrans(nn.Module):
    def __init__(self, args):
        super(dab_txt_EmbTrans, self).__init__()
        self.model_name = args.model

        if 'ViT-M-16' in args.model:
            student_dims = 512
            teacher_dims = 512
            self.align2 = nn.ModuleList([
                nn.Linear(student_dims, teacher_dims, bias=True)
                for i in range(3)])

        elif 'ViT-S-16' in args.model:
            student_dims = 384
            teacher_dims = 512
            self.align2 = nn.ModuleList([
                nn.Linear(student_dims, teacher_dims, bias=True)
                for i in range(3)])

        elif 'ViT-T-16' in args.model:
            student_dims = 384
            teacher_dims = 512
            self.align2 = nn.ModuleList([
                nn.Linear(student_dims, teacher_dims, bias=True)
                for i in range(3)])
        else:
            self.align2 = None

    def forward(self,
                preds):
        """Forward function.
        Args:
            preds_S(List): [B*3*N*D, B*N*D], student's feature map
            preds_T(List): [B*3*N*D, B*N*D], teacher's feature map
        """
        low_s = preds

        '''ViTKD: Mimicking'''
        if self.align2 is not None:
            for i in range(3):
                if i == 0:
                    xc = self.align2[i](low_s[:, i]).unsqueeze(1)
                else:
                    xc = torch.cat((xc, self.align2[i](low_s[:, i]).unsqueeze(1)), dim=1)
        else:
            xc = low_s

        return xc
