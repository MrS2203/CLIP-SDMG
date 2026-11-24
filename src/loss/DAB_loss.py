import math

import torch
from torch import nn

from .common import (
    DAB_LAYERS,
    TEACHER_TEXT_DIM,
    TEACHER_VISION_DIM,
    MultiLayerLinearProjection,
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

        dims = get_student_dims(args.model)
        vision_dim, text_dim = dims if dims is not None else (None, None)

        # 视觉/文本维度对齐
        self.v_proj = MultiLayerLinearProjection(vision_dim, TEACHER_VISION_DIM, len(DAB_LAYERS))
        self.t_proj = MultiLayerLinearProjection(text_dim, TEACHER_TEXT_DIM, len(DAB_LAYERS))
        # 总轮次
        self.epochs = args.epochs

        self.mse_sum = nn.MSELoss(reduction='sum')
        self.mse_none = nn.MSELoss(reduction='none')

    def forward(self, intermediate_features, t_intermediate_features,
                txt_intermediate_features, t_txt_intermediate_features,
                attention_weights, t_attention_weights, now_epoch, texts):
        """
          对于注意力分数，我们针对图像patch采用注意力分数的形式进行选择对齐，对于文本token我们只针对end_token之前的所有进行对齐
          """
        # 教师的注意权重分数：随训练轮次由学生主导渐变到教师主导
        weight = 0.5 * (1 + math.cos(now_epoch * math.pi / self.epochs))

        # 获取学生和教师注意力分数
        v_at = attention_weights[0]
        v_t_at = t_attention_weights[0]

        # 视觉部分：按注意力阈值筛选需要对齐的 patch
        mask_number = 0
        masked_s_features = []
        masked_t_features = []
        for layer in DAB_LAYERS:
            blend_at = weight * v_t_at[layer][:, 0, 1:] + (1 - weight) * v_at[layer][:, 0, 1:]
            # 获取注意力分数阈值
            threshold = blend_at.mean(dim=1, keepdim=True) + DAB_MASK_ALPHA * blend_at.std(dim=1, keepdim=True)
            mask = blend_at >= threshold
            mask_number += (~mask).sum(dim=1).float().mean()

            mask_expanded = mask.unsqueeze(-1)
            masked_s_features.append(intermediate_features[layer][:, 1:, :] * mask_expanded)
            masked_t_features.append(t_intermediate_features[layer][:, 1:, :] * mask_expanded)

        # 维度对齐
        s_vision_features = self.v_proj(torch.stack(masked_s_features, dim=1))  # B * 3 * N * D
        t_vision_features = torch.stack(masked_t_features, dim=1)  # B * 3 * N * D

        B = intermediate_features[DAB_LAYERS[0]].shape[0]
        patches = intermediate_features[DAB_LAYERS[0]].shape[1]
        # mask_number 为 0 时分子也为 0，clamp 避免除零产生 NaN
        masked_ratio = (mask_number / len(DAB_LAYERS) / patches).clamp_min(1e-6)
        loss_dab_vision = self.mse_sum(s_vision_features, t_vision_features) / B * DAB_ALPHA_VISION / masked_ratio

        # 文本部分
        """
             现在的文本组成token的形式是 
             [sot_token] + _tokenizer.encode(text) + [eot_token]
             [eot_token]作为文本最后的总结token了。
             而进行文本模仿的应该是_tokenizer.encode(text)两个token中间的部分。
        """
        # 获取每个文本token的长度
        txt_token_length = texts.argmax(dim=-1) - 1
        # 计算文本长度的总和，clamp 避免空文本导致除零
        total_txt_length = txt_token_length.sum().clamp_min(1)

        s_txt_features = torch.stack(
            [extract_valid_tokens(texts, txt_intermediate_features[layer]) for layer in DAB_LAYERS],
            dim=1)  # B * 3 * N * D
        s_txt_features = self.t_proj(s_txt_features)

        t_txt_features = torch.stack(
            [extract_valid_tokens(texts, t_txt_intermediate_features[layer]) for layer in DAB_LAYERS],
            dim=1)  # B * 3 * N * D

        # 计算每个样本的文本token长度比例，作为加权因子
        txt_weight = txt_token_length / total_txt_length  # 计算每个样本的权重，使得总权重为1
        loss_dab_txt = self.mse_none(s_txt_features, t_txt_features).sum(dim=(1, 2, 3)) * DAB_ALPHA_TEXT
        # 计算每个样本的总损失，并根据文本长度加权
        loss_dab_txt = (loss_dab_txt * txt_weight).sum()

        return loss_dab_vision + loss_dab_txt


# 只获取有效的文本token
def extract_valid_tokens(text, feature):
    """
    提取每个文本中 eot_token 之前的有效 token，去除 sot_token 和 eot_token。

    参数:
    text (torch.LongTensor): 输入的文本 token 序列，形状为 [batch_size, sequence_length]。
    sot_token (int): <start_of_text> token 的索引。
    eot_token (int): <end_of_text> token 的索引。

    返回:
    torch.Tensor: 每个文本中 eot_token 之前的有效 token 特征（不包括 eot_token 和 sot_token），
                  形状为 [batch_size, max_valid_tokens, dim]，不足处补零。
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
    valid_tokens = torch.nn.utils.rnn.pad_sequence(valid_tokens,
                                                   batch_first=True,
                                                   padding_value=0)
    return valid_tokens
