"""生成蒸馏损失：视觉 Mask-Verify + 文本 Autoregressive 生成。

结构说明：
  - TokenGenerator: 最后一层视觉特征的掩码生成与重建（返回 GenerationOutput NamedTuple）；
  - MLM: 文本解码器（逐 step autoregressive next-token 预测）；
  - GENLoss: 组合视觉 mse + 文本 CE，用 beta 系数做量级缩放。
"""

from collections import namedtuple

import torch
from torch import nn

from src.generation_module.SEG import SEModule
from src.generation_module.MLM import MLM

from .common import TEACHER_VISION_DIM, get_student_dims

# ---- Loss 权重（集中在 common.py，这里只做 fallback 默认值）----
GEN_BETA_VISION = 0.000003
GEN_BETA_TEXT = 0.006
GEN_MASK_RATIO_DEFAULT = 0.5


# ---- TokenGenerator 返回类型 ----
# 用 NamedTuple 替代 magic-string dict key，IDE 可补全、重构安全。
GenerationOutput = namedtuple('GenerationOutput', ['tokens', 'mask', 'mask_ratio'])


class GENLoss(nn.Module):
    def __init__(self, args):
        super().__init__()
        # ---- 超参集中化：优先用 args 里的值，否则 fallback 到文件顶部默认 ----
        gen_mask_ratio = getattr(args, 'gen_mask_ratio', GEN_MASK_RATIO_DEFAULT)

        # 视觉生成
        self.mvm = TokenGenerator(args.model, lambda_vitkd=gen_mask_ratio)
        # 文本解码器
        self.mlm = MLM(args)
        # 文本生成损失
        self.mlm_loss = nn.CrossEntropyLoss(reduction='sum', ignore_index=0)
        self.mse_sum = nn.MSELoss(reduction='sum')

    def forward(self, intermediate_features, t_intermediate_features,
                txt_intermediate_features, texts):
        # ---- 视觉生成部分 ----
        v_gen: GenerationOutput = self.mvm(intermediate_features[-1][:, 1:, :])
        # v_gen.tokens: [B, N, D] 重建后的视觉特征
        # v_gen.mask   : [B, N, 1]  0=原位置, 1=被 mask 重建
        # v_gen.mask_ratio: float, 遮挡比例
        high_t = t_intermediate_features[-1][:, 1:, :]  # 教师原始视觉特征

        mse_masked = self.mse_sum(v_gen.tokens * v_gen.mask,
                                  high_t * v_gen.mask)
        loss_gen_vision = mse_masked / v_gen.tokens.shape[0] * GEN_BETA_VISION / v_gen.mask_ratio

        # ---- 文本生成部分 ----
        """
        texts token 格式: [sot_token] + encode(text) + [eot_token]
        MLM 解码器输入编码输出，预测下一个 token。
        """
        encode_output = txt_intermediate_features[-1]  # [B, T, D]
        encode_length = texts.argmax(dim=-1)            # [B] 每段 eot 的位置
        logits = self.mlm(encode_output, encode_length)

        # shift labels: decoder 输入是 texts[:, :-1]，目标是 texts[:, 1:]
        # 不覆盖原 texts 变量，避免后续维护者困惑
        text_labels = texts[:, 1:]
        loss_gen_txt = (self.mlm_loss(
            logits.reshape(-1, logits.size(-1)),
            text_labels.reshape(-1),
        ) * GEN_BETA_TEXT)

        return loss_gen_txt + loss_gen_vision


class TokenGenerator(nn.Module):
    """最后一层视觉特征的掩码生成：先对齐到教师维度，再遮挡部分 token 并重建。"""

    def __init__(self, model_name, lambda_vitkd=GEN_MASK_RATIO_DEFAULT):
        super().__init__()
        self.model_name = model_name
        # 遮挡比例（被 mask 的 token 占比，不是保留比例）
        self.lambda_vitkd = lambda_vitkd

        dims = get_student_dims(model_name)
        if dims is None:
            self.align = None
            return

        student_dims = dims[0]
        self.align = nn.Linear(student_dims, TEACHER_VISION_DIM, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, TEACHER_VISION_DIM))
        self.generation = SEModule(TEACHER_VISION_DIM)

    def forward(self, preds):
        """
        Args:
            preds: [B, N, D_student] 学生的视觉特征（去掉 cls token）

        Returns:
            GenerationOutput(tokens, mask, mask_ratio)
        """
        high = preds

        if self.align is not None:
            x = self.align(high)                       # [B, N, D_t]
            B, N, D = x.shape
            x, mat, ids, _ = self.random_masking(x, self.lambda_vitkd)

            # 把 mask token 填回被遮挡位置
            mask_tokens = self.mask_token.repeat(B, N - x.shape[1], 1)
            x_ = torch.cat([x, mask_tokens], dim=1)
            x = torch.gather(x_, dim=1, index=ids.unsqueeze(-1).repeat(1, 1, D))
            mask = mat.unsqueeze(-1)  # [B, N, 1]  0=保留, 1=被 mask 重建

            # 还原到 2D 做卷积式重建，再 flatten 回来
            hw = int(N ** 0.5)
            x = x.reshape(B, hw, hw, D).permute(0, 3, 1, 2)
            x = self.generation(x).flatten(2).transpose(1, 2)
            return GenerationOutput(tokens=x, mask=mask, mask_ratio=self.lambda_vitkd)

        # 未支持的学生模型，退化为恒等映射
        return GenerationOutput(tokens=high,
                                mask=torch.zeros(high.shape[0], high.shape[1], 1, device=high.device),
                                mask_ratio=1.0)

    def random_masking(self, x, mask_ratio):
        """随机遮挡部分 token（MAE 风格：小噪声保留大噪声遮挡）。

        Returns:
            x_keep:     [B, len_keep, D] 保留的 token
            mask:       [B, L] 0=保留 1=被 mask（按原始顺序排列）
            ids_restore:[B, L] 用于还原原始顺序
            ids_masked: [B, len_mask] 被 mask 的位置索引（原始顺序）
        """
        B, L, D = x.shape
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(B, L, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)       # 小噪声在前（保留），大噪声在后（遮挡）
        ids_restore = torch.argsort(ids_shuffle, dim=1)  # 反向还原原始索引

        ids_keep = ids_shuffle[:, :len_keep]
        ids_masked = ids_shuffle[:, len_keep:L]

        x_keep = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # mask 先按 shuffle 顺序写 0/1，再 unshuffle 回原始顺序
        mask = torch.ones(B, L, device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_keep, mask, ids_restore, ids_masked
