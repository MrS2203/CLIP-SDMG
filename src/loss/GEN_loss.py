import torch
from torch import nn
from src.generation_module.SEG import SEModule
from src.generation_module.MLM import MLM


class gen_loss(nn.Module):
    def __init__(self, args):
        super(gen_loss, self).__init__()
        # 视觉生成
        self.mvm = gen_EmbTrans(args.model)
        # 文本解码器
        self.mlm = MLM(args)
        # 文本生成损失
        self.mlm_loss = nn.CrossEntropyLoss(reduction='sum', ignore_index=0)

    def forward(self, intermediate_features, t_intermediate_features,
                txt_intermediate_features, texts):
        # 视觉生成部分
        loss_mse = nn.MSELoss(reduction='sum')
        beta_vitkd = 0.000003
        v_gen = self.mvm(intermediate_features[-1][:, 1:, :])
        x = v_gen['tokens']
        B = x.shape[0]
        mask = v_gen['mask'][0]
        lambda_vitkd = v_gen['mask_ratio']

        high_t = t_intermediate_features[-1][:, 1:, :]

        loss_gen = loss_mse(torch.mul(x, mask), torch.mul(high_t, mask))
        loss_gen_vision = loss_gen / B * beta_vitkd / lambda_vitkd

        # 文本生成部分
        """
        bos_id 49406
        eos_id 49407 
        编码器的输出 [<bos>, token1, token2, ....., tokenN, <eos>, ...]
        """
        beta_vitkd_txt = 0.006
        # 获取取编码器的输出和有效长度(以0为起始)
        encode_output, encode_length = txt_intermediate_features[-1], texts.argmax(dim=-1)
        logits = self.mlm(encode_output, encode_length)

        # 计算损失
        texts = texts[:, 1:]
        loss_gen_txt = self.mlm_loss(logits.reshape(-1, logits.size(-1)), texts.reshape(-1)) * beta_vitkd_txt

        return loss_gen_txt + loss_gen_vision


# VIT 11生成
class gen_EmbTrans(nn.Module):
    def __init__(self,
                 model_name,
                 lambda_vitkd=0.5,
                 ):
        super(gen_EmbTrans, self).__init__()
        self.model_name = model_name
        # 遮挡比例
        self.lambda_vitkd = lambda_vitkd

        if 'ViT-M-16' in model_name:
            student_dims = 512
            teacher_dims = 768
            self.align = nn.Linear(student_dims, teacher_dims, bias=True)
            self.mask_token = nn.Parameter(torch.zeros(1, 1, teacher_dims))
            self.generation = SEModule(teacher_dims)


        elif 'ViT-S-16' in model_name:
            student_dims = 384
            teacher_dims = 768
            self.align = nn.Linear(student_dims, teacher_dims, bias=True)

            self.mask_token = nn.Parameter(torch.zeros(1, 1, teacher_dims))
            self.generation = SEModule(teacher_dims)

        elif 'ViT-T-16' in model_name:
            student_dims = 192
            teacher_dims = 768
            self.align = nn.Linear(student_dims, teacher_dims, bias=True)

            self.mask_token = nn.Parameter(torch.zeros(1, 1, teacher_dims))
            self.generation = SEModule(teacher_dims)
        else:
            self.align = None

    def forward(self, preds):
        """Forward function.
        Args:
            preds_S(List): [B*N*D], student's feature map
            preds_T(List): [B*N*D], teacher's feature map
        """
        gen_variable = {}
        high = preds

        '''ViTKD: Generation'''
        # 学生教师的维度对齐
        if self.align is not None:
            x = self.align(high)
            # Mask tokens
            # Batch_size\ patch \ 维度
            B, N, D = x.shape
            x, mat, ids, ids_masked = self.random_masking(x, self.lambda_vitkd)
            mask_tokens = self.mask_token.repeat(B, N - x.shape[1], 1)
            x_ = torch.cat([x, mask_tokens], dim=1)
            x = torch.gather(x_, dim=1, index=ids.unsqueeze(-1).repeat(1, 1, D))
            mask = mat.unsqueeze(-1)

            hw = int(N ** 0.5)
            x = x.reshape(B, hw, hw, D).permute(0, 3, 1, 2)
            x = self.generation(x).flatten(2).transpose(1, 2)
            gen_variable['tokens'] = x
            gen_variable['mask'] = [mask]
            gen_variable['mask_ratio'] = self.lambda_vitkd
        else:
            gen_variable['tokens'] = high

        return gen_variable

    def random_masking(self, x, mask_ratio):
        N, L, D = x.shape  # batch, length, dim
        # 计算保留的长度
        len_keep = int(L * (1 - mask_ratio))

        # 生成随机噪声，生成一个矩阵(里面的数字为0到1的随机数)
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        # 对每个样本的噪声排序
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        # 恢复原始索引
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset 选择保留和被掩码的tokens
        # 包含要保留的tokens的索引
        ids_keep = ids_shuffle[:, :len_keep]
        # 包含被掩码的tokens的索引
        ids_masked = ids_shuffle[:, len_keep:L]

        # 提取保留的tokens
        x_keep = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        # 生成二进制掩码
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        # 使用 ids_restore 将掩码矩阵调整为与原始顺序相对应的形状。
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_keep, mask, ids_restore, ids_masked

# class gen_txt_EmbTrans(nn.Module):
#     def __init__(self, model_name):
#         super(gen_txt_EmbTrans, self).__init__()
#         self.model_name = model_name
#         # 解码器层数
#         self.num_layers = 2
#         # 解码器的config
#
#
#
#         if self.model_name in ['ViT-M-16']:
#             student_dims = 512
#             teacher_dims = 512
#             # 对齐层，将学生模型维度映射到教师模型维度
#             self.align = nn.Linear(student_dims, teacher_dims, bias=True)
#             # Mask token初始化
#             self.mask_token = nn.Parameter(torch.zeros(1, 1, teacher_dims))  # [1, 1, teacher_dims]
#             # 生成模块
#             self.generation = Decoder(embed_dim=teacher_dims, num_heads=8, dropout=0.1, num_layers=self.num_layers)
#         elif self.model_name in ['ViT-S-16', 'ViT-T-16']:
#             student_dims = 384
#             teacher_dims = 512
#             self.align = nn.Linear(student_dims, teacher_dims, bias=True)
#             self.mask_token = nn.Parameter(torch.zeros(1, 1, teacher_dims))  # [1, 1, teacher_dims]
#             self.generation = Decoder(embed_dim=teacher_dims, num_heads=8, dropout=0.1, num_layers=self.num_layers)
#         else:
#             self.align = None
#
#     def forward(self, text, feature):
#         """ 输入text是token序列，形状为 [B, N]，输入feature是学生模型的特征，形状为 [B, N, D] """
#
#         gen_variable = {}
#
#         if self.align is not None:
#             if self.model_name in ['ViT-S-16', 'ViT-T-16']:
#                 feature = self.align(feature)
#             masked_feature, mask = self.random_masking(text, feature, mask_ratio=self.lambda_vitkd)
#             # 将 tgt_mask 传入解码器
#             output = self.generation(masked_feature)
#
#             gen_variable['tokens'] = output
#             gen_variable['mask'] = mask
#             gen_variable['mask_ratio'] = self.lambda_vitkd
#         else:
#             gen_variable['tokens'] = feature
#
#         return gen_variable
#
