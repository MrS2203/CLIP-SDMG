import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

# 引入我们的蒸馏损失
from src.loss.PGS_loss import pgs_loss
from src.loss.DAB_loss import dab_loss
from src.loss.GEN_loss import gen_loss


try:
    import torch.distributed.nn
    from torch import distributed as dist

    has_distributed = True
except ImportError:
    has_distributed = False

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None


class ClipLoss(nn.Module):

    def __init__(
            self,
            local_loss=False,
            gather_with_grad=False,
            cache_labels=False,
            rank=0,
            world_size=1,
            use_horovod=False,
    ):
        super().__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        self.use_horovod = use_horovod

        # cache state
        self.prev_num_logits = 0
        self.labels = {}

    def forward(self, image_features, text_features, logit_scale):
        device = image_features.device
        logits_per_image = logit_scale * image_features @ text_features.T
        logits_per_text = logit_scale * text_features @ image_features.T

        # calculated ground-truth and cache if enabled
        num_logits = logits_per_image.shape[0]
        if self.prev_num_logits != num_logits or device not in self.labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
            if self.world_size > 1 and self.local_loss:
                labels = labels + num_logits * self.rank
            if self.cache_labels:
                self.labels[device] = labels
                self.prev_num_logits = num_logits
        else:
            labels = self.labels[device]
        total_loss = (
                             F.cross_entropy(logits_per_image, labels) +
                             F.cross_entropy(logits_per_text, labels)
                     ) / 2

        return total_loss


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


class KDClipLoss(nn.Module):

    def __init__(
            self,
            args,
            local_loss=False,
            gather_with_grad=False,
            cache_labels=False,
            use_horovod=False,
    ):
        super().__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.use_horovod = use_horovod
        self.args = args

        if args.t_embed_dim != args.s_embed_dim:
            self.visual_proj = nn.Linear(args.s_embed_dim, args.t_embed_dim)
            self.text_proj = nn.Linear(args.s_embed_dim, args.t_embed_dim)

        if args.patch_vis != args.t_patch_vis:
            self.visual_cls_vis = nn.Linear(args.patch_vis, args.t_patch_vis)
        if args.token_txt != args.t_token_txt:
            self.text_cls_txt = nn.Linear(args.token_txt, args.t_token_txt)

        # cache state
        self.prev_num_logits = 0
        self.kl_loss = DistillKL(T=1)
        self.cross_logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.fusion_logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.labels = {}

        # 全局语义渐进式蒸馏损失
        self.pgs_loss = pgs_loss(args)
        # 浅层局部语义蒸馏损失
        self.dab_loss = dab_loss(args)
        # 深层局部语义掩码生成损失
        self.gen_loss = gen_loss(args)

    def forward(self, image_features, text_features, logit_scale,
                t_image_features, t_text_features, t_logit_scale,
                intermediate_features, t_intermediate_features,
                txt_intermediate_features, t_txt_intermediate_features,
                attention_weights, t_attention_weights, texts, epoch):

        device = image_features.device

        # 学生模型的相似度矩阵计算
        normalized_image_features = F.normalize(image_features, dim=1)
        normalized_text_features = F.normalize(text_features, dim=1)
        logits_per_image = logit_scale * normalized_image_features @ normalized_text_features.T
        logits_per_text = logit_scale * normalized_text_features @ normalized_image_features.T

        # calculated ground-truth and cache if enabled
        num_logits = logits_per_image.shape[0]
        if self.prev_num_logits != num_logits or device not in self.labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
            if self.cache_labels:
                self.labels[device] = labels
                self.prev_num_logits = num_logits
        else:
            labels = self.labels[device]

        if self.args.t_embed_dim != self.args.s_embed_dim:
            image_features = self.visual_proj(image_features)
            text_features = self.text_proj(text_features)

        normalized_all_image_features = F.normalize(image_features, dim=1)
        normalized_all_text_features = F.normalize(text_features, dim=1)

        # 响应蒸馏损失
        fd_loss = F.mse_loss(normalized_all_image_features, t_image_features) + \
                  F.mse_loss(normalized_all_text_features, t_text_features)

        # CLIP的任务损失
        task_loss = (F.cross_entropy(logits_per_image, labels) + F.cross_entropy(logits_per_text, labels)) / 2

        # 渐进式全局语义蒸馏损失
        pgs_loss = torch.tensor(0.).cuda()
        # 基于注意权重的浅层模仿
        dab_loss = torch.tensor(0.).cuda()
        # 深层掩码生成
        gen_loss = torch.tensor(0.).cuda()

        if self.args.alpha_pgs_loss > 0.:
            pgs_loss = self.pgs_loss(intermediate_features, t_intermediate_features,
                                     txt_intermediate_features, t_txt_intermediate_features, texts)
        # vit0、1、6部分模仿
        if self.args.alpha_dab_loss > 0.:
            dab_loss = self.dab_loss(intermediate_features, t_intermediate_features,
                                     txt_intermediate_features, t_txt_intermediate_features,
                                     attention_weights, t_attention_weights, epoch, texts)

        # vit11部分生成
        if self.args.alpha_gen_loss > 0.:
            gen_loss =self.gen_loss(intermediate_features, t_intermediate_features,
                                    txt_intermediate_features, texts)

        fd_loss = self.args.alpha_fd_loss * fd_loss
        dab_loss = self.args.alpha_dab_loss * dab_loss
        pgs_loss = self.args.alpha_pgs_loss * pgs_loss
        gen_loss = self.args.alpha_gen_loss * gen_loss

        return task_loss, fd_loss, gen_loss, pgs_loss, dab_loss