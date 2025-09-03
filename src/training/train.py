import json
import logging
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

try:
    import wandb
except ImportError:
    wandb = None

from src.open_clip import ClipLoss, KDClipLoss, get_cast_dtype
from src.training.zero_shot import zero_shot_eval
from src.training.precision import get_autocast


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def unwrap_model(model):
    if hasattr(model, 'module'):
        return model.module
    else:
        return model


def train_one_epoch(model, data, epoch, optimizer, scaler, scheduler, args, tb_writer=None):
    device = torch.device(args.device)
    autocast = get_autocast(args.precision)
    cast_dtype = get_cast_dtype(args.precision)

    # 训练模型
    model.train()
    loss = ClipLoss(
        local_loss=args.local_loss,
        gather_with_grad=args.gather_with_grad,
        cache_labels=True,
    )

    data['train'].set_epoch(epoch)  # set epoch in process safe manner via sampler or shared_epoch
    dataloader = data['train'].dataloader
    num_batches_per_epoch = dataloader.num_batches
    sample_digits = math.ceil(math.log(dataloader.num_samples + 1, 10))

    loss_m = AverageMeter()
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()  # 加载数据时间
    end = time.time()
    for i, batch in enumerate(dataloader):
        step = num_batches_per_epoch * epoch + i

        if not args.skip_scheduler:
            # 学习率衰减
            scheduler(step)

        # images [B,C,H,W]
        # texts [B, 文本长度(77)]
        images, texts = batch
        images = images.to(device=device, dtype=cast_dtype, non_blocking=True)
        texts = texts.to(device=device, non_blocking=True)

        data_time_m.update(time.time() - end)
        # 每个Batch_size梯度归零
        optimizer.zero_grad()

        with autocast():
            """
            image_features [B, 嵌入维度]
            text_features [B, 嵌入维度]
            logit_scale 缩放因子(可学习)
            """
            image_features, _, text_features, _, logit_scale = model(images, texts)
            # CLIP经典损失
            total_loss = loss(image_features, text_features, logit_scale)

        # 是否使用梯度缩放
        if scaler is not None:
            # 使用AMP进行损失缩放并反向传播
            scaler.scale(total_loss).backward()

            # 如果设置了梯度裁剪, 先取消缩放再裁剪梯度
            if args.grad_clip_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)

            # 使用AMP进行优化器更新
            scaler.step(optimizer)

            # 更新梯度缩放器，准备下一轮训练
            scaler.update()
        else:
            # 不使用AMP，直接反向传播
            total_loss.backward()

            # 如果设置了梯度裁剪，进行裁剪
            if args.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)

            # 优化器更新
            optimizer.step()

        # Note: we clamp to 4.6052 = ln(100), as in the original paper.
        with torch.no_grad():
            unwrap_model(model).logit_scale.clamp_(0, math.log(100))

        # 查看梯度
        for name, param in model.named_parameters():
            if param.grad is None:
                print(f"Grad for {name}: {param.grad}")

        # 处理一个Batch的时间
        batch_time_m.update(time.time() - end)
        end = time.time()
        batch_count = i + 1
        if (i % 5 == 0 or batch_count == num_batches_per_epoch):
            batch_size = len(images)
            num_samples = batch_count * batch_size
            samples_per_epoch = dataloader.num_samples
            percent_complete = 100.0 * batch_count / num_batches_per_epoch

            # NOTE loss is coarsely sampled, just master node and per log update
            loss_m.update(total_loss.item(), batch_size)
            logit_scale_scalar = logit_scale.item()

            # 获取显存情况
            current_memory_allocated = torch.cuda.memory_allocated() / (1024 ** 3)  # GB
            current_memory_reserved = torch.cuda.memory_reserved() / (1024 ** 3)  # GB

            logging.info(
                f"Train Epoch: {epoch} [{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_complete:.0f}%)] "
                f"Loss: {loss_m.val:#.5g} ({loss_m.avg:#.4g}) "
                f"Data (t): {data_time_m.avg:.3f} "
                f"Batch (t): {batch_time_m.avg:.3f}, {args.batch_size / batch_time_m.val:#g}/s "
                f"lr: {optimizer.param_groups[0]['lr']:5f} "
                f"Logit Scale: {logit_scale_scalar:.3f}"
                f"Memory Allocated: {current_memory_allocated:.2f} GB "
                f"Memory Reserved: {current_memory_reserved:.2f} GB"
            )

            # Save train loss / etc. Using non avg meter values as loggers have their own smoothing
            log_data = {
                "loss": loss_m.val,
                "data_time": data_time_m.val,
                "batch_time": batch_time_m.val,
                "samples_per_scond": args.batch_size / batch_time_m.val,
                "scale": logit_scale_scalar,
                "lr": optimizer.param_groups[0]["lr"]
            }
            for name, val in log_data.items():
                name = "train/" + name
                if tb_writer is not None:
                    tb_writer.add_scalar(name, val, step)

            # resetting batch / data time meters per log window
            batch_time_m.reset()
            data_time_m.reset()


def train_kd_one_epoch(model, t_model, data, epoch, loss, optimizer, scaler, scheduler, args, tb_writer=None):
    # 设备选择
    device = torch.device(args.device)
    # 精度选择
    autocast = get_autocast(args.precision)
    cast_dtype = get_cast_dtype(args.precision)

    model.train()

    data['train'].set_epoch(epoch)  # set epoch in process safe manner via sampler or shared_epoch
    # 获取训练数据类
    dataloader = data['train'].dataloader
    # 每一轮epoch的总batch数量
    num_batches_per_epoch = dataloader.num_batches
    sample_digits = math.ceil(math.log(dataloader.num_samples + 1, 10))

    # 各损失记录
    loss_m = AverageMeter()
    loss_task = AverageMeter()
    loss_fd = AverageMeter()
    loss_dab = AverageMeter()
    loss_pgs = AverageMeter()
    loss_gen = AverageMeter()
    # batch加载和处理时间
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()
    for i, batch in enumerate(dataloader):
        step = num_batches_per_epoch * epoch + i

        # 调整学习率
        if not args.skip_scheduler:
            scheduler(step)

        # 获取batch的图像和文本 移到GPU
        images, texts = batch
        images = images.to(device=device, dtype=cast_dtype, non_blocking=True)
        texts = texts.to(device=device, non_blocking=True)

        data_time_m.update(time.time() - end)
        optimizer.zero_grad()

        with autocast():
            """
            image_features [batch, 512]
            intermediate_features 视觉编码器的12层的特征列表 每一层的形状为  [batch, patch+1 (197), 维度]
            text_features [batch, 512]
            txt_intermediate_features 文本编码器的12层的特征列表 每一层的形状为  [batch, 77, 维度]
            """

            """
            attention_weights is list =  [attention_weights, txt_attention_weights]
            attention_weights 视觉编码器的12层的注意力列表  每一层的形状为  [batch, patch+1 (197), patch+1 (197)]
            txt_attention_weights 文本编码器的12层的注意力列表  每一层的形状为  [batch, 77, 77]
            """
            image_features, intermediate_features, text_features, txt_intermediate_features, logit_scale, attention_weights = model(
                images,
                texts,
                distill=True)

            # 因为教师模型不需要更新 因此不需要计算梯度
            with torch.no_grad():
                t_image_features, t_intermediate_features, t_text_features, t_txt_intermediate_features, t_logit_scale, t_attention_weights = t_model(
                    images, texts)

            # 损失函数
            losses = loss(image_features, text_features, logit_scale,  # 学生模型最终的图像和文本特征以及logit_scale
                          t_image_features, t_text_features, t_logit_scale,  # 教师模型最终的图像和文本特征以及logit_scale
                          intermediate_features, t_intermediate_features,  # 学生和教师模型中间层的视觉特征
                          txt_intermediate_features, t_txt_intermediate_features,  # 学生和教师模型中间层的文本特征
                          attention_weights, t_attention_weights, texts,
                          epoch)  # 学生和教师模型中间层的注意力权重

            task_loss, fd_loss, gen_loss, pgs_loss, dab_loss = losses
            total_loss = task_loss + fd_loss + gen_loss + pgs_loss + dab_loss

        if scaler is not None:
            scaler.scale(total_loss).backward()
            if args.horovod:
                optimizer.synchronize()
                scaler.unscale_(optimizer)
                if args.grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
                with optimizer.skip_synchronize():
                    scaler.step(optimizer)
            else:
                if args.grad_clip_norm is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
                scaler.step(optimizer)
            scaler.update()
        else:
            total_loss.backward()
            if args.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
            optimizer.step()

        # Note: we clamp to 4.6052 = ln(100), as in the original paper.
        with torch.no_grad():
            unwrap_model(model).logit_scale.clamp_(0, math.log(100))


        batch_time_m.update(time.time() - end)
        end = time.time()
        batch_count = i + 1
        if (i % 100 == 0 or batch_count == num_batches_per_epoch):
            batch_size = len(images)
            num_samples = batch_count * batch_size
            samples_per_epoch = dataloader.num_samples
            percent_complete = 100.0 * batch_count / num_batches_per_epoch

            # NOTE loss is coarsely sampled, just master node and per log update
            loss_m.update(total_loss.item(), batch_size)
            loss_task.update(task_loss.item(), batch_size)
            loss_fd.update(fd_loss.item(), batch_size)
            loss_gen.update(gen_loss.item(), batch_size)
            loss_dab.update(dab_loss.item(), batch_size)
            loss_pgs.update(pgs_loss.item(), batch_size)
            logit_scale_scalar = logit_scale.item()
            logging.info(
                f"Train Epoch: {epoch} [{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_complete:.0f}%)] "
                f"Total Loss: {loss_m.val:#.5g} ({loss_m.avg:#.4g}) "
                f"Task(CLIP) Loss: {loss_task.val:#.5g} ({loss_task.avg:#.4g}) "
                f"FD Loss: {loss_fd.val:#.5g} ({loss_fd.avg:#.4g}) "
                f"pgs Loss: {loss_pgs.val:#.5g} ({loss_pgs.avg:#.4g}) "
                f"dab Loss: {loss_dab.val:#.5g} ({loss_dab.avg:#.4g}) "
                f"gen Loss: {loss_gen.val:#.5g} ({loss_gen.avg:#.4g}) "
                f"Data (t): {data_time_m.avg:.3f} "
                f"Batch (t): {batch_time_m.avg:.3f}, {args.batch_size / batch_time_m.val:#g}/s "
                f"LR: {optimizer.param_groups[0]['lr']:5f} "
                f"Logit Scale: {logit_scale_scalar:.3f}"
            )

            # Save train loss / etc. Using non avg meter values as loggers have their own smoothing
            log_data = {
                "loss": loss_m.val,
                "data_time": data_time_m.val,
                "batch_time": batch_time_m.val,
                "samples_per_scond": args.batch_size / batch_time_m.val,
                "scale": logit_scale_scalar,
                "lr": optimizer.param_groups[0]["lr"]
            }
            for name, val in log_data.items():
                name = "train/" + name
                if tb_writer is not None:
                    tb_writer.add_scalar(name, val, step)
                if args.wandb:
                    assert wandb is not None, 'Please install wandb.'
                    wandb.log({name: val, 'step': step})

            # resetting batch / data time meters per log window
            batch_time_m.reset()
            data_time_m.reset()


def evaluate(model, data, epoch, args, tb_writer=None):
    # 评估结果字典
    metrics = {}
    device = torch.device(args.device)
    # 评估模型
    model.eval()

    # 零样本评估 (包括imagenet的5个变体)
    zero_shot_metrics = zero_shot_eval(model, data, epoch, args)
    metrics.update(zero_shot_metrics)

    autocast = get_autocast(args.precision)
    cast_dtype = get_cast_dtype(args.precision)

    if 'val' in data and (args.val_frequency and ((epoch % args.val_frequency) == 0 or epoch == args.epochs)):
        dataloader = data['val'].dataloader
        num_samples = 0
        samples_per_val = dataloader.num_samples

        # FIXME this does not scale past small eval datasets
        # all_image_features @ all_text_features will blow up memory and compute very quickly
        cumulative_loss = 0.0
        all_image_features, all_text_features = [], []
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                images, texts = batch
                images = images.to(device=device, dtype=cast_dtype, non_blocking=True)
                texts = texts.to(device=device, non_blocking=True)

                with autocast():
                    # 模型推理
                    image_features, intermediate_features, text_features, txt_intermediate_features, logit_scale, an = model(
                        images, texts)
                    # features are accumulated in CPU tensors, otherwise GPU memory exhausted quickly
                    # however, system RAM is easily exceeded and compute time becomes problematic
                    all_image_features.append(image_features.cpu())
                    all_text_features.append(text_features.cpu())
                    logit_scale = logit_scale.mean()
                    # 计算相似度
                    logits_per_image = logit_scale * image_features @ text_features.t()
                    logits_per_text = logits_per_image.t()

                    batch_size = images.shape[0]
                    labels = torch.arange(batch_size, device=device).long()
                    total_loss = (  # 交叉熵损失
                                         F.cross_entropy(logits_per_image, labels) +
                                         F.cross_entropy(logits_per_text, labels)
                                 ) / 2

                # cross_entropy 本身算的就是平均损失
                cumulative_loss += total_loss * batch_size
                num_samples += batch_size
                # 第i个batch
                if (i % 100) == 0:
                    logging.info(
                        f"Eval Epoch: {epoch} [{num_samples} / {samples_per_val}]\t"
                        f"Loss: {cumulative_loss / num_samples:.6f}\t")

            # 验证模型性能
            val_metrics = get_metrics(
                # all_image_features 整个验证集的图像特征
                image_features=torch.cat(all_image_features),
                # all_text_features 整个验证集的文本特征
                text_features=torch.cat(all_text_features),
                # 整个验证集的平均缩放因子
                logit_scale=logit_scale.cpu(),
            )
            loss = cumulative_loss / num_samples
            metrics.update(
                {**val_metrics, "val_loss": loss.item(), "epoch": epoch, "num_samples": num_samples}
            )

    if not metrics:
        return metrics

    logging.info(
        f"Eval Epoch: {epoch} "
        + "\t".join([f"{k}: {round(v, 4):.4f}" for k, v in metrics.items()])
    )

    # 保存排名和召回率信息
    if args.save_logs:
        for name, val in metrics.items():
            if tb_writer is not None:
                tb_writer.add_scalar(f"val/{name}", val, epoch)

        with open(os.path.join(args.checkpoint_path, "results.jsonl"), "a+") as f:
            f.write(json.dumps(metrics))
            f.write("\n")

    return metrics


def get_metrics(image_features, text_features, logit_scale):
    metrics = {}
    # logits_per_image [B, B] 每张图像对每个文本的匹配得分 (等价于相似度  得分越高,相似度越高)
    # logits_per_text [B, B] 每个文本对每张图像的匹配得分
    logits_per_image = (logit_scale * image_features @ text_features.t()).detach().cpu()
    logits_per_text = logits_per_image.t().detach().cpu()
    logits = {"image_to_text": logits_per_image, "text_to_image": logits_per_text}

    # ground_trut [B, 1] 每个值表示文本的正确匹配索引
    ground_truth = torch.arange(len(text_features)).view(-1, 1)

    for name, logit in logits.items():
        """
        name:image_to_text, text_to_image
        logit: 得分矩阵 [B, B]
        """
        ranking = torch.argsort(logit, descending=True)
        preds = torch.where(ranking == ground_truth)[1]
        preds = preds.detach().cpu().numpy()
        metrics[f"{name}_mean_rank"] = preds.mean() + 1
        metrics[f"{name}_median_rank"] = np.floor(np.median(preds)) + 1
        for k in [1, 5, 10]:
            metrics[f"{name}_R@{k}"] = np.mean(preds < k)

    """
    平均排名和中位数排名都是越低越好  召回率越高越好
    image_to_text_mean_rank: 图像到文本匹配的平均排名
    image_to_text_median_rank: 图像到文本匹配的中位数排名
    image_to_text_R@1, R@5, R@10: 图像到文本匹配的前1、5、10名召回率

    text_to_image_mean_rank: 文本到图像匹配的平均排名。
    text_to_image_median_rank: 文本到图像匹配的中位数排名
    text_to_image_R@1, R@5, R@10: 文本到图像匹配的前1、5、10名召回率
    """
    return metrics
