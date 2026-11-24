import numpy as np
import torch


def get_metrics(image_features, text_features, logit_scale, chunk_size=512):
    """分块计算图文检索指标，避免一次性构造 N x N 相似度矩阵。

    Args:
        image_features: [N, D] 图像特征（CPU）
        text_features: [N, D] 文本特征（CPU）
        logit_scale: 缩放因子（标量或 [1]）
        chunk_size: 每次参与计算的 query 数量，控制峰值内存
    """
    metrics = {}
    num_samples = len(text_features)
    if num_samples == 0:
        return metrics
    ground_truth = torch.arange(num_samples).view(-1, 1)

    # 图像到文本、文本到图像两个方向的检索指标
    for name, (queries, keys) in {
        "image_to_text": (image_features, text_features),
        "text_to_image": (text_features, image_features),
    }.items():
        preds = []
        for start in range(0, num_samples, chunk_size):
            end = min(start + chunk_size, num_samples)
            # 只保留 chunk_size x N 的相似度矩阵
            logits = logit_scale * queries[start:end] @ keys.t()
            ranking = torch.argsort(logits, descending=True)
            preds.append(torch.where(ranking == ground_truth[start:end])[1].numpy())
        preds = np.concatenate(preds)

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
