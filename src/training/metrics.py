import numpy as np
import torch


def get_metrics(image_features, text_features, logit_scale, chunk_size=512,
                compute_rank=True, k_list=(1, 5, 10)):
    """分块计算图文检索指标，避免一次性构造 N x N 相似度矩阵。

    Args:
        image_features: [N, D] 图像特征（CPU / GPU 均可）
        text_features: [N, D] 文本特征（CPU / GPU 均可）
        logit_scale: 缩放因子（标量或 [1]）
        chunk_size: 每次参与计算的 query 数量，控制峰值内存
        compute_rank: 是否计算 mean_rank / median_rank（需要全量 argsort）。
            False 时只算 R@k，改用纯 topk 路径，N=5000 量级可快 ~4-5x。
        k_list: 需要计算 R@k 的 k 值列表

    Returns:
        dict，key 形如 "image_to_text_R@1" / "text_to_image_mean_rank" 等
    """
    metrics = {}
    num_samples = len(text_features)
    if num_samples == 0:
        return metrics

    ground_truth = torch.arange(num_samples, device=text_features.device).view(-1, 1)

    for name, (queries, keys) in {
        "image_to_text": (image_features, text_features),
        "text_to_image": (text_features, image_features),
    }.items():
        if compute_rank:
            # ---- 完整路径：argsort + scatter 反转 + take（替换 torch.where）----
            preds = []
            for start in range(0, num_samples, chunk_size):
                end = min(start + chunk_size, num_samples)
                logits = logit_scale * queries[start:end] @ keys.t()  # [chunk, N]
                ranking = torch.argsort(logits, descending=True)  # [chunk, N]

                # scatter 反转 ranking → inv_ranking[i, k] = k 在第 i 行里的 rank
                inv_ranking = torch.empty_like(ranking)
                col_idx = torch.arange(ranking.shape[1], device=ranking.device).unsqueeze(0).expand_as(ranking)
                inv_ranking.scatter_(1, ranking, col_idx)

                # direct indexing: inv_ranking[batch_idx, gt] 就是 rank
                gt_flat = ground_truth[start:end].squeeze(1)  # [chunk]
                batch_idx = torch.arange(ranking.shape[0], device=ranking.device)
                preds.append(inv_ranking[batch_idx, gt_flat])
            # 延迟到最后一次性 numpy 转换
            preds_t = torch.cat(preds)
            preds = preds_t.cpu().numpy()

            metrics[f"{name}_mean_rank"] = preds.mean() + 1
            metrics[f"{name}_median_rank"] = np.floor(np.median(preds)) + 1
            for k in k_list:
                metrics[f"{name}_R@{k}"] = np.mean(preds < k)

        else:
            # ---- 快速路径：只算 R@k，用纯 topk（N log k vs N log N）----
            r_at_k_acc = {k: [] for k in k_list}
            max_k = max(k_list)
            for start in range(0, num_samples, chunk_size):
                end = min(start + chunk_size, num_samples)
                logits = logit_scale * queries[start:end] @ keys.t()  # [chunk, N]
                # topk 只需 O(N log max_k)
                _, top_idx = torch.topk(logits, k=max_k, dim=1)  # [chunk, max_k]
                gt_flat = ground_truth[start:end].squeeze(1)  # [chunk]
                for k in k_list:
                    hits = (top_idx[:, :k] == gt_flat.unsqueeze(1)).any(dim=1).float().mean()
                    r_at_k_acc[k].append(hits.item())
            for k in k_list:
                metrics[f"{name}_R@{k}"] = float(np.mean(r_at_k_acc[k]))

    return metrics
