import torch


def build_encoder_mask(encode_length, max_len, device):
    """
    根据 encode_length 构造 attention mask.
    Args:
        encode_length: int，当前样本的有效长度
        max_len: int，编码器输出的序列总长度（例如77）
        device: torch.device

    Returns:
        mask: [1, 1, 1, max_len] 的张量，1表示可见，0表示mask掉
    """
    # 初始化全1，表示所有位置可见
    mask = torch.ones(max_len, device=device)
    # 把超出 encode_length 的部分置0
    mask[encode_length:] = 0
    # 扩展到 [1,1,1,max_len]，能和 scores = [B,h,L,Lk] 对齐
    mask = mask.unsqueeze(0).unsqueeze(0).unsqueeze(0)
    return mask

if __name__ == '__main__':
    encode_length = 11
    max_len = 77
    device = torch.device("cuda")

    mask = build_encoder_mask(encode_length, max_len, device)
    print(mask.shape)  # torch.Size([1, 1, 1, 77])

    scores = torch.randn(1, 6, 1, 77, device=device)  # 模拟attention分数
    scores = scores.masked_fill(mask == 0, float('-inf'))
    print(scores.shape)