import torch
import math
import torch.nn as nn
import torch.nn.functional as F

# CLIP BPE 词表中的 <start_of_text> token
BOS_TOKEN_ID = 49406
# 文本生成解码器的层数
DECODER_LAYERS = 2
# 与 CLIP 约定保持一致：每个 head 的维度固定为 64
HEAD_DIM = 64


# === 多头注意力（带 mask）===
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        # d_model 输入维度
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model  # 输入维度
        self.num_heads = num_heads  # 头数
        self.d_k = d_model // num_heads  # 每个 head 的维度

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.fc_out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def build_encoder_mask(self, encode_length, max_len, device):
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

    def forward(self, query, key, value, len=None):
        B, L, _ = query.size()

        # 线性映射并分头
        Q = self.W_q(query).view(B, L, self.num_heads, self.d_k).transpose(1, 2)  # [B, h, L, d_k]
        K = self.W_k(key).view(B, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(B, -1, self.num_heads, self.d_k).transpose(1, 2)

        # 注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)  # [B, h, L, L_k]
        if len is not None:
            # mask 长度与 key 的序列长度保持一致，避免写死上下文长度
            mask = self.build_encoder_mask(len + 1, K.shape[2], scores.device)
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, V)  # [B, h, L, d_k]

        # 拼接 heads
        out = out.transpose(1, 2).contiguous().view(B, L, self.d_model)
        return self.fc_out(out)


# === 前馈网络 ===
class FeedForward(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model)
        )

    def forward(self, x):
        return self.net(x)


class AddNorm(nn.Module):
    """残差连接后进行层规范化"""

    def __init__(self, normalized_shape, dropout, **kwargs):
        super(AddNorm, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)  # 暂退法 用于正则化(防止过拟合)
        self.ln = nn.LayerNorm(normalized_shape)  # 层规范化 输入的(时间步，特征维度)，与Batch_size没有关系

    def forward(self, X, Y):
        return self.ln(self.dropout(Y) + X)  # X为原始输入，Y为经过操作后的X


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=77):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 预先计算所有位置的正弦/余弦编码 [max_len, d_model]
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # [max_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)  # 不参与训练参数更新

    def forward(self, x, step=None):
        """
        参数:
            x: [batch_size, seq_len, d_model]  输入的embedding向量
            step: int 或 None
                - 如果是 None: 表示训练阶段，整段输入，按 seq_len 加位置编码
                - 如果是 int: 表示推理阶段，只取对应 step 的位置编码
        返回:
            加了位置编码的 x
        """
        if step is None:
            # === 训练阶段 ===
            x = x + self.pe[:, :x.size(1)]
        else:
            # === 推理阶段（逐步解码）===
            # x: [batch_size, 1, d_model]
            pos_emb = self.pe[:, step:step + 1, :]  # 取第 step 个位置
            x = x + pos_emb

        return self.dropout(x)


class DecoderBlock(nn.Module):
    """解码器中第i个块"""

    def __init__(self, num_hiddens, num_heads, dropout, i, **kwargs):
        super(DecoderBlock, self).__init__(**kwargs)
        self.i = i
        self.attention1 = MultiHeadAttention(num_hiddens, num_heads, dropout)  # 掩蔽多头注意力
        self.addnorm1 = AddNorm(num_hiddens, dropout)  # 残差连接和层规范化
        self.attention2 = MultiHeadAttention(num_hiddens, num_heads, dropout)  # 多头注意力
        self.addnorm2 = AddNorm(num_hiddens, dropout)  # 残差连接和层规范化
        self.ffn = FeedForward(num_hiddens)  # 基于位置的前馈网络
        self.addnorm3 = AddNorm(num_hiddens, dropout)  # 残差连接和层规范化

        # 不训练
        self.training = False

    def forward(self, X, state):
        """
        :param X: [B,tokens,dims]  其中一个样本的形式 [<bos>, token1, token2, token3, pad, pad, ...]
        :param state:
        :return:
        """
        # 编码器的输出  文本的有效长度
        enc_outputs, enc_valid_lens = state[0], state[1]

        # state每一个batch会重置
        if state[2][self.i] is None:  # 训练阶段
            key_values = X
        else:  # 推理阶段
            key_values = torch.cat((state[2][self.i], X), axis=1)
        state[2][self.i] = key_values  # 存储当前时间步之前和该时间步的输出表示
        if self.training:
            batch_size, num_steps, _ = X.shape
            # dec_valid_lens的开头:(batch_size,num_steps),
            # 其中每一行是[1,2,...,num_steps]
            dec_valid_lens = torch.arange(
                1, num_steps + 1, device=X.device).repeat(batch_size, 1)
        else:
            dec_valid_lens = None  # 解码器在预测阶段不需要进行mask ，因为本身是没有下一步的token的，但是在训练阶段，是有全部的token的信息的，因此需要进行遮挡
        # 自注意力
        # dec_valid_lens
        X2 = self.attention1(X, key_values, key_values, dec_valid_lens)
        Y = self.addnorm1(X, X2)
        # 编码器－解码器注意力。
        # enc_outputs的开头:(batch_size,num_steps,num_hiddens)
        Y2 = self.attention2(Y, enc_outputs, enc_outputs, enc_valid_lens)
        Z = self.addnorm2(Y, Y2)
        return self.addnorm3(Z, self.ffn(Z)), state


# === 整个解码器 ===
class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, d_model=512, num_layers=2, num_heads=8, dropout=0.1, context_length=77):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, dropout=dropout, max_len=context_length)
        self.num_layers = num_layers
        self.layers = nn.ModuleList([
            DecoderBlock(d_model, num_heads, dropout, i)
            for i in range(num_layers)
        ])
        self.lm_head = nn.Linear(d_model, vocab_size)

    # 初始化编码器中的state
    def init_state(self, enc_outputs, enc_valid_lens):
        return [enc_outputs, enc_valid_lens, [None] * self.num_layers]

    def forward(self, X, state, step):
        # X: [B] = [1]
        x = self.embedding(X).unsqueeze(1)  # [B, 1, d_model]
        x = self.pos_encoding(x, step)  # [B, 1, d_model]

        for layer in self.layers:
            x, state = layer(x, state)  # 保证维度一致

        logits = self.lm_head(x)  # [B, 1, vocab_size]
        return logits


class MLM(nn.Module):
    def __init__(self, args):
        super(MLM, self).__init__()
        self.args = args
        # 解码器维度必须与学生文本编码器一致（ViT-S/T-16 为 384，ViT-M-16 为 512）
        d_model = args.token_txt
        num_heads = max(1, d_model // HEAD_DIM)
        self.max_logits_len = args.s_context_length - 1
        self.decoder = TransformerDecoder(
            vocab_size=args.s_vocab_size,
            d_model=d_model,
            num_layers=DECODER_LAYERS,
            num_heads=num_heads,
            context_length=args.s_context_length)
        self.decoder.apply(self.xavier_init_weights)

    @staticmethod
    def xavier_init_weights(m):
        # 处理 Linear 层
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

        # 处理 GRU 层
        elif isinstance(m, nn.GRU):
            for name, param in m.named_parameters():
                if "weight" in name:
                    nn.init.xavier_uniform_(param.data)
                elif "bias" in name:
                    nn.init.zeros_(param.data)

    def pad_logits(self, logit, max_len=76):
        """
        Args:
            logit: [B, L, vocab_size]
            max_len: int, 目标长度
        Returns:
            padded_logits: [B, max_len, vocab_size]
        """
        B, L, vocab_size = logit.size()
        if L == max_len:
            return logit
        elif L < max_len:
            pad_len = max_len - L
            # 补零 [B, pad_len, vocab_size]
            pad = torch.zeros(B, pad_len, vocab_size, device=logit.device)
            padded_logits = torch.cat([logit, pad], dim=1)
            return padded_logits
        else:
            # 超长就截断
            return logit[:, :max_len, :]

    def forward(self, encode_output, encode_length):
        # encode_length 以0为起始
        # 解码器的输出以目标长度为标准
        # encode_output.shape [batch, context_length, d_model]
        # 第一层for循环是遍历批次
        logits = []
        for i in range(encode_output.shape[0]):
            # 初始化编码器的sate
            state = self.decoder.init_state(encode_output[i], encode_length[i])
            seq = [BOS_TOKEN_ID]
            logit = []
            # 第二层for循环是遍历批次中的每个样本的长度
            for j in range(encode_length[i]):
                current_token = torch.tensor([seq[-1]], device=encode_output.device)  # [1]
                next_token_logits = self.decoder(current_token, state, j)  # [1, 1, vocab_size]
                logit.append(next_token_logits)
                next_token = next_token_logits[:, -1, :].argmax(dim=-1).item()
                seq.append(next_token)  # [1, len]
            logit = torch.cat(logit, dim=1)  # [B, L-1, vocab_size] 本身不包含<bos>
            logit = self.pad_logits(logit, self.max_logits_len)
            logits.append(logit)
        logits = torch.cat(logits, dim=0)
        return logits
