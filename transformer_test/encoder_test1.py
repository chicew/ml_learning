import torch
import math
import torch.nn as nn
import torch.nn.functional as F

# 维度符号说明(和图上一致):
#   batch : 一次处理几条序列
#   seq   : 序列长度,图上的 ds
#   d_m   : 每个 token 的向量维度,图上的 de
#   h     : 头数 num_head
#   d_head: 每个头的维度 = d_m / h,图上的 dq = dk = dv


def dotproduct_function(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """用于计算Q,K,V的注意力得分
    Q, K, V: (..., seq, d_head)   "..." 可以是 (batch,) 或 (batch, h)
    返回:    (..., seq, d_head)
    """
    dk: int = K.size(-1)  #读取最后一个维度,即 d_head
    # K.transpose(-2, -1): (..., seq, d_head) → (..., d_head, seq)
    # Q @ Kᵀ:              (..., seq, d_head) @ (..., d_head, seq) → (..., seq, seq)
    scale_dp = Q @ K.transpose(-2, -1)/math.sqrt(dk) #transpose可以用来交换维度   (..., seq, seq)
    # softmax 沿最后一维,每行和为 1:(..., seq, seq)
    # @ V: (..., seq, seq) @ (..., seq, d_head) → (..., seq, d_head)
    out = F.softmax(scale_dp, dim=-1) @ V
    return out

class multihead_attention (nn.Module):
    """一个多头注意力块
    输入 x: (batch, seq, d_m)  →  输出: (batch, seq, d_m)
    """
    def __init__(self, d_m: int, num_head: int) -> None:
        super().__init__()
        assert d_m % num_head == 0, "d_model 必须能被 n_head 整除"
        self.n_head: int = num_head
        self.d_m: int = d_m
        self.d_head: int = d_m // num_head      # 每个头的维度 d_m / h

        # 每个都是 d_m → d_m,权重形状 (d_m, d_m) = h 个头的小矩阵拼在一起
        self.W_Q = nn.Linear(d_m, d_m, bias= True)
        self.W_K = nn.Linear(d_m, d_m, bias= True)
        self.W_V = nn.Linear(d_m, d_m, bias= True)
        self.W_o = nn.Linear(d_m, d_m)          # Concat 之后的那个 Linear

    def split_heads(self, M: torch.Tensor) -> torch.Tensor:
        # M: (batch, seq, d_m)
        batch, seq, _ = M.shape                                 # 两个都是 int
        M = M.view(batch, seq, self.n_head, self.d_head)        # (batch, seq, h, d_head)
        return M.transpose(1, 2)                                # (batch, h, seq, d_head)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq, d_m)
        batch, seq, _ = x.shape
        Q = self.split_heads(self.W_Q(x))       # W_Q(x): (batch, seq, d_m) → split 后 (batch, h, seq, d_head)
        K = self.split_heads(self.W_K(x))       # (batch, h, seq, d_head)
        V = self.split_heads(self.W_V(x))       # (batch, h, seq, d_head)
        out = dotproduct_function(Q, K ,V)      # (batch, h, seq, d_head)

        # transpose(1, 2): (batch, h, seq, d_head) → (batch, seq, h, d_head)
        # view:            → (batch, seq, d_m)    这一步就是 Concat
        out = out.transpose(1, 2).contiguous().view(batch, seq, self.d_m)

        return self.W_o(out)                    # (batch, seq, d_m)

class MLP_net (nn.Module):
    """前馈神经网络
    输入 x: (batch, seq, d_m)  →  输出: (batch, seq, d_m)
    """
    def __init__(self, d_m: int, d_f: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_m, d_f, bias= True),   # (batch, seq, d_m) → (batch, seq, d_f)
                                 nn.ReLU(),                         # 形状不变
                                 nn.Linear(d_f, d_m, bias=True),)   # (batch, seq, d_f) → (batch, seq, d_m)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)                      # (batch, seq, d_m)

class encoder_block (nn.Module):
    """包含一个多头注意力机制的编码块
    输入 x: (batch, seq, d_m)  →  输出: (batch, seq, d_m),形状不变所以可以叠很多层
    """
    def __init__(self, d_m: int, num_head: int, d_f: int) -> None:
        super().__init__()
        self.multihead_attention = multihead_attention(d_m, num_head)
        self.MLP_net = MLP_net(d_m, d_f)
        self.norm1 = nn.LayerNorm(d_m)          # 对每个 token 的 d_m 维向量做归一化
        self.norm2 = nn.LayerNorm(d_m)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq, d_m)
        x = self.norm1(x + self.multihead_attention(x))   # 注意力 + 残差 + 归一化,(batch, seq, d_m)
        x = self.norm2(x + self.MLP_net(x))                       # MLP + 残差 + 归一化,  (batch, seq, d_m)

        return x

class encoder (nn.Module):
    """n_block 个编码块首尾相接
    输入 x: (batch, seq, d_m)  →  输出: (batch, seq, d_m)
    """
    def __init__(self, d_m: int, num_head: int, d_f: int, n_block: int) -> None:
        super().__init__()
        self.block_net = nn.ModuleList([encoder_block(d_m, num_head, d_f) for _ in range(n_block)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block_net in self.block_net:
            x = block_net(x)                    # 每一块都是 (batch, seq, d_m) → (batch, seq, d_m)
        return x

if __name__ == "__main__":
    torch.manual_seed(0)   # 固定随机种子,每次运行结果都一样,方便对照

    batch, seq_len, d_model, n_head = 16, 256, 256, 4
    d_head = d_model // n_head   # 256 / 4 = 64
    d_ff = 4 * d_model           # 1024

    # 随机造一个输入 X:16 条序列,每条 256 个 token,每个 token 是 256 维向量
    x = torch.randn(batch, seq_len, d_model)                      # (16, 256, 256)
    print(f"输入 x:                      {tuple(x.shape)}   ← (batch, ds, de)")

    # ── 测试 1:缩放点积注意力 ──────────────────────────────────────
    q = torch.randn(batch, seq_len, d_head)                       # (16, 256, 64)
    out= dotproduct_function(q, q, q)                             # (16, 256, 64)
    print(f"① 注意力输出:                {tuple(out.shape)}   ← (batch, ds, d_head)")


    mha_fast = multihead_attention(d_model, n_head)
    print(f"②-B 多头注意力(向量化)输出:  {tuple(mha_fast(x).shape)}   ← (batch, ds, de)")


    # ── 测试 5:MLP ─────────────────────────────────────────────────
    mlp = MLP_net(d_model, d_ff)
    print(f"③ MLP 输出:                  {tuple(mlp(x).shape)}   ← (batch, ds, de)")

    # ── 测试 6:编码器块 ────────────────────────────────────────────
    block = encoder_block(d_model, n_head, d_ff)
    y = block(x)                                                  # (16, 256, 256)
    print(f"④ 编码器块输出:              {tuple(y.shape)}   ← (batch, ds, de),和输入一样")
    assert y.shape == x.shape

    # ── 测试 7:完整编码器(叠 3 层)────────────────────────────────
    encoder = encoder(d_model, n_head, d_ff, 3)
    z = encoder(x)                                                # (16, 256, 256)
    print(f"⑤ 3 层编码器输出:            {tuple(z.shape)}   ← (batch, ds, de)")
    assert z.shape == x.shape

    # ── 顺便看看一个编码器块里有哪些参数、各是什么形状 ──────────────
    print("\n一个编码器块里的全部参数:")
    for name, p in block.named_parameters():
        print(f"   {name:<22} {tuple(p.shape)}")
    n_params = sum(p.numel() for p in block.parameters())
    print(f"   合计 {n_params} 个参数")
