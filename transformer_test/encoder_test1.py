import torch
import math
import torch.nn as nn
import torch.nn.functional as F



def dotproduct_function (Q, K, V):
    """用于计算Q,K,V的注意力得分"""
    dk = K.size(-1)
    scale_dp = Q @ K.transpose(-2, -1)/math.sqrt(dk)
    out = F.softmax(scale_dp, dim=-1) @ V
    return out

class multihead_attention (nn.Module):
    """一个多头注意力块"""
    def __init__(self, d_m, num_head):
        super().__init__()
        assert d_m % num_head == 0, "d_model 必须能被 n_head 整除"
        self.n_head = num_head
        self.d_m = d_m
        self.d_head = d_m // num_head

        self.W_Q = nn.Linear(d_m, d_m, bias= True)
        self.W_K = nn.Linear(d_m, d_m, bias= True)
        self.W_V = nn.Linear(d_m, d_m, bias= True)
        self.W_o = nn.Linear(d_m, d_m)

    def split_heads(self, M):
        batch, seq, _ = M.shape
        M = M.view(batch, seq, self.n_head, self.d_head)
        return M.transpose(1, 2)
    
    def forward(self, x):
        batch, seq, _ = x.shape
        Q = self.split_heads(self.W_Q(x))         
        K = self.split_heads(self.W_K(x))         
        V = self.split_heads(self.W_V(x))
        out = dotproduct_function(Q, K ,V)

        out = out.transpose(1, 2).contiguous().view(batch, seq, self.d_m)

        return self.W_o(out)

class MLP_net (nn.Module):
    """前馈神经网络"""
    def __init__(self,d_m, d_f):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_m, d_f, bias= True),
                                 nn.ReLU(),
                                 nn.Linear(d_f, d_m, bias=True),)

    def forward (self, x):
        return self.net(x)
        
class encoder_block (nn.Module):
    """包含一个多头注意力机制的编码块"""
    def __init__(self, d_m, num_head, d_f):
        super().__init__()
        self.multihead_attention = multihead_attention(d_m, num_head)
        self.MLP_net = MLP_net(d_m, d_f)
        self.norm1 = nn.LayerNorm(d_m)
        self.norm2 = nn.LayerNorm(d_m)

    def forward (self, x):
        x = self.norm1(x + self.multihead_attention.forward(x))
        x = self.norm2(x + self.MLP_net(x))

        return x

class encoder (nn.Module):
    def __init__(self, d_m, num_head, d_f, n_block):
        super().__init__()
        self.block_net = nn.ModuleList([encoder_block(d_m, num_head, d_f) for _ in range(n_block)])

    def forward (self, x):
        for block_net in self.block_net:
            x = block_net(x)
        return x

if __name__ == "__main__":
    torch.manual_seed(0)   # 固定随机种子,每次运行结果都一样,方便对照

    # 故意选几个互不相同的小数字:如果哪里把维度搞混了,形状马上就会对不上
    batch, seq_len, d_model, n_head = 16, 256, 256, 4
    d_head = d_model // n_head   # 8 / 4 = 2
    d_ff = 4 * d_model           # 32

    # 随机造一个输入 X:2 条序列,每条 5 个 token,每个 token 是 8 维向量
    x = torch.randn(batch, seq_len, d_model)
    print(f"输入 x:                      {tuple(x.shape)}   ← (batch, ds, de)")

    # ── 测试 1:缩放点积注意力 ──────────────────────────────────────
    q = torch.randn(batch, seq_len, d_head)
    out= dotproduct_function(q, q, q)
    print(f"① 注意力输出:                {tuple(out.shape)}   ← (batch, ds, d_head)")


    mha_fast = multihead_attention(d_model, n_head)
    print(f"②-B 多头注意力(向量化)输出:  {tuple(mha_fast(x).shape)}   ← (batch, ds, de)")


    # ── 测试 5:MLP ─────────────────────────────────────────────────
    mlp = MLP_net(d_model, d_ff)
    print(f"③ MLP 输出:                  {tuple(mlp(x).shape)}   ← (batch, ds, de)")

    # ── 测试 6:编码器块 ────────────────────────────────────────────
    block = encoder_block(d_model, n_head, d_ff)
    y = block(x)
    print(f"④ 编码器块输出:              {tuple(y.shape)}   ← (batch, ds, de),和输入一样")
    assert y.shape == x.shape

    # ── 测试 7:完整编码器(叠 3 层)────────────────────────────────
    encoder = encoder(d_model, n_head, d_ff, 3)
    z = encoder(x)
    print(f"⑤ 3 层编码器输出:            {tuple(z.shape)}   ← (batch, ds, de)")
    assert z.shape == x.shape

    # ── 顺便看看一个编码器块里有哪些参数、各是什么形状 ──────────────
    print("\n一个编码器块里的全部参数:")
    for name, p in block.named_parameters():
        print(f"   {name:<22} {tuple(p.shape)}")
    n_params = sum(p.numel() for p in block.parameters())
    print(f"   合计 {n_params} 个参数")    



        

        