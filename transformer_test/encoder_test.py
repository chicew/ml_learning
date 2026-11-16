"""
Transformer 编码器块(Encoder Block)—— 教学版,对应你画的那张框架图

═══════════════════════════════════════════════════════════════════════
写这类代码的固定套路(不知道怎么入手时,就照这 4 步走)
═══════════════════════════════════════════════════════════════════════
1. 先写形状,再写代码。
   你的图已经把每一步的维度标好了,这其实就是最难的那部分。
   代码只是把 "R^{ds×de} → R^{ds×de}" 翻译成 PyTorch 而已。

2. 从最小的零件写起,由内向外拼。
   本文件的顺序就是拼装顺序:
     ① 缩放点积注意力(一个纯函数,没有参数)
     ② 多头注意力(图下半部分)
     ③ MLP / 前馈网络
     ④ 编码器块(图上半部分:注意力 → 加 → 归一化 → MLP → 加 → 归一化)
     ⑤ 把 N 个块叠起来 = 编码器

3. 每个 nn.Module 只回答两个问题:
     __init__ :我有哪些"带参数的零件"?  (Linear、LayerNorm ……)
     forward  :数据按什么顺序流过这些零件?
   图里的每个"方框"对应 __init__ 里的一行,每根"箭头"对应 forward 里的一行。

4. 每写完一个零件,马上喂一个随机张量,检查输出形状对不对。
   (文件最底下的 if __name__ == "__main__" 就是在做这件事)

═══════════════════════════════════════════════════════════════════════
符号对照(图 ↔ 代码)
═══════════════════════════════════════════════════════════════════════
   图上的记号        代码里的名字      含义
   ds               seq_len          序列长度(有几个 token / 几个时间步)
   de               d_model          每个 token 的向量维度
   h                n_head           注意力头的个数
   dq = dk = dv     d_head           每个头内部的维度,= de / h
   (图上没有)        batch            一次同时处理几条序列

注意:图上的张量是 R^{ds×de},而真实代码里永远多一个最前面的 batch 维,
所以代码里是 (batch, ds, de)。下面注释里的形状都带着 batch。
"""

# ─────────────────────────────────────────────────────────────────────
# 引入什么库?—— 写 PyTorch 模型基本就是这固定的四行
# ─────────────────────────────────────────────────────────────────────
import math                        # 只用到 math.sqrt,算 √dk

import torch                       # 张量本身和张量运算:torch.randn、torch.cat、@ ……
import torch.nn as nn              # "带参数的层",都是类:nn.Linear、nn.LayerNorm、nn.Module
import torch.nn.functional as F    # "不带参数的运算",都是函数:F.softmax、F.relu
# 怎么判断用 nn 还是 F?
#   有需要训练的权重 → 用 nn 里的类,并且必须在 __init__ 里创建(这样权重才会被登记、被训练)
#   没有权重,只是个数学运算 → 用 F 里的函数,在 forward 里直接调用


# ═════════════════════════════════════════════════════════════════════
# ① 缩放点积注意力 —— 图中蓝色的 "Scaled Dot-Product Attention"
# ═════════════════════════════════════════════════════════════════════
# 公式:Attention(Q, K, V) = softmax( Q·Kᵀ / √dk ) · V
#
# 它没有任何可训练参数,所以写成普通函数就行,不需要 nn.Module。
def scaled_dot_product_attention(Q, K, V):
    """
    Q, K, V 的形状:(..., ds, d_head)
        "..." 表示前面可以有任意多个维度(batch,或者 batch 和 head),
        下面的写法只操作最后两个维度,所以对这两种情况都适用。
    返回:
        out     (..., ds, d_head)  每个位置加权汇总后的新向量
        weights (..., ds, ds)      注意力权重,第 i 行 = 第 i 个位置对所有位置的关注程度
    """
    d_k = K.size(-1)  # 取最后一维的大小,也就是 dk

    # 第 1 步:Q·Kᵀ —— 每个位置的 query 和所有位置的 key 做点积,得到"相关性打分"
    #   K.transpose(-2, -1) 交换最后两个维度:(..., ds, dk) → (..., dk, ds)
    #   @ 是矩阵乘法:(..., ds, dk) @ (..., dk, ds) → (..., ds, ds)
    # 第 2 步:除以 √dk —— 维度越大点积的数值越大,softmax 会变得极端(接近 one-hot),
    #   梯度就会很小。除以 √dk 把数值拉回正常范围。这就是 "Scaled" 的含义。
    scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)       # (..., ds, ds)

    # 第 3 步:softmax —— 把打分变成"和为 1 的权重"
    #   dim=-1 表示沿最后一维做,即每一行各自归一化
    weights = F.softmax(scores, dim=-1)                     # (..., ds, ds)

    # 第 4 步:用权重对 V 加权求和
    #   (..., ds, ds) @ (..., ds, dv) → (..., ds, dv)
    out = weights @ V                                       # (..., ds, d_head)

    return out, weights


# ═════════════════════════════════════════════════════════════════════
# ②-A 多头注意力:"照着图一比一写"的版本(用 for 循环,最容易看懂)
# ═════════════════════════════════════════════════════════════════════
class AttentionHead(nn.Module):
    """
    一个注意力头 = 图里叠在一起的那几层中的"一层":
        三个 Linear(产生 Q、K、V)+ 一次缩放点积注意力

    对应你写的:
        Q = X·W_Q,   W_Q ∈ R^{de×dq}
        K = X·W_K,   W_K ∈ R^{de×dk}
        V = X·W_V,   W_V ∈ R^{de×dv}
    """

    def __init__(self, d_model, d_head):
        # 这一行必须写、而且必须写在最前面!
        # 它让 nn.Module 完成初始化,之后 self.xxx = nn.Linear(...) 才能被正确登记为参数。
        # 忘了它会直接报错:"cannot assign module before Module.__init__() call"
        super().__init__()

        # nn.Linear(输入维度, 输出维度) 就是"乘一个权重矩阵(再加偏置)"
        # bias=False:不要偏置,这样就和你写的 Q = X·W_Q 完全一致。
        # (很多实际实现会保留偏置,两种都可以,差别很小)
        self.W_q = nn.Linear(d_model, d_head, bias=False)
        self.W_k = nn.Linear(d_model, d_head, bias=False)
        self.W_v = nn.Linear(d_model, d_head, bias=False)

    def forward(self, x):
        # x: (batch, ds, de)
        # nn.Linear 只作用在"最后一维"上,前面的维度原样保留,
        # 所以不用自己写循环,每个位置都会被同一个 W 变换。
        Q = self.W_q(x)                                     # (batch, ds, d_head)
        K = self.W_k(x)                                     # (batch, ds, d_head)
        V = self.W_v(x)                                     # (batch, ds, d_head)

        # Q、K、V 都来自同一个 x,所以叫"自注意力"(self-attention)
        out, _ = scaled_dot_product_attention(Q, K, V)      # (batch, ds, d_head)
        return out                                          # 对应图上的 R^{ds × de/h}


class MultiHeadAttentionLoop(nn.Module):
    """
    多头注意力(循环版):h 个头各算各的 → Concat → Linear
    和图的下半部分一一对应。
    """

    def __init__(self, d_model, n_head):
        super().__init__()

        # assert:条件不成立就立刻报错。de 必须能被 h 整除,否则 de/h 不是整数。
        assert d_model % n_head == 0, "d_model 必须能被 n_head 整除"
        d_head = d_model // n_head                          # 图上的 dq = dk = dv = de / h

        # 要放"一串子模块"时,必须用 nn.ModuleList,不能用普通的 Python list!
        # 用普通 list 的话 PyTorch 看不到里面的参数,它们不会被训练,也不会跟着 .to(device) 搬家。
        self.heads = nn.ModuleList(
            [AttentionHead(d_model, d_head) for _ in range(n_head)]
        )

        # 图上 Concat 之后的那个 Linear,权重是 R^{de×de}
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x):
        # x: (batch, ds, de)

        # 每个头各算一遍,得到 h 个 (batch, ds, d_head) 的张量
        head_outputs = [head(x) for head in self.heads]

        # Concat:沿最后一维拼接,h 个 d_head 拼回 de
        #   h × (batch, ds, de/h) → (batch, ds, de)
        concat = torch.cat(head_outputs, dim=-1)            # (batch, ds, de)

        # 最后的 Linear:把各个头的信息混合起来
        return self.W_o(concat)                             # (batch, ds, de)


# ═════════════════════════════════════════════════════════════════════
# ②-B 多头注意力:实际工程中的写法(不用循环,所有头一次算完)
# ═════════════════════════════════════════════════════════════════════
# 关键观察:
#   h 个 "de → de/h" 的小矩阵并排拼起来,正好是 1 个 "de → de" 的大矩阵。
#   所以可以只用一个大 Linear 一次算出所有头的 Q,再用 view 把最后一维"切"成 h 份。
#   结果和循环版完全一样(文件底部的测试 4 会验证这一点),但 GPU 上快得多。
#   你以后读别人的代码(包括 PyTorch 官方实现)看到的基本都是这种写法。
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head):
        super().__init__()
        assert d_model % n_head == 0, "d_model 必须能被 n_head 整除"

        # 存下来,forward 里做形状变换时要用
        self.n_head = n_head
        self.d_head = d_model // n_head

        # 每个都是 de → de 的大矩阵 = h 个头的小矩阵拼在一起
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model)

    def split_heads(self, t):
        """(batch, ds, de) → (batch, h, ds, d_head):把最后一维切成 h 个头,并把 head 维挪到前面"""
        batch, seq_len, _ = t.shape
        # view:不改变数据,只改变"怎么看"这块数据。把 de 拆成 (h, d_head)
        t = t.view(batch, seq_len, self.n_head, self.d_head)    # (batch, ds, h, d_head)
        # transpose(1, 2):交换第 1、2 维,让 head 紧跟在 batch 后面。
        # 这样最后两维是 (ds, d_head),正好是 scaled_dot_product_attention 要操作的两维,
        # 而 batch 和 head 都被当作"批量维度",每个头互不干扰地各算各的。
        return t.transpose(1, 2)                                # (batch, h, ds, d_head)

    def forward(self, x):
        # x: (batch, ds, de)
        batch, seq_len, d_model = x.shape

        # 1) 三个 Linear,然后切成多头
        Q = self.split_heads(self.W_q(x))                   # (batch, h, ds, d_head)
        K = self.split_heads(self.W_k(x))                   # (batch, h, ds, d_head)
        V = self.split_heads(self.W_v(x))                   # (batch, h, ds, d_head)

        # 2) 所有头同时做缩放点积注意力
        out, _ = scaled_dot_product_attention(Q, K, V)      # (batch, h, ds, d_head)

        # 3) Concat:就是 split_heads 的逆操作
        out = out.transpose(1, 2)                           # (batch, ds, h, d_head)
        # transpose 之后数据在内存里不再连续,必须先 .contiguous() 才能 view
        # (不想记这个的话,直接用 .reshape(...) 代替 .contiguous().view(...) 也行)
        out = out.contiguous().view(batch, seq_len, d_model)  # (batch, ds, de)

        # 4) 最后的 Linear
        return self.W_o(out)                                # (batch, ds, de)


# ═════════════════════════════════════════════════════════════════════
# ③ MLP(前馈网络)—— 图中并排的那几个 "MLP" 小方块
# ═════════════════════════════════════════════════════════════════════
# 两个容易误解的地方:
#
# (a) 图上画了好几个 MLP 方块,但它们是"同一个" MLP(共享同一套权重),
#     只是分别作用在每个位置上。方块的个数其实等于 ds(每个 token 一个)。
#     代码里不需要循环:nn.Linear 只作用于最后一维,自动对每个位置各算一次。
#
# (b) 你在图上给 MLP 标的是 R^{de×de}。标准 Transformer 的 MLP 其实是两层:
#         de → d_ff → de      (原论文 d_ff = 4·de,先放大再缩回来)
#     权重分别是 R^{de×d_ff} 和 R^{d_ff×de},中间夹一个非线性激活函数。
#     整体的输入输出确实都是 de 维,所以你标的"进 de、出 de"是对的,只是内部有个更宽的中间层。
#     没有中间的激活函数的话,两层 Linear 等价于一层,整个模型就失去非线性了。
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        # nn.Sequential:把几层按顺序串起来,forward 时自动依次调用
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),   # 放大:de → d_ff
            nn.ReLU(),                  # 非线性(原论文用 ReLU,BERT/GPT 用 GELU)
            nn.Linear(d_ff, d_model),   # 缩回:d_ff → de
        )

    def forward(self, x):
        return self.net(x)              # (batch, ds, de) → (batch, ds, de)


# ═════════════════════════════════════════════════════════════════════
# ④ 编码器块 —— 图的上半部分(整个浅蓝色框)
# ═════════════════════════════════════════════════════════════════════
# 沿着图从左到右读,数据流是:
#
#   x ──┬──→ [多头注意力] ──→ ⊕ ──→ [Layer norm] ──┬──→ [MLP] ──→ ⊕ ──→ [Layer norm] ──→ 输出
#       └──────────────────────┘                    └─────────────┘
#              残差连接(那条绕过去的线)                    残差连接
#
# 每一步的形状都是 (batch, ds, de),从头到尾不变 —— 正因如此,块才能一个接一个地叠起来。
class EncoderBlock(nn.Module):
    def __init__(self, d_model, n_head, d_ff):
        super().__init__()
        # 图里有 4 个方框,所以这里正好 4 行
        self.attn = MultiHeadAttention(d_model, n_head)     # 黄色:Multi-head Attention block
        self.norm1 = nn.LayerNorm(d_model)                  # 第一个红色:Layer norm
        self.mlp = FeedForward(d_model, d_ff)               # 灰色:MLP
        self.norm2 = nn.LayerNorm(d_model)                  # 第二个红色:Layer norm
        # 两个 LayerNorm 必须是两个独立的对象(各有各的参数),不能复用同一个。
        #
        # LayerNorm(d_model) 做的事:对每个 token 的 de 维向量单独做归一化
        # (减均值、除标准差),再乘、加两个可学习的参数。作用是让数值稳定,训练更容易。

    def forward(self, x):
        # x: (batch, ds, de)

        # 子层 1:注意力 → 残差相加(图上的 ⊕)→ 归一化
        #   "x + ..." 就是残差连接:把输入原样加回来。
        #   好处是梯度可以沿着这条"近路"直接传回去,深层网络才训练得动。
        #   能相加的前提是两者形状相同,这也是为什么注意力的输出必须还是 (ds, de)。
        x = self.norm1(x + self.attn(x))                    # (batch, ds, de)

        # 子层 2:MLP → 残差相加 → 归一化
        x = self.norm2(x + self.mlp(x))                     # (batch, ds, de)

        return x


# ═════════════════════════════════════════════════════════════════════
# ⑤ 编码器 = N 个编码器块首尾相接
# ═════════════════════════════════════════════════════════════════════
class Encoder(nn.Module):
    def __init__(self, d_model, n_head, d_ff, n_layer):
        super().__init__()
        # 又是"一串子模块",所以用 nn.ModuleList
        self.blocks = nn.ModuleList(
            [EncoderBlock(d_model, n_head, d_ff) for _ in range(n_layer)]
        )

    def forward(self, x):
        for block in self.blocks:       # 上一块的输出就是下一块的输入
            x = block(x)
        return x                        # (batch, ds, de)


# ═════════════════════════════════════════════════════════════════════
# 测试:每个零件都喂一个随机张量,检查形状
# ═════════════════════════════════════════════════════════════════════
# if __name__ == "__main__": 的意思是
#   "只有直接运行这个文件(python encoder_test.py)时才执行下面的代码;
#    如果是别的文件 import 这个文件,就不执行"。
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
    out, weights = scaled_dot_product_attention(q, q, q)
    print(f"① 注意力输出:                {tuple(out.shape)}   ← (batch, ds, d_head)")
    print(f"① 注意力权重:                {tuple(weights.shape)}   ← (batch, ds, ds)")
    # softmax 之后每一行的和应该等于 1
    assert torch.allclose(weights.sum(dim=-1), torch.ones(batch, seq_len))
    print("   每行权重之和 = 1  ✓")

    # ── 测试 2:单个注意力头 ────────────────────────────────────────
    head = AttentionHead(d_model, d_head)
    print(f"②-A 单个头输出:              {tuple(head(x).shape)}   ← (batch, ds, de/h)")

    # ── 测试 3:两种多头注意力 ──────────────────────────────────────
    mha_loop = MultiHeadAttentionLoop(d_model, n_head)
    mha_fast = MultiHeadAttention(d_model, n_head)
    print(f"②-A 多头注意力(循环版)输出:  {tuple(mha_loop(x).shape)}   ← (batch, ds, de)")
    print(f"②-B 多头注意力(向量化)输出:  {tuple(mha_fast(x).shape)}   ← (batch, ds, de)")

    # ── 测试 4:验证两种写法在数学上完全等价 ────────────────────────
    # 做法:把向量化版的大矩阵"切开",抄给循环版的每个头,然后看两边输出是否相同。
    #
    # 小知识:nn.Linear 的 weight 形状是 (输出维度, 输入维度),它算的是 y = x @ weight.T。
    # 所以和你图上写的 W ∈ R^{de×dk}(输入×输出)正好是转置关系。
    # 大矩阵 W_q.weight 是 (de, de),其中第 i 个头占的是第 i*d_head ~ (i+1)*d_head 行。
    with torch.no_grad():   # 手动改权重时不需要记录梯度
        for i, h in enumerate(mha_loop.heads):
            rows = slice(i * d_head, (i + 1) * d_head)
            h.W_q.weight.copy_(mha_fast.W_q.weight[rows])
            h.W_k.weight.copy_(mha_fast.W_k.weight[rows])
            h.W_v.weight.copy_(mha_fast.W_v.weight[rows])
        mha_loop.W_o.weight.copy_(mha_fast.W_o.weight)
        mha_loop.W_o.bias.copy_(mha_fast.W_o.bias)
    assert torch.allclose(mha_loop(x), mha_fast(x), atol=1e-6)
    print("   权重相同时,循环版和向量化版输出完全一致  ✓")

    # ── 测试 5:MLP ─────────────────────────────────────────────────
    mlp = FeedForward(d_model, d_ff)
    print(f"③ MLP 输出:                  {tuple(mlp(x).shape)}   ← (batch, ds, de)")

    # ── 测试 6:编码器块 ────────────────────────────────────────────
    block = EncoderBlock(d_model, n_head, d_ff)
    y = block(x)
    print(f"④ 编码器块输出:              {tuple(y.shape)}   ← (batch, ds, de),和输入一样")
    assert y.shape == x.shape

    # ── 测试 7:完整编码器(叠 3 层)────────────────────────────────
    encoder = Encoder(d_model, n_head, d_ff, n_layer=3)
    z = encoder(x)
    print(f"⑤ 3 层编码器输出:            {tuple(z.shape)}   ← (batch, ds, de)")
    assert z.shape == x.shape

    # ── 顺便看看一个编码器块里有哪些参数、各是什么形状 ──────────────
    print("\n一个编码器块里的全部参数:")
    for name, p in block.named_parameters():
        print(f"   {name:<22} {tuple(p.shape)}")
    n_params = sum(p.numel() for p in block.parameters())
    print(f"   合计 {n_params} 个参数")


# ═════════════════════════════════════════════════════════════════════
# 手写练习(从易到难)
# ═════════════════════════════════════════════════════════════════════
# 练习 0(最重要):新建一个空文件,只看着你的那张图,不看本文件,
#         按 ① → ⑤ 的顺序自己重写一遍。卡住了再回来看对应的那一段。
#         每写完一个零件就像上面那样打印一次形状。
#
# 练习 1:把 scaled_dot_product_attention 里的 "/ math.sqrt(d_k)" 删掉,
#         把 d_model 调到 512,打印 weights 的最大值,对比有无缩放时的区别。
#
# 练习 2:给 FeedForward 和注意力权重加上 nn.Dropout(原论文 p=0.1)。
#         想一想:Dropout 没有可训练参数,为什么通常还是写在 __init__ 里?
#         (提示:它在训练和推理时行为不同,需要跟着 model.train()/model.eval() 切换)
#
# 练习 3:你的图是 Post-LN(先相加、再归一化)。
#         把 EncoderBlock 改成 Pre-LN:x = x + self.attn(self.norm1(x))。
#         现在的大模型基本都用 Pre-LN,训练更稳定。
#
# 练习 4:自注意力本身不知道 token 的先后顺序(把输入打乱,输出只是跟着打乱)。
#         先写代码验证这一点,再实现原论文的正弦位置编码,加到输入 x 上。
#
# 练习 5:给 scaled_dot_product_attention 加一个 mask 参数,
#         在 softmax 之前用 scores.masked_fill(mask == 0, float("-inf")) 屏蔽掉 padding 位置。
#
# 练习 6:用同样的权重,把 MultiHeadAttention 的输出和 PyTorch 官方的
#         nn.MultiheadAttention(d_model, n_head, batch_first=True) 对比,看能否对上。
