# 🧠 makemore — Zero to Hero 系列学习笔记

> 基于 [Andrej Karpathy](https://github.com/karpathy) 的 **Neural Networks: Zero to Hero** 课程  
> 从零手写语言模型，从 Bigram 到 WaveNet，一步步拆解深度学习的底层原理。

---

## 📁 文件结构

| 文件 | 对应章节 | 核心主题 |
|------|---------|---------|
| `makemore_bigram.py` | Part 1 | Bigram 语言模型（统计法 + 神经网络法） |
| `makemore_mlp.py` | Part 2 | MLP 多层感知机 + Embedding |
| `makemore_bn.py` | Part 3 | 激活值诊断 + Kaiming 初始化 + BatchNorm |
| `makemore_backprop.py` | Part 4 | 手写反向传播（Backprop Ninja）|
| `makemore_wavenet.py` | Part 5 | WaveNet 层级架构 + 容器类 + FlattenConsecutive |
| `names.txt` | — | 训练数据（32033 个英文名字）|

---

## 📖 各章知识点详解

---

### Part 1 · `makemore_bigram.py`
#### Bigram 语言模型：统计 vs 神经网络

**核心思想：** 用前 1 个字符预测下一个字符，两种实现方式本质等价。

#### 🔑 关键知识点

**1. Bigram 计数矩阵**
```python
N = torch.zeros((vocab_size, vocab_size), dtype=torch.int32)
for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        N[stoi[ch1], stoi[ch2]] += 1
# N[i][j] = 字符 i 后面紧跟字符 j 的总次数（27×27 矩阵）
```

**2. 归一化 → 条件概率（拉普拉斯平滑）**
```python
P = (N + 1).float()                       # +1 平滑，避免概率为 0
P = P / P.sum(dim=1, keepdim=True)        # 每行归一化 → P(j|i)
# keepdim=True：保持 [27,1] 形状，方便广播
```

**3. 自回归采样**
```python
ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
# 按概率分布采样，不是取 argmax，而是"随机按概率选"
```

**4. 负对数似然损失（NLL Loss）**
```python
loss = -probs[torch.arange(num), ys].log().mean()
# 取正确类别的概率 → log → 取反 → 平均 = NLL Loss
```

**5. One-Hot 编码 + 单层线性网络**
```python
xenc  = F.one_hot(xs, num_classes=vocab_size).float()   # [N, 27]
W     = torch.randn((vocab_size, vocab_size), requires_grad=True)
logits = xenc @ W        # 线性变换
probs  = logits.exp() / logits.exp().sum(dim=1, keepdim=True)  # Softmax
```

**6. 梯度下降三步走**
```python
W.grad = None          # ① 清空梯度
loss.backward()        # ② 反向传播，计算 dL/dW
W.data += -lr * W.grad # ③ 更新参数（沿梯度负方向走）
```

#### 💡 核心结论
> **方法一（统计归一化）≡ 方法二（神经网络）**：训练后 `W ≈ log(N)`，Softmax + NLL + 梯度下降 等价于最大似然估计（MLE）。

---

### Part 2 · `makemore_mlp.py`
#### MLP 多层感知机：上下文扩展 + Embedding 表示

**核心思想：** 把上下文从 1 个字符扩展到 `block_size` 个，引入 Embedding 和隐藏层，大幅提升模型能力。

#### 🔑 关键知识点

**1. 滑动窗口数据集构建**
```python
def build_dataset(words):
    X, Y = [], []
    for w in words:
        context = [0] * block_size           # 初始用 '.' 填充
        for ch in w + '.':
            ix = stoi[ch]
            X.append(context)                # 前 block_size 个字符作为输入
            Y.append(ix)                     # 当前字符作为预测目标
            context = context[1:] + [ix]     # 窗口右移
    return torch.tensor(X), torch.tensor(Y)  # X: [N, 3], Y: [N]
```

**2. 训练集 / 验证集 / 测试集划分（8:1:1）**
```python
random.shuffle(words)
Xtr,  Ytr  = build_dataset(words[:n1])    # 80% 训练
Xdev, Ydev = build_dataset(words[n1:n2])  # 10% 验证（调超参）
Xte,  Yte  = build_dataset(words[n2:])    # 10% 测试（最终评估）
```

**3. Embedding 查表（核心！）**
```python
C   = torch.randn((vocab_size, emb_dim))  # Embedding 表 [27, 2]
emb = C[Xtr[ix]]     # [32, 3, 2]  — 比 one_hot @ C 更高效，结果相同
# 每个字符索引 → 一个可学习的低维向量
```

**4. MLP 前向传播**
```python
# 展平拼接：[32, 3, 2] → [32, 6]
h      = torch.tanh(emb.view(-1, block_size * emb_dim) @ W1 + b1)  # 隐藏层
logits = h @ W2 + b2                                                  # 输出层
loss   = F.cross_entropy(logits, Ytr[ix])   # Softmax + NLL，数值稳定
```

**5. Mini-Batch 梯度下降**
```python
ix = torch.randint(0, Xtr.shape[0], (32,), generator=g)  # 随机采 32 条
# 每次只用一小批数据，比全批量更快，随机性有助于跳出局部最优
```

**6. 学习率调度（Learning Rate Decay）**
```python
lr = 0.1 if i < 100000 else 0.01  # 前期大步走，后期小步精调
```

**7. 过拟合诊断**
```python
with torch.no_grad():
    loss_tr  = F.cross_entropy(C[Xtr]  ..., Ytr)
    loss_dev = F.cross_entropy(C[Xdev] ..., Ydev)
# train loss ≈ dev loss → 正常；dev loss >> train loss → 过拟合
```

#### 📊 效果对比
| 模型 | 验证集 Loss |
|------|------------|
| Bigram（统计法）| ~2.45 |
| MLP（Part 2） | ~2.17 ⬇️ |

---

### Part 3 · `makemore_bn.py`
#### 激活值诊断 + Kaiming 初始化 + Batch Normalization

**核心思想：** 诊断 MLP 的训练不稳定问题，学习两种修复方案。

#### 🔑 关键知识点

**1. 激活值饱和诊断**
```python
h_bad = torch.tanh(emb.view(-1, ...) @ W1_bad + b1_bad)
print((h_bad.abs() > 0.97).float().mean())  # 饱和比例
# 大量神经元饱和 → tanh 梯度 ≈ 0 → 梯度消失！
```

**2. Kaiming 初始化（修复方案 1）**
```python
# 推导：若 x~N(0,1)，W~N(0,σ²)，则 xW 的方差 = fan_in × σ²
# 令方差=1，需要 σ = 1/sqrt(fan_in)，tanh 的 gain = 5/3
W1 = torch.randn((fan_in, hidden_dim)) * (5/3) / fan_in ** 0.5

W2 = torch.randn((hidden_dim, vocab_size)) * 0.01  # 输出层要更小，避免初始loss过大
b1 = torch.zeros(hidden_dim)                        # 偏置初始化为 0，不是随机！
```

**3. Batch Normalization（修复方案 2，核心！）**
```python
# 正向传播：
bn_mean   = h_preact.mean(0, keepdim=True)          # [1, 200] batch 均值
bn_std    = h_preact.std(0, keepdim=True)            # [1, 200] batch 标准差
h_norm    = (h_preact - bn_mean) / (bn_std + 1e-5)  # 归一化到 N(0,1)
h_bn      = bn_gain * h_norm + bn_bias               # 可学习缩放+偏移
h         = torch.tanh(h_bn)                         # 激活

# 推理时用 running statistics（不能用单条样本算 batch 统计量）：
with torch.no_grad():
    bn_mean_running = 0.999 * bn_mean_running + 0.001 * bn_mean  # EMA 更新
    bn_std_running  = 0.999 * bn_std_running  + 0.001 * bn_std
```

**4. 模块化重写（nn.Module 风格）**
```python
class Linear:
    def __init__(self, fan_in, fan_out, bias=True, generator=None):
        self.weight = torch.randn((fan_in, fan_out), generator=generator) / fan_in**0.5
        self.bias   = torch.zeros(fan_out) if bias else None
    def __call__(self, x):
        self.out = x @ self.weight + (self.bias if self.bias is not None else 0)
        return self.out
    def parameters(self):
        return [self.weight] + ([] if self.bias is None else [self.bias])

class BatchNorm1d:
    def __init__(self, dim, eps=1e-5, momentum=0.1):
        self.gamma, self.beta = torch.ones(dim), torch.zeros(dim)
        self.running_mean, self.running_var = torch.zeros(dim), torch.ones(dim)
        self.training = True
    def __call__(self, x):
        xmean = x.mean(0, keepdim=True) if self.training else self.running_mean
        xvar  = x.var(0, keepdim=True)  if self.training else self.running_var
        xhat  = (x - xmean) / torch.sqrt(xvar + self.eps)
        self.out = self.gamma * xhat + self.beta
        if self.training:
            with torch.no_grad():
                self.running_mean = (1-self.momentum)*self.running_mean + self.momentum*xmean.squeeze()
                self.running_var  = (1-self.momentum)*self.running_var  + self.momentum*xvar.squeeze()
        return self.out

class Tanh:
    def __call__(self, x):
        self.out = torch.tanh(x)
        return self.out

# 组装成 Sequential 风格
model = [Linear(fan_in, hidden_dim, bias=False), BatchNorm1d(hidden_dim), Tanh(), Linear(hidden_dim, vocab_size)]
parameters = [C] + [p for layer in model for p in layer.parameters()]
```

#### 💡 核心结论
| 问题 | 原因 | 解决方案 |
|------|------|---------|
| tanh 饱和 | `h_preact` 方差太大 | Kaiming 初始化 |
| 初始 loss 过大 | `W2` 太大，softmax 过于"自信" | `W2 *= 0.01` |
| 训练不稳定 | 激活值分布随层数漂移 | Batch Normalization |
| 代码杂乱 | 直接写矩阵 | 模块化 Layer 类 |

---

### Part 4 · `makemore_backprop.py`
#### Becoming a Backprop Ninja：完全手写反向传播

**核心思想：** 完全不调用 `.backward()`，手动推导并实现每一步的梯度，用 PyTorch autograd 验证正确性。

#### 🔑 关键知识点

**1. 梯度验证工具 `cmp()`**
```python
def cmp(label, dt, t):
    ex   = torch.all(dt == t.grad).item()           # 完全相等？
    app  = torch.allclose(dt, t.grad, atol=1e-4)    # 近似相等？
    maxd = (dt - t.grad).abs().max().item()          # 最大绝对误差
    print(f'{"✅" if ex or app else "❌"} {label}: exact={ex}, approx={app}, maxdiff={maxd:.2e}')
```

**2. Cross Entropy 手写反向（拆解成 7 步）**
```python
# 正向：logits → 减最大值（数值稳定） → exp → sum → 归一化 → log → 取正确类别 → 取反均值
dlogprobs = torch.zeros_like(logprobs)
dlogprobs[range(B), Yb] = -1.0 / B               # (g) -mean(logprobs[正确类])

dprobs          = dlogprobs / probs                # (f) log(x) 的导数 = 1/x
dcounts_sum_inv = (dprobs * counts).sum(1, keepdim=True)
dcounts         = dprobs * counts_sum_inv          # (e) probs = counts * counts_sum_inv
dcounts_sum     = -counts_sum**-2 * dcounts_sum_inv
dcounts        += dcounts_sum.expand_as(counts)    # ← 广播的逆操作是 sum！
dnorm_logits    = dcounts * counts                 # (c) exp 的导数 = 自身
dlogits         = dnorm_logits.clone()             # (b) 减法：直接传递
dlogit_maxes    = -dnorm_logits.sum(1, keepdim=True)
dlogits        += (logits == logit_maxes) * dlogit_maxes  # (a) max 只有最大值位置有梯度
```

**3. 线性层反向（矩阵乘法梯度）**
```python
# 正向: logits = h @ W2 + b2
dh  = dlogits @ W2.T        # dL/dh  = dL/d(logits) @ W2ᵀ
dW2 = h.T @ dlogits         # dL/dW2 = hᵀ @ dL/d(logits)
db2 = dlogits.sum(0)        # dL/db2 = sum over batch（偏置对每个样本都加了一次）
```

**4. Tanh 反向**
```python
# 正向: h = tanh(hpreact)
# d(tanh(x))/dx = 1 - tanh(x)² = 1 - h²
dhpreact = dh * (1.0 - h**2)
# 当 h ≈ ±1（饱和区），梯度 ≈ 0 → 梯度消失
```

**5. BatchNorm 反向（最复杂，最重要！）**
```python
# 逐步推导版（8步）：
dbn_gain  = (dhpreact * bnraw).sum(0, keepdim=True)
dbn_bias  = dhpreact.sum(0, keepdim=True)
dbnraw    = dhpreact * bn_gain
dbnvar_inv = (dbnraw * bndiff).sum(0, keepdim=True)
dbnvar    = dbnvar_inv * (-0.5) * (bnvar + 1e-5)**(-1.5)
dbndiff2  = dbnvar * (1.0/(B-1)) * torch.ones_like(bndiff2)
dbndiff   = dbnraw * bnvar_inv + dbndiff2 * 2.0 * bndiff
dbnmean_i = -dbndiff.sum(0, keepdim=True)
dhprebn   = dbndiff + dbnmean_i * (1.0/B)

# 向量化化简版（等价，更优雅）：
dhprebn = (bn_gain * bnvar_inv / B) * (
    B * dbnraw - dbnraw.sum(0) - bnraw * (dbnraw * bnraw).sum(0)
)
```

**6. Embedding 反向（scatter add）**
```python
# 正向: emb = C[Xb]（查表，同一 index 可能被多个 batch 共享）
# 反向: 梯度要按 index 累加（不能直接赋值！）
dC = torch.zeros_like(C)
for i in range(Xb.shape[0]):
    for j in range(Xb.shape[1]):
        dC[Xb[i,j]] += demb[i,j]   # += 累加，同一字符的梯度叠加
```

#### 📐 手写反向传播速查表

| 操作 | 正向 | 梯度公式 |
|------|------|---------|
| 矩阵乘法 | `C = A @ B` | `dA = dC @ Bᵀ`，`dB = Aᵀ @ dC` |
| 加法 | `c = a + b` | `da = dc`，`db = dc` |
| 乘法 | `c = a * b` | `da = dc * b`，`db = dc * a` |
| 求和 | `b = a.sum(0)` | `da = db.expand_as(a)`（广播回去） |
| 均值 | `b = a.mean()` | `da = db / n * ones_like(a)` |
| exp | `b = exp(a)` | `da = db * b`（exp 导数等于自身）|
| log | `b = log(a)` | `da = db / a` |
| 幂 | `b = a**n` | `da = db * n * a**(n-1)` |
| tanh | `b = tanh(a)` | `da = db * (1 - b²)` |
| max | `b = max(a)` | 只有最大值位置有梯度，其余为 0 |

#### 💡 核心原则

> 1. **链式法则**：从输出往输入倒推，每步乘以局部导数
> 2. **多路径累加**：同一变量被多处使用，梯度必须 `+=`，不能 `=`
> 3. **广播的逆**：正向是广播（`expand`），反向就是 `sum`；正向是 `sum`，反向就是广播
> 4. **保存中间变量**：正向传播中所有中间结果都要保存，反向传播会用到

---

### Part 5 · `makemore_wavenet.py`
#### WaveNet 层级架构：从"平铺 MLP"到"树状压缩"

**核心思想：** Part 2~4 的 MLP 把所有 `block_size` 个字符一次性展平输入，信息密度丢失严重。  
Part 5 借鉴 WaveNet（DeepMind 2016）的思路，**每层只合并相邻一对字符**，逐层向上汇聚，形成层级特征树。同时引入容器类，让代码像 PyTorch 官方 API 一样整洁。

#### 🔑 关键知识点

**1. 问题根源：平铺展平丢失了位置层级信息**
```python
# ❌ Part 2~4 的做法：8个字符直接一次性展平
# [B, 8, 10] → view → [B, 80] → Linear → [B, 200]
# 所有字符同等对待，没有"先合并近邻"的归纳偏置

# ✅ Part 5 的做法：逐层合并，像一棵二叉树
# 第1层：[B, 8, 10] → 每次合并2个 → [B, 4, 20] → Linear → [B, 4, 200]
# 第2层：[B, 4, 200] → 每次合并2个 → [B, 2, 400] → Linear → [B, 2, 200]
# 第3层：[B, 2, 200] → 每次合并2个 → [B, 1, 400] → Linear → [B, 200]
# 最终才展平，信息是逐步、有序地汇聚的
```

**2. `FlattenConsecutive`：核心新层，每次只展平相邻 n 个时间步**
```python
class FlattenConsecutive:
    """
    每次只把相邻 n 个时间步合并（而不是全部展平）
    输入：[B, T, C]
    输出：[B, T//n, C*n]  ← T 减半，C 翻倍（以 n=2 为例）
    """
    def __init__(self, n):
        self.n = n   # 每次合并几个相邻时间步

    def __call__(self, x):
        B, T, C = x.shape
        x = x.view(B, T // self.n, C * self.n)
        # 如果 T//n == 1，把多余的维度 squeeze 掉（最后一层）
        if x.shape[1] == 1:
            x = x.squeeze(1)
        self.out = x
        return self.out

    def parameters(self):
        return []   # 无参数，只是 reshape
```

**3. `Sequential` 容器类：像俄罗斯套娃一样嵌套层**
```python
class Sequential:
    """
    把一组层顺序组合，统一调用和管理参数
    等价于 PyTorch 的 nn.Sequential
    """
    def __init__(self, layers):
        self.layers = layers

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)     # 依次流过每一层
        self.out = x
        return self.out

    def parameters(self):
        # 递归收集所有子层的参数（包括嵌套的 Sequential 内部的层）
        return [p for layer in self.layers for p in layer.parameters()]
```

**4. 完整 WaveNet 网络构建（block_size=8，4层压缩）**
```python
# 超参数
block_size = 8    # 上下文长度（Part5 升级到8，需要3层才能压缩完）
n_embd     = 24   # embedding 维度
n_hidden   = 128  # 隐藏层维度

model = Sequential([
    # --- Embedding 查表 ---
    Embedding(vocab_size, n_embd),           # [B, 8] → [B, 8, 24]

    # --- 第1个 WaveNet 块：合并相邻2个时间步 ---
    FlattenConsecutive(2),                   # [B, 8, 24] → [B, 4, 48]
    Linear(n_embd * 2, n_hidden, bias=False),# [B, 4, 48] → [B, 4, 128]
    BatchNorm1d(n_hidden),
    Tanh(),

    # --- 第2个 WaveNet 块：再次合并 ---
    FlattenConsecutive(2),                   # [B, 4, 128] → [B, 2, 256]
    Linear(n_hidden * 2, n_hidden, bias=False),
    BatchNorm1d(n_hidden),
    Tanh(),

    # --- 第3个 WaveNet 块：最后合并 ---
    FlattenConsecutive(2),                   # [B, 2, 128] → [B, 256]（squeeze）
    Linear(n_hidden * 2, n_hidden, bias=False),
    BatchNorm1d(n_hidden),
    Tanh(),

    # --- 输出层 ---
    Linear(n_hidden, vocab_size),            # [B, 128] → [B, 27]
])
```

**5. `Embedding` 层（把查表操作也封装进来）**
```python
class Embedding:
    """
    词表查表层（封装 C[x] 操作）
    等价于 nn.Embedding
    """
    def __init__(self, num_embeddings, embedding_dim, generator=None):
        self.weight = torch.randn((num_embeddings, embedding_dim), generator=generator)

    def __call__(self, IX):
        self.out = self.weight[IX]   # 查表：[B, T] → [B, T, C]
        return self.out

    def parameters(self):
        return [self.weight]
```

**6. `BatchNorm1d` 升级：同时支持 2D `[B, C]` 和 3D `[B, T, C]` 输入**
```python
class BatchNorm1d:
    def __call__(self, x):
        # WaveNet 中 x 可能是 [B, T, C]（中间层），也可能是 [B, C]（最后层）
        if x.ndim == 2:
            dim = 0               # 2D：沿 batch 维度归一化
        elif x.ndim == 3:
            dim = (0, 1)          # 3D：沿 batch 和 time 两个维度归一化
        xmean = x.mean(dim, keepdim=True)
        xvar  = x.var(dim,  keepdim=True)
        # ... 后续和 Part3 一样
```

**7. training / eval 模式切换（统一管理）**
```python
# 不用每层手动设 .training，直接遍历所有层
for layer in model.layers:
    layer.training = False   # 切换到 eval 模式（BN 用 running statistics）

# 等价于 PyTorch 的 model.eval()
```

**8. 参数量统计（验证架构设计是否合理）**
```python
parameters = model.parameters()
print(sum(p.nelement() for p in parameters))   # 打印总参数量

# 各参数的 shape（帮助 debug 架构）
for p in parameters:
    print(p.shape)
```

#### 🌊 WaveNet 层级压缩示意图

```
输入: [B, 8, 24]  ← 8个字符，每个24维
         │
   FlattenConsecutive(2)
         │
      [B, 4, 48]  ← 4组，每组是相邻2字符拼接
         │ Linear + BN + Tanh
      [B, 4, 128]
         │
   FlattenConsecutive(2)
         │
      [B, 2, 256]  ← 2组，每组是相邻2个128维特征拼接
         │ Linear + BN + Tanh
      [B, 2, 128]
         │
   FlattenConsecutive(2)
         │
       [B, 256]   ← 最终合并成1个向量（squeeze 掉 T 维）
         │ Linear + BN + Tanh
       [B, 128]
         │ Linear
       [B, 27]    ← logits，27个词表概率
```

#### 💡 核心结论
| 对比维度 | Part 2~4 MLP | Part 5 WaveNet |
|---------|-------------|---------------|
| 上下文处理 | 一次性展平全部 | 逐层合并相邻对 |
| block_size | 3 | 8（可以更大！）|
| 架构归纳偏置 | 无位置概念 | 近邻字符优先合并 |
| 代码组织 | 裸列表 | Sequential 容器 |
| BN 输入维度 | 仅支持 2D | 同时支持 2D/3D |
| 验证集 Loss | ~2.17 | **~2.03** ⬇️ |

> **WaveNet 的本质**：给模型一个「先看近邻，再看远邻」的归纳偏置（Inductive Bias），  
> 而不是把所有信息一股脑平铺，让模型自己从噪音里刨出结构。

---

## 🚀 快速开始

### 环境要求
```bash
pip install torch matplotlib
# 或使用 uv（推荐）
uv add torch matplotlib
```

### 运行任意章节
```bash
# 确保 names.txt 与代码文件在同一目录
python makemore_bigram.py    # Part 1
python makemore_mlp.py       # Part 2
python makemore_bn.py        # Part 3
python makemore_backprop.py  # Part 4
python makemore_wavenet.py   # Part 5
```

---

## 📈 各章模型性能对比

| 章节 | 模型 | 验证集 Loss | 改进点 |
|------|------|------------|-------|
| Part 1 | Bigram（统计法）| ~2.45 | 基准线 |
| Part 2 | MLP | ~2.17 | Embedding + 多层网络 |
| Part 3 | MLP + BN | ~2.10 | Kaiming 初始化 + BatchNorm |
| Part 4 | MLP + BN（手写梯度）| ~2.10 | 手动实现，效果与 autograd 等价 |
| Part 5 | WaveNet | **~2.03** | 层级架构 + FlattenConsecutive + 容器类 |

---

## 🔗 参考资料

- 🎥 [Neural Networks: Zero to Hero（Karpathy）](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ)
- 📄 [Bengio et al. 2003 - A Neural Probabilistic Language Model](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf)
- 📦 [makemore 原始仓库](https://github.com/karpathy/makemore)
- 📦 [names.txt 数据集](https://github.com/karpathy/makemore/blob/master/names.txt)
