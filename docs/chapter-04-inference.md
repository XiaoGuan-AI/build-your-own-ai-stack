# 第 4 章：推理引擎——让模型真正「跑起来」

> 目标：把第 3 章训练好的模型变成一个高效、可控、可交互的生成服务。
> 学完本章，你将亲手实现大模型部署的三大件：**KV Cache（提速）**、**采样策略（控质量）**、**服务循环（产品化）**。
> 代码：`code/ch04/inference.py`（约 200 行）+ 对第 2 章 `generate()` 的升级。

```bash
# 1. 确保有个训练好的模型（第 3 章产物）
.venv/Scripts/python.exe code/ch03/train.py --demo
# 2. 用推理引擎生成（默认加载 checkpoints/ch03-demo.pt）
.venv/Scripts/python.exe code/ch04/inference.py --prompt "床前明月光，疑是地上霜"
# 3. 看 KV Cache 加速效果
.venv/Scripts/python.exe code/ch04/inference.py --benchmark
# 4. 交互式聊天（试试它学到的「白话」部分）
.venv/Scripts/python.exe code/ch04/inference.py --interactive
```

---

## 4.1 「训练快、推理慢」的悖论

第 3 章训练时，一次前向算**整个序列的所有位置**——几百个 token 一步算完，还能并行。但**生成时是逐 token 的**：预测第 t 个 token，把它拼进序列，再预测 t+1……

```
训练：一次前向算全部位置（可并行）     推理：t 步生成 = t 次前向（串行）
  [床前明月光...] → 全序列 logits        [床] → 明 → 月 → 光 → ...
```

推理慢的根源：**第 t 步前向时，前 t-1 个 token 的 K/V 又算了一遍**——而它们和上一步算出的**一模一样**（K/V 只取决于该 token 本身和之前的信息，与「谁在预测」无关）。

---

## 4.2 KV Cache：把算过的 K/V 存起来

**核心思想**：第 t 步只计算新 token 自己的 Q、K、V，K/V 拼到缓存上；注意力只在新 token 的 Q 和历史 K/V 之间算。

```
无 Cache：每步重算整个序列          KV Cache：每步只算新 token + 拼接
  第 t 步成本 = O(t)                  第 t 步成本 = O(1)
  总成本     = O(T²)                  总成本     = O(T)
```

**你在本章代码里会看到的三个实现细节（全是真实工程的坑）：**

**① 掩码的绝对位置。** 增量推理时，新 token 的 query 要能看到**所有历史** key：掩码不再是从第 0 行开始，而是从 `T_full - T` 行开始（`T_full` = 历史+当前长度）。完整前向时 `T_full == T`，自动退化回原逻辑——**同一个掩码代码，两种模式都对**：

```python
att = att.masked_fill(self.mask[:, :, T_full - T:T_full, :T_full] == 0, float("-inf"))
```

**② 位置编码错位（本会话真实踩坑）。** 增量步输入只有 1 个 token，如果位置编码仍从位置 0 取，这个新 token 会以为自己是「序列开头」——**输出就错了**。必须传 `start_pos`（= 历史 K/V 长度），位置编码按绝对位置取：

```python
x = x + self.pos_encoding[:, start_pos : start_pos + T, :]
```

**③ 超长回退。** 教学版假设序列 ≤ max_len（真实系统用滑动窗口/RoPE 管理长序列，第 9 章扩展）。超出时清空 cache 退化为完整前向——**宁可慢，不能错**：

```python
if idx.size(1) > self.max_len:
    idx_cond = idx[:, -self.max_len:]
    cache = {}          # 回退，保证输出永远正确
```

**验证纪律**（本章最重要的工程习惯）：Cache 版和非 Cache 版，**同种子同参数，生成序列必须逐 token 完全一致**——任何 KV Cache 实现都必须过这关：

```python
torch.manual_seed(42)
out_cache = generate(..., use_cache=True)
torch.manual_seed(42)
out_full  = generate(..., use_cache=False)
assert torch.equal(out_cache, out_full)   # 一致性断言
```

---

## 4.3 采样策略：让模型说人话

模型输出的是词表上的概率分布，**怎么从这个分布里挑 token，就是采样策略**。

**先看反面教材——贪心（argmax）**：每步取概率最大的 token。结果：回答千篇一律、容易陷入复读（「我喜欢我喜欢我喜欢……」）。为什么？一个 token 的概率高，它的后续往往也是它自己（自增强循环）。

**三种标准武器（可叠加）：**

| 策略 | 做法 | 效果 | 类比 |
|------|------|------|------|
| **temperature** | logits ÷ T 再 softmax | T<1 更保守（分布变尖），T>1 更发散 | 气温：低=三思后行，高=天马行空 |
| **top-k** | 只留概率最大的 k 个 | 硬性排除离谱候选 | 只从「最有道理」的 k 个词里选 |
| **top-p**（nucleus）| 按概率降序累计到 p 就截断 | 候选数是**动态的**（分布尖时少、平时多）| 从「累计有道理到 90%」的词里选 |

**为什么 top-p 比 top-k 更聪明？** top-k 是「固定人数」：分布很尖时（模型很确定）k=50 还是放进了 49 个废话；分布很平时（模型不确定）k=5 又太武断。top-p 按概率说话：确定时只留 1-2 个候选，不确定时留几十个。**主流模型（GPT 系列）默认组合：temperature=0.8~1.0 + top-p=0.9~0.95**。

---

## 4.4 服务循环：从「脚本」到「产品」

推理引擎最后一步：把生成封装成可重复调用的服务。本章的 `inference.py` 做了三件事：

1. **加载模型**：checkpoint 现在带 `cfg`（第 3 章升级），加载不需要手工传配置：
   ```python
   state = torch.load(ckpt_path, map_location=device, weights_only=True)
   model = MiniTransformer(**state["cfg"])     # 配置从 checkpoint 自动恢复
   ```
2. **词表一致性**（容易踩的坑）：加载模型后必须用**训练时同一份语料**构造 `CharTokenizer`——词表对不上，生成全是乱码。
3. **接口**：`--prompt` 一次性生成 / `--interactive` 交互聊天 / `--benchmark` 测速。这就是 ChatGPT 网页版的「最小内核」。

---

## 4.5 实测数据（本机 CPU）

```
===== KV Cache 基准（prompt=41 字符，生成 23 token）=====
KV Cache ✅：0.02s（1173.3 token/s）
无 Cache  ：0.03s（883.2 token/s）
✅ 一致性验证：Cache 与非 Cache 生成序列完全一致
```

**为什么加速比才 1.3x？** 序列太短（41+23=64，模型 max_len 就是 64）。KV Cache 的收益正比于序列长度：无 Cache 每步 O(t)，Cache 每步 O(1)——**prompt 越长、生成越长，加速越猛**（真实系统 prompt 几千 token，加速 10~100x）。动手实验 2 让你自己验证。

---

## 4.6 动手实验

1. **采样策略对比**：同一个 prompt，分别用 `--temperature 0.2`（保守）、`1.0`、`2.0`（发散）、`--top-k 5`、`--top-p 0.9` 生成，对比文本的多样性和质量。
2. **序列长度 vs 加速比**：训练一个大 max_len 的模型（`--block-size 256`），用长 prompt（>100 字符）跑 benchmark，记录加速比。画一条「序列长度-加速比」曲线。
3. **复读机实验**：把 `sample_token` 换成 argmax（`torch.argmax`），生成 200 token，数一数重复 n-gram 的比例，和 top-p 采样对比。
4. **（进阶）温度调度**：生成开头用高温（探索），中段降回低温（稳定）——模仿人类「先发散后收敛」的写作。
5. **（进阶）给 KV Cache 做「缓存淘汰」**：把超长回退改成「滑动窗口 cache」（只保留最近 k 个 K/V），验证一致性断言是否还成立——不成立的话，想想为什么（答案：位置编码需要同步平移，真实系统用 RoPE 解决）。

---

## 4.7 检验点（Checkpoint）

**判断题：**

1. 「KV Cache 缓存的是每层注意力的 K 和 V 矩阵，Q 不用缓存。」 —— 对 / 错
2. 「增量推理时，新 token 的位置编码应该从 0 开始取。」 —— 对 / 错
3. 「top-p 采样比 top-k 好的原因之一：候选数是动态的，分布尖时自动少留候选。」 —— 对 / 错
4. 「贪心解码（每步取 argmax）一定比采样生成的文本更高质量。」 —— 对 / 错
5. 「KV Cache 的收益与序列长度无关，只与模型大小有关。」 —— 对 / 错

**编程题（每题 15 分钟）：**

- **A**：给 `sample_token` 加一个 `--repetition-penalty` 参数（出现过 token 的 logits 除以 1.2），实现「防复读」。
- **B**：写一个 `generate_with_progress`，每生成 10 个 token 打印一次速度和已生成文本，观察「越到后面越慢」（无 cache）vs「匀速」（有 cache）。
- **C**：把第 1 章的 BPE 分词器接进推理引擎（词表一致性那步换成 BPE），跑通端到端。
- **D**：实现 4.6 实验 5 的滑动窗口 cache。

---

<details>
<summary>答案（先自己做完再看）</summary>

1. **对**。Q 只用于当前 token 的查询，用完即弃；K/V 会被未来所有 token 查询，所以要缓存。
2. **错**。新 token 的绝对位置 = 历史长度 + i，位置编码从 `start_pos` 取（4.2 的坑②）。从 0 取会输出错误结果。
3. **对**。这是 nucleus sampling 的核心优势：候选数随分布形态自适应。
4. **错**。argmax 会陷入复读和千篇一律（4.3 反面教材）。采样在质量和多样性之间 trade-off，「高质量」对生成任务的定义本身就是多维的。
5. **错**。收益正比于序列长度：无 Cache O(T²) vs 有 Cache O(T)，T 越大差距越大（4.5 实测解释了为什么短序列加速比小）。

编程题 A 提示：penalty 要在 softmax **之前**作用于 logits，且只作用于「已经出现过的 token id」——你可以在生成循环里维护一个 `seen` 集合。

</details>
