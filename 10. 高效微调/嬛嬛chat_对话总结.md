# 嬛嬛 Chat 微调对话总结

本文总结围绕 `10. 高效微调/02. 嬛嬛chat.ipynb` 中 Llama-3.1-Instruct LoRA 微调代码的讨论，供其他 agent 快速接续上下文。

## 1. `process_func` 中为什么手写 special token

原代码手动拼了 Llama-3 Instruct 的聊天格式，例如：

```text
<|begin_of_text|>
<|start_header_id|>system<|end_header_id|>
...
<|eot_id|>
<|start_header_id|>user<|end_header_id|>
...
<|start_header_id|>assistant<|end_header_id|>
```

所以 tokenizer 调用中使用：

```python
add_special_tokens=False
```

原因是 special token 已经手动写进文本里了，如果再让 tokenizer 自动添加，可能会重复添加 BOS/EOS 等 token，导致 chat 格式不一致。

## 2. `system`、`user`、`assistant` 三个角色

在 Chat/Instruct 模型里，对话不是普通文本，而是带角色结构的消息。

```text
system：系统设定，规定模型身份、风格、规则
user：用户输入，也就是问题或指令
assistant：模型应该生成的回答
```

当前任务里：

```text
system：现在你要扮演皇帝身边的女人--甄嬛
user：example["instruction"] + example["input"]
assistant：example["output"]
```

微调时模型看到完整上下文，但通常只训练 assistant 的回答部分。

## 3. 为什么 instruction 和 response 分开 tokenize

原代码大致是：

```python
instruction = tokenizer(prompt, add_special_tokens=False)
response = tokenizer(answer, add_special_tokens=False)

input_ids = instruction["input_ids"] + response["input_ids"]
labels = [-100] * len(instruction["input_ids"]) + response["input_ids"]
```

分开 tokenize 的核心目的不是为了得到不同输入，而是为了准确构造 `labels`。

`input_ids` 需要包含：

```text
system + user + assistant answer
```

因为模型回答时需要完整上下文。

但 `labels` 中 prompt 部分要设置成 `-100`：

```python
[-100] * len(instruction["input_ids"])
```

这表示 system/user/assistant header 部分不参与 loss，模型只学习预测 assistant 的回答。

如果把整段文本一次性 tokenizer，也仍然需要单独知道 prompt 有多少 token，才能给前面填 `-100`。

## 4. 使用 tokenizer 的 chat template 改写 `process_func`

推荐用 `tokenizer.apply_chat_template` 代替手写 special token，这样更不容易写错 Llama-3 的聊天格式。

示例：

```python
def process_func(example):
    MAX_LENGTH = 384

    system_prompt = "现在你要扮演皇帝身边的女人--甄嬛"
    user_prompt = example["instruction"] + example["input"]
    assistant_answer = example["output"]

    prompt_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    prompt_text = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True
    )

    full_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": assistant_answer},
    ]

    full_text = tokenizer.apply_chat_template(
        full_messages,
        tokenize=False,
        add_generation_prompt=False
    )

    prompt_ids = tokenizer(
        prompt_text,
        add_special_tokens=False
    )["input_ids"]

    full = tokenizer(
        full_text,
        add_special_tokens=False
    )

    input_ids = full["input_ids"]
    attention_mask = full["attention_mask"]
    labels = [-100] * len(prompt_ids) + input_ids[len(prompt_ids):]

    input_ids = input_ids[:MAX_LENGTH]
    attention_mask = attention_mask[:MAX_LENGTH]
    labels = labels[:MAX_LENGTH]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }
```

注意：

```python
apply_chat_template(..., tokenize=False)
```

会先生成带特殊 token 的聊天文本，因此后面再 tokenizer 时仍然要：

```python
add_special_tokens=False
```

否则可能重复添加 special token。

## 5. 为什么用 chat template，而不是直接 tokenizer

`tokenizer` 只负责：

```text
文本 -> token ids
```

但 `chat_template` 负责：

```text
system/user/assistant 消息结构 -> 模型熟悉的聊天格式
```

对于 Llama-3-Instruct，模型训练时看到的是带角色 header 和 `<|eot_id|>` 的格式。如果直接：

```python
tokenizer(system_prompt + user_prompt + assistant_answer)
```

模型看到的只是普通连续文本，不知道哪段是 system、哪段是 user、哪段是 assistant，也不知道什么时候轮到 assistant 回答。

因此对 Instruct/Chat 模型，优先使用 `apply_chat_template`。

## 6. 截断逻辑

如果 tokenizer 里使用：

```python
truncation=True,
max_length=MAX_LENGTH
```

那么 `input_ids` 和 `attention_mask` 已经被 tokenizer 截断。

但更清楚、更一致的写法是：先不在 tokenizer 里截断，手动统一截断三者：

```python
input_ids = input_ids[:MAX_LENGTH]
attention_mask = attention_mask[:MAX_LENGTH]
labels = labels[:MAX_LENGTH]
```

这样可以避免 `prompt_ids` 长度超过 `MAX_LENGTH` 时，`labels` 全部变成 `-100` 的隐性问题。

## 7. Data Collator 在哪里 padding

当前训练中一般使用：

```python
DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    padding=True
)
```

`process_func` 只负责把单条样本转成：

```python
{
    "input_ids": ...,
    "attention_mask": ...,
    "labels": ...
}
```

真正的 padding 在 `data_collator` 里发生。

流程：

```text
Dataset 中每条样本长度不同
Trainer 取出一个 batch
DataCollatorForSeq2Seq 找到该 batch 内最长长度
把 input_ids / attention_mask / labels pad 到一样长
喂给模型训练
```

这叫 dynamic padding，只 pad 到当前 batch 的最长长度，不一定 pad 到 `MAX_LENGTH`，更省显存和计算。

常见 padding 行为：

```text
input_ids：用 tokenizer.pad_token_id 补齐
attention_mask：真实 token 为 1，padding 为 0
labels：padding 通常为 -100，表示不参与 loss
```

## 8. LoRA 的 `target_modules`

代码中：

```python
target_modules=[
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj"
]
```

含义是：指定 LoRA 参数插入模型的哪些模块。

LoRA 不训练原模型全部参数，而是在指定线性层旁边加可训练的小矩阵。

这些模块对应：

```text
q_proj：attention 的 Query 投影
k_proj：attention 的 Key 投影
v_proj：attention 的 Value 投影
o_proj：attention 输出投影

gate_proj：MLP/FFN 门控投影
up_proj：MLP/FFN 升维投影
down_proj：MLP/FFN 降维投影
```

这个配置等于 attention 和 MLP 主要线性层都加 LoRA，效果通常较好，但显存和训练参数会比只调 `q_proj`、`v_proj` 多。

## 9. TrainingArguments 参数含义

示例：

```python
args = TrainingArguments(
    output_dir="./output/llama3_1_instruct_lora",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    logging_steps=10,
    num_train_epochs=3,
    save_steps=100,
    learning_rate=1e-4,
    save_on_each_node=True,
    gradient_checkpointing=True
)
```

### `per_device_train_batch_size`

每张 GPU 每次 forward/backward 处理的样本数。

如果 1 张 GPU：

```text
每个小 batch = 4 条样本
```

如果 2 张 GPU：

```text
每个全局小 batch = 4 * 2 = 8 条样本
```

### `gradient_accumulation_steps`

累计多少个小 batch 后，再更新一次参数。

如果 1 张 GPU：

```text
4 条/小 batch * 累计 4 次 = 等效 batch size 16
```

公式：

```text
effective_batch_size =
per_device_train_batch_size * gradient_accumulation_steps * GPU数量
```

### `logging_steps`

通常按 optimizer update step 计数。

```python
logging_steps=10
```

表示每更新 10 次参数记录一次日志。

### `save_steps`

也通常按 optimizer update step 计数。

```python
save_steps=100
```

表示每更新 100 次参数保存一次 checkpoint。

### `save_on_each_node`

多机分布式训练相关。

```text
node = 机器节点，不是 GPU
```

例如：

```text
1 台机器 4 张 GPU = 1 个 node
2 台机器每台 4 张 GPU = 2 个 node
```

```python
save_on_each_node=True
```

表示多机训练时每台机器都保存 checkpoint。单机训练基本不用关心。

## 10. `gradient_checkpointing=True`

名字容易误导。它不是保存中间梯度。

普通训练：

```text
forward 时保存大量中间激活值 activation
backward 时直接使用这些激活值计算梯度
优点：快
缺点：占显存
```

开启 gradient checkpointing：

```text
forward 时只保存少数关键 activation checkpoint
backward 需要中间结果时，从最近 checkpoint 重新 forward 算出来
再继续反传
```

因此它是：

```text
省显存，但训练更慢
```

它保存的是少量关键激活值，不是保存所有中间结果，也不是保存中间梯度。

## 11. 当前建议

如果继续优化这个 notebook，建议：

1. 用 `tokenizer.apply_chat_template` 代替手写 Llama-3 special token。
2. tokenizer 后统一手动截断 `input_ids`、`attention_mask`、`labels`。
3. 保持 `labels` 中 prompt 部分为 `-100`，只训练 assistant answer。
4. 继续使用 data collator 做 dynamic padding。
5. LoRA 的 `target_modules` 当前配置合理，适合追求更好微调效果；如果显存不足，可以先只调 `q_proj`、`v_proj`。
