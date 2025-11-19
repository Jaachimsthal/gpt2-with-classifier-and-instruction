import torch
import torch.nn as nn
from GPTModel import GPTModel2, GPT_CONFIG_124M
import tiktoken
from generate_text import generate_text_simple
from tiktoken.core import Encoding

# 模型生成文本
torch.manual_seed(123)
GPT_CONFIG_124M['context_length'] = 256 # 从124M的配置中将context_length减少到256个词元
model = GPTModel2(GPT_CONFIG_124M)
model.eval()

tokenizer = tiktoken.get_encoding("gpt2")

def text_to_token_ids(text: str, tokenizer: Encoding) -> torch.Tensor:
    """文本转token_ids

    Args:
        text (str): 文本
        tokenizer (Encoding): 编码器

    Returns:
        torch.Tensor: token ids张量
    """
    encoded = tokenizer.encode(text=text, allowed_special={'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    return encoded_tensor

def token_ids_to_text(token_ids: torch.Tensor, tokenizer: Encoding) -> str:
    """token_ids张量转文本

    Args:
        token_ids (torch.Tensor): 词元ID张量
        tokenizer (Encoding): 解码器

    Returns:
        str: 文本
    """
    return tokenizer.decode(token_ids.squeeze(0).tolist())

start_context = "Every effort moves you"

token_ids = generate_text_simple(
    model=model,
    idx=text_to_token_ids(start_context, tokenizer),
    max_new_tokens=10,
    context_size=GPT_CONFIG_124M['context_length']
)

print("Output text:\n", token_ids_to_text(token_ids=token_ids, tokenizer=tokenizer))

# 计算文本损失
inputs = torch.tensor([
    [16833, 3626, 6100], # every effort moves
    [40, 1107, 588]      # I really like
])

targets = torch.tensor([
    [3626, 6100, 345], # effort moves you
    [1107, 588, 11311] # really like chocolate
])

with torch.no_grad():
    logits: torch.Tensor = model(inputs)
probs = torch.softmax(logits, dim=-1)
# 获取与目标词元对应的初始softmax概率
text_idx = 0
target_probas_1 = probs[text_idx, [0, 1, 2], targets[text_idx]]
print("Text 1: ", target_probas_1)

text_idx = 1
target_probas_2 = probs[text_idx, [0, 1, 2], targets[text_idx]]
print("Text 2: ", target_probas_2)

token_ids = torch.argmax(probs, dim=-1, keepdim=True)
print(token_ids)
print(f"Targets batch 1:{token_ids_to_text(targets[0].flatten(), tokenizer)}")
print(f"Ouputs batch 1:{token_ids_to_text(token_ids[0].flatten(), tokenizer)}")

# 对概率分数应用对数
log_probas = torch.log(torch.cat((target_probas_1, target_probas_2)))
print(log_probas)
avg_log_probas = torch.mean(log_probas)
print(avg_log_probas * (-1))

logits_flat = logits.flatten(0, 1)
targets_flat = targets.flatten(0)

print("Flattened Logits:",  logits_flat.shape)
print("Flattened Targets:", targets_flat.shape)

loss = torch.nn.functional.cross_entropy(logits_flat, targets_flat)
print(loss)