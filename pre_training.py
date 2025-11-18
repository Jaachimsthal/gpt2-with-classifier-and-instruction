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