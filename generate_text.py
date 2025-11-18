import torch
import torch.nn as nn

def generate_text_simple(model: nn.Module, idx: torch.Tensor, max_new_tokens: int, context_size: int) -> torch.Tensor:
    """LLM 文本生成

    Args:
        model (nn.Module): LLM Model
        idx (torch.Tensor): 输入张量
        max_new_tokens (int): 最大可生成新token数
        context_size (int): 上下文长度

    Returns:
        torch.Tensor: 生成结果文本
    """
    # 定义遍历循环，控制需要生成的新token数量
    for _ in range(max_new_tokens):
        # 当前文本截断至支持的长度，若LLM仅支持5个token，若文本长度为10，则只有最后5个词元被作为输入
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond) # 传入模型开始预测下一个词元
        # 只关注最后一个输出的内容，因此形状会从(batch, n_token, vocab_size)变为(batch, vocab_size)
        # 形状变化是因为已经取出了最后一个token，因此对于最后一个token，其只剩下维度vocab_size
        # 注意：torch.Tensor[:, :, :]中的:符号表示取当前轴的全部元素，而[:, -1, :]则表示第2轴只取最后一个向量，所以变为两批数据，只分别留一个维度为第3轴的向量
        logits = logits[:, -1, :]
        # 对两批数据的最后一个轴应用softmax，形状不变，只是和为1，概率均大于0
        probas = torch.softmax(logits, dim=-1)
        # 从probas中选择概率最大的词，得到最可能的预测结果，因为只有一个词，因此形状变为(batch, 1)
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)
        # 将计算出的下一个字符的索引添加到索引数组中，此时idx的形状会变为(batch, n_tokens+1)
        idx = torch.cat((idx, idx_next), dim=-1)
    
    return idx