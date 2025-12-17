import torch
import torch.nn as nn
from typing_extensions import Dict, Any
import tiktoken
from tiktoken.core import Encoding

GPT_CONFIG_124M = {
    'vocab_size': 50257, # 词汇表大小
    'context_length': 1024, # 上下文维度
    'emb_dim': 768, # 嵌入维度
    'n_heads': 12, # 注意力头的数量
    'n_layers': 12, # 层数
    'drop_rate': 0.1, # 暂退率
    'qkv_bias': False # Q、K、V偏置
}

GPT_CONFIG_MEDIUM = {
    'vocab_size': 50257, # 词汇表大小
    'context_length': 1024, # 上下文维度
    'emb_dim': 1024, # 嵌入维度
    'n_heads': 16, # 注意力头的数量
    'n_layers': 24, # 层数
    'drop_rate': 0.1, # 暂退率
    'qkv_bias': False # Q、K、V偏置
}

GPT_CONFIG_LARGE = {
    'vocab_size': 50257, # 词汇表大小
    'context_length': 1024, # 上下文维度
    'emb_dim': 1280, # 嵌入维度
    'n_heads': 20, # 注意力头的数量
    'n_layers': 36, # 层数
    'drop_rate': 0.1, # 暂退率
    'qkv_bias': False # Q、K、V偏置
}

GPT_CONFIG_XL = {
    'vocab_size': 50257, # 词汇表大小
    'context_length': 1024, # 上下文维度
    'emb_dim': 1600, # 嵌入维度
    'n_heads': 25, # 注意力头的数量
    'n_layers': 48, # 层数
    'drop_rate': 0.1, # 暂退率
    'qkv_bias': False # Q、K、V偏置
}

class GPTModel2(nn.Module):
    """GPT-2 PyTorch Module

    Args:
        nn (nn.Module): PyTorch Module
    """
    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        self.token_embeddings    = nn.Embedding(cfg['vocab_size'], cfg['emb_dim']) # 词嵌入
        self.position_embeddings = nn.Embedding(cfg['context_length'], cfg['emb_dim']) # 位置嵌入
        self.dropout_embeddings  = nn.Dropout(cfg['drop_rate']) # 暂退层
        self.transformer_blocks  = nn.Sequential(*[TransformerBlock(cfg) for _ in range(cfg['n_layers'])]) # Transformer块（使用12次、12层transformer block）
        self.final_norm          = LayerNorm(cfg) # 最后一层层归一化
        self.out_head            = nn.Linear(cfg['emb_dim'], cfg['vocab_size'], bias=False) # 线性输出层
    
    def forward(self, in_idx: torch.Tensor) -> torch.Tensor:
        """前向传播

        Args:
            in_idx (torch.Tensor): 输入

        Returns:
            torch.Tensor: 输出
        """
        batch_size, seq_len = in_idx.shape
        # 词嵌入(nn.Embeddings)本质上是一种one-hot encoding的更有效的实现，Embeddings对嵌入的词元ID先进行独热编码，随后再全连接层中进行线性变换（W·in_idx）
        # 线性变换并不会改变原本的信息，但Embeddings却可以反向传播学习更新
        token_embeddings = self.token_embeddings(in_idx)
        # 位置编码的原因是因为Self-Attentions无法感知词元在序列中的位置和顺序
        position_embeddings = self.position_embeddings(torch.arange(seq_len, device=in_idx.device)) # 位置嵌入
        
        x = token_embeddings + position_embeddings
        x = self.dropout_embeddings(x)
        x = self.transformer_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits
        
class TransformerBlock(nn.Module):
    """掩码因果自注意力机制

    Args:
        nn (nn.Module): PyTorch Module
    """
    def __init__(self, cfg: Dict[str, Any]):
        """Transformer块，架构为 PreLayerNorm->MultiHeadAttention->dropout->cutshot->PostLayerNorm->FeedForward->dropout

        Args:
            cfg (Dict[str, Any]): _description_
        """
        super().__init__()
        self.pre_layer_norm       = LayerNorm(cfg=cfg) # 前层归一化
        self.post_layer_nrom      = LayerNorm(cfg=cfg) # 后层归一化
        self.multi_head_attention = MultiHeadAttention(cfg=cfg) # 多头因果点积缩放自注意力
        self.dropout              = nn.Dropout(cfg['drop_rate']) # 暂退层
        self.feed_forward         = FeedForwardLayer(cfg=cfg) # GELU激活的前馈神经网络
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播

        Args:
            x (torch.Tensor): 词嵌入输入

        Returns:
            torch.Tensor: Transformer输出
        """
        shortcut = x # 快捷连接
        x = self.pre_layer_norm(x)
        x = self.multi_head_attention(x)
        x = self.dropout(x)
        x = shortcut + x # H(x)=F(x)+x
        
        shortcut = x
        x = self.post_layer_nrom(x)
        x = self.feed_forward(x)
        x = self.dropout(x)
        x = shortcut + x
        return x
        
class FeedForwardLayer(nn.Module):
    """前馈神经网络

    Args:
        nn (nn.Module): PyTorch Module
    """
    def __init__(self, cfg: Dict[str, Any]):
        """_summary_

        Args:
            cfg (Dict[str, Any]): _description_
        """
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg['emb_dim'], 4*cfg['emb_dim']),
            GELU(),
            nn.Linear(4*cfg['emb_dim'], cfg['emb_dim'])
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播

        Args:
            x (torch.Tensor): inputs tensor

        Returns:
            torch.Tensor: output tensor
        """
        return self.layers(x)

class GELU(nn.Module):
    """GELU激活函数层

    Args:
        nn (nn.Module): PyTorch Module
    """
    def __init__(self):
        """构造函数
        """
        super().__init__()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播（GELU激活）

        Args:
            x (torch.Tensor): inputs tensor

        Returns:
            torch.Tensor: output tensor
        """
        
        # GELU(x)=x·Φ(x)，其中Φ(x)是标准高斯分布的累积分布函数
        # 考虑性能使用当前这种计算量较小的近似实现：GELU(x)≈0.5·x·(1+tanh[ Sqrt(2/Π) · (x+0.044715 · x^{3}) ])
        # GELU类似于ReLU，但相比ReLU更加平滑，几乎在所有负值（除了x≈-0.75）上都有非零梯度
        return 0.5 * x * (1+torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) * (x + 0.044715 * torch.pow(x, 3))
        ))

class LayerNorm(nn.Module):
    """层归一化

    Args:
        nn (nn.Module): PyTorch Module
    """
    def __init__(self, cfg: Dict[str, Any]):
        """初始化

        Args:
            cfg (Dict[str, Any]): 超参配置
        """
        super().__init__()
        self.epsilon = 1e-5 # 微小常数，用于在归一化过程中加到方差上避免除0错误
        # Scale和Shift最终会对归一化结果进行缩放和偏移，使用可训练的参数便于在模型训练过程中自行调整缩放尺度和偏移情况
        self.scale   = nn.Parameter(torch.ones(cfg['emb_dim']))
        self.shift   = nn.Parameter(torch.zeros(cfg['emb_dim']))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播

        Args:
            x (torch.Tensor): 输入

        Returns:
            torch.Tensor: 层归一化输出
        """
        # dim=-1表示对最后一个轴计算，也就是说会保留最后一个轴之前的所有轴，相当于将数据集中，最终样本每一行计算均值或方差，是每个样本的均值和方差
        # keepdim=True表示维持张量原本的维度，最后一维是样本数据的维度，计算会导致其降维
        mean = x.mean(dim=-1, keepdim=True)
        var  = x.var(dim=-1, keepdim=True, unbiased=False)
        # 层归一化的计算是针对每一行样本，使整个数据集中的每一个样本都减去均值并除以方差开根号的值，使得所有样本的均值为0，方差为1（形状不会改变，mean和var的结果是每一行样本的均值和方差）
        norm = (x-mean) / torch.sqrt(var + self.epsilon)
        return (self.scale * norm + self.shift)

class MultiHeadAttention(nn.Module):
    """多头因果点积缩放注意力

    Args:
        nn (nn.Module): PyTorch Module
    """
    def __init__(self, cfg: Dict[str, Any]):
        """初始化多头因果点积缩放注意力

        Args:
            cfg (Dict[str, Any]): 超参配置
        """
        super().__init__()
        assert (cfg['emb_dim'] % cfg['n_heads'] == 0), \
            "output dimension must be divisible by heads numbers"
        
        # 多头注意力机制的输出维度
        self.out_dimension = cfg['emb_dim']
        # Attention头的数量
        self.heads_nums  = cfg['n_heads']
        # 每个Attention头对应的权重矩阵的维度
        self.heads_dimension = cfg['emb_dim'] // self.heads_nums
        # Attention机制同样需要对Query、Key、Value增加可学习的线性层，以便在模型训练过程中优化Q、K、V计算
        self.query_layer = nn.Linear(cfg['emb_dim'], cfg['emb_dim'], bias=cfg['qkv_bias'])
        self.key_layer   = nn.Linear(cfg['emb_dim'], cfg['emb_dim'], bias=cfg['qkv_bias'])
        self.value_layer = nn.Linear(cfg['emb_dim'], cfg['emb_dim'], bias=cfg['qkv_bias'])
        # Attention机制在训练过程中使用dropout层正则化
        self.dropout     = nn.Dropout(cfg['drop_rate'])
        # 最后输出注意力分数需要一个线性层组织结果
        self.out_layer   = nn.Linear(cfg['emb_dim'], cfg['emb_dim'])
        # 掩码层
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(cfg['context_length'], cfg['context_length']), diagonal=1)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """多头因果点积缩放注意力前向传播

        Args:
            x (torch.Tensor): 输入

        Returns:
            torch.Tensor: 上下文向量
        """
        # 0.输入数据的形状（批次、token数量、token维度）
        # 即输入张量的第一个轴表示了当前批次，第二个轴表示了序列包含的词元数量、第三个轴表示了每个词元的维度
        batch_size, nums_tokens, token_dimension = x.shape
        
        # 1.首先需要分别计算Query、Key、Value张量（W_query @ x=queries...）（这里计算了一个更大的Q、K、V矩阵）
        queries: torch.Tensor = self.query_layer(x)
        keys: torch.Tensor    = self.key_layer(x)
        values: torch.Tensor  = self.value_layer(x)
        
        # 1.1.为了适应多头注意力，需要将计算的矩阵拆解，根据超参设置，将输出维度根据头的数量进行拆分，每个头对应emb_dim（Attention最终输出的维度，head_out_dim） / heads_nums维
        queries = queries.view(batch_size, nums_tokens, self.heads_nums, self.heads_dimension)
        keys    = keys.view(batch_size, nums_tokens, self.heads_nums, self.heads_dimension)
        values  = values.view(batch_size, nums_tokens, self.heads_nums, self.heads_dimension)
        # 1.2.交换（转置）nums_tokens和self.heads_nums所在的两个轴，这里可正确对其不同头的Q、K、V矩阵
        queries = queries.transpose(1, 2)
        keys    = keys.transpose(1, 2)
        values  = values.transpose(1, 2)
        
        # 2.计算注意力分数（本质上注意力分数通过输入一个Query词元，遍历传入整个序列的每一行向量（每一个词元向量），并与之计算点积得到注意力分数）
        # 如果采用遍历的方式（for循环）会大量消耗资源，因此通过矩阵乘法实现快速运算
        # 这里注意：遍历时一个query需要与每一个key计算点积，如果采用矩阵乘法，仍然需要确保query能够与key的每一行进行点积运算
        #          因为矩阵乘法是左侧矩阵的行与右侧矩阵的列计算点积得到对应i行、j列的元素，因此需要对右侧矩阵转置，使需要计算点积的key的向量能够正确与query中的每一行运算
        
        # !!! TIPS：包含多个轴的张量，会在张量最后两个维度上进行矩阵乘法，并对每个头重复矩阵乘法之前的所有操作，即除了最后两个维度以外，其余维度都是批量维度
        # 即：第一个轴为batch_size，第二个轴为heads_nums、第三个轴为nums_tokens，第四个轴为head_out_dim
        attention_scores: torch.Tensor = queries @ keys.transpose(2, 3)
        
        # 3.对注意力分数进行掩码
        #   在计算注意力分数的过程中，因为矩阵运算会导致当前query词元之后的词元也参与运算，为了避免未来的词元的关系权重被当前query考虑进去，需要对其之后的分数进行掩码；
        #   掩码方式为通过生成一个attentions_scores的同型矩阵，并将其对角线以下设置为0，以上设置为1，随后通过torch.Tensor.bool()将1/0转换为bool类型；
        #   最后通过masked_fill根据true/false将分数矩阵对应true的位置设置为-inf完成掩码；
        # 这里注意：
        #          queries @ keys.T进行矩阵乘法运算时，queries的第n行与keys.T的第m列（m≤n）计算点积时，才是当前query与之前的词汇之间的注意力分数（相关性的权重）
        #          即为主对角线以下的元素，以上的元素都是未来词汇，需要被掩码
        mask = self.mask.bool()[:nums_tokens, :nums_tokens]
        attention_scores.masked_fill_(mask=mask, value=(-torch.inf))
        
        # 4.使用Softmax归一化处理注意力分数得到注意力权重
        #   采用Softmax归一化利用了softmax函数的特性，将归一化结果设置为和为1（便于作为权重与values进行运算）
        #   同时softmax可以更好的处理极值、避免overflow和underflow，同时均为正值（y=softmax(o)=exp(o_{i})/Σexp(o_{k})）（e^{x}，无论x是否为负，都是整数）
        #   即：缩放点积，使用attention_scores / keys.shape[-1] ** 0.5缩放是为了避免因维度过大，分数进入softmax时，函数进入梯度极小区域，产生梯度消失
        #       因为在GPT LLM中，嵌入维度通常很大，会导致点积更大，从而反向传播时，由于softmax函数的作用导致梯度非常小，当点积增大时，softmax会更像阶跃函数，导致梯度消失
        attention_weight = torch.softmax(input=attention_scores / keys.shape[-1] ** 0.5, dim=-1)
        
        # 5.计算上下文向量
        #   此时已经得到了当前queries与相对于序列中位置的权重，需要继续根据权重计算可以取值的大小，确定哪个值更应该被取到（查询到value）
        #   同时需要将所有头的上下文转置回最开始的Q、K、V张量的形状，便于后续重塑（展平）为此注意力机制预期需要输出的形状
        context_vector = (attention_weight @ values).transpose(1, 2)
        context_vector = context_vector.contiguous().view(batch_size, nums_tokens, self.out_dimension)
        
        # 6.最终输出前需要再经过一个线性层
        return self.out_layer(context_vector)

def generate(model: nn.Module, idx: torch.Tensor, max_new_tokens: int, context_size: int, temperature=0.0, top_k = None, eos_id=None) -> torch.Tensor:
    """支持温度缩放和top-k的词元生成

    Args:
        model (torch.Module): LLM
        idx (torch.Tensor): 初始上下文
        max_new_tokens (int): 允许生成的最大token数
        context_size (int): 上下文长度
        temperature (float, optional): 温度系数. Defaults to 0.0.
        top_k (_type_, optional): top-k. Defaults to None.
        eos_id (_type_, optional): 停止词元. Defaults to None.

    Returns:
        torch.Tensor: 生成结果
    """
    for _ in range(max_new_tokens):
        # 根据允许生成的最大上下文长度遍历，确定可以生成的词元数量
        idx_cond = idx[:, -context_size:] # 根据上下文限制切分输入token
        with torch.no_grad():
            logits: torch.Tensor = model(idx_cond)
        logits = logits[:, -1, :] # 只获取最后一个token（LLM会针对输入序列的每一个词元生成一组logits向量，对于下一个词预测只需要获取最后一个向量即可）
        if top_k is not None:
            # 使用Top-k采用筛选logits
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1] # top-k中的最小词元
            logits = torch.where(logits < min_val, torch.tensor(float('-inf')).to(logits.device), logits) # top-k掩码
        if temperature > 0.0:
            # 使用温度缩放
            logits = logits / temperature # 温度缩放
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1) # 概率采样
        else:
            # 使用贪心解码
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.argmax(probs, dim=-1, keepdim=True)
        if idx_next == eos_id:
            # 如果遇到序列结束词元则提前停止生成
            break
        idx = torch.cat((idx, idx_next), dim=-1)
    return idx

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

# 加载GPT-2参数
from gpt_download import download_and_load_gpt2
settings, params = download_and_load_gpt2(model_size="124M", models_dir="gpt2")

model_configs = {
    "gpt2-small(124M)":  {"emb_dim": 768,  "n_layers": 12, "n_heads": 12},
    "gpt2-medium(355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large(774M)":  {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl(1558M)":    {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}

# 更新之前使用的GPT_CONFIG_124M
model_name = "gpt2-small(124M)"
NEW_CONFIG = GPT_CONFIG_124M.copy()
NEW_CONFIG.update(model_configs[model_name])
NEW_CONFIG.update("context_length", 1024) # 原始GPT-2模型使用1024个词元长度进行训练
NEW_CONFIG.update("qkv_bias", True)

tokenizer = tiktoken.get_encoding("gpt2")

torch.manual_seed(123)
token_ids = generate(
    model=GPTModel2(cfg=NEW_CONFIG),
    idx=text_to_token_ids("Every effort moves you", tokenizer=tokenizer),
    max_new_tokens=15,
    context_size=NEW_CONFIG['context_length'],
    top_k=25,
    temperature=1.4
)

print("Output text:\n", token_ids_to_text(token_ids=token_ids, tokenizer=tokenizer))
    
# torch.manual_seed(123)
# model = GPTModel2(cfg=GPT_CONFIG_124M)

# total_params = sum(p.numel() for p in model.parameters())
# print(f"Total number of parameters: {total_params}")
# print("Token Embeddings Layer Shape:", model.token_embeddings.weight.shape)
# print("Output Layer Shape:", model.out_head.weight.shape)

# batch = torch.tensor([
#     [6109, 3626, 6100, 345],
#     [6109, 1110, 6622, 257]
# ])

# out: torch.Tensor = model(batch)
# print("Input Batch:\n", batch)
# print("Output Shape:\n", out.shape)
# print(out)