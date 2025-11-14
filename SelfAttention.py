import torch.nn as nn
import torch

class SelfAttention_v1(nn.Module):
    """由nn.Module派生出来的类

    Args:
        nn.Module (nn.Module)
    """
    def __init__(self, d_in: int, d_out: int):
        """初始化函数，负责初始化权重维度

        Args:
            d_in (int): _description_
            d_out (int): _description_
        """
        super().__init__()
        self.W_query = nn.Parameter(torch.rand(d_in, d_out))
        self.W_key = nn.Parameter(torch.rand(d_in, d_out))
        self.W_value = nn.Parameter(torch.rand(d_in, d_out))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """自注意力机制前向传播

        Args:
            x (torch.Tensor): 输入序列词元嵌入（向量化+位置编码）

        Returns:
            torch.Tensor: 上下文向量
        """
        keys = x @ self.W_key
        queries = x @ self.W_query
        values = x @ self.W_value
        
        attn_scores = queries @ keys.T
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        
        # 创建掩码矩阵（对角线以上为0，其余元素为1的，attn_scores的同型矩阵）
        mask_simple = torch.tril(torch.ones(attn_scores.shape[0], attn_scores.shape[1]))
        mask_simple = attn_weights * mask_simple
        # 掩码矩阵重新归一化
        mask_simple_norm = mask_simple / mask_simple.sum(dim=-1, keepdim=True)
        
        context_vec = mask_simple_norm @ values
        return context_vec
    
class SelfAttention_v2(nn.Module):
    def __init__(self, d_in: int, d_out: int, qkv_bias: bool = False):
        """初始化自注意力query、key、value层

        Args:
            d_in (int): 线性层输入维度
            d_out (int): 线性层输出维度
            qkv_bias (bool, optional): 是否启用偏置项. Defaults to False.
        """
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """自注意力前向传播

        Args:
            x (torch.Tensor): 输入张量

        Returns:
            torch.Tensor: 上下文张量
        """
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)
        
        attn_scores = queries @ keys.T
        # 创建掩码矩阵
        mask = torch.triu(torch.ones(attn_scores.shape[0], attn_scores.shape[1]), diagonal=1)
        # 进行掩码
        masked = attn_scores.masked_fill(mask.bool(), -torch.inf)
        # 掩码矩阵重新归一化
        attn_weights = torch.softmax(masked / keys.shape[-1]**0.5, dim=-1)
        # 进行dropout掩码
        dropout = torch.nn.Dropout(0.5)
        attn_weights = dropout(attn_weights)
        print(attn_weights)
        
        context_vec = attn_weights @ values
        return context_vec

class CausalAttention(nn.Module):
    """因果注意力机制

    Args:
        nn (Module): torch Module
    """
    def __init__(self, d_in: int, d_out: int, context_length: int, dropout_rate: float, qkv_bias=False):
        """初始化query、key、value矩阵以及dropout层和掩码层

        Args:
            d_in (int): _description_
            d_out (int): _description_
            context_length (int): _description_
            dropout_rate (float): _description_
            qkv_bias (bool, optional): _description_. Defaults to False.
        """
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout_rate) # attention weight dropout
        self.register_buffer(
            'mask',
            torch.triu(torch.ones(context_length, context_length), diagonal=1),
        ) # 创建掩码矩阵（0，1元素构成），register_buffer可使缓冲区会与模型一起移动到适当的设备上（CPU或GPU），在训练时无需手动确保张量和模型在同一设备上
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """因果注意力前向传播

        Args:
            x (torch.Tensor): 嵌入+位置信息编码输入

        Returns:
            torch.Tensor: Context Vector
        """
        # 2, 6, 3
        b, num_tokens, d_in = x.shape
        keys    = self.W_key(x)
        queries = self.W_query(x)
        values  = self.W_value(x)
        
        attn_scores = queries @ keys.transpose(1, 2)
        attn_scores.masked_fill_(self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)
        
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        context_vec = attn_weights @ values
        return context_vec

class MultiHeadAttentionWrapper(nn.Module):
    """多头因果注意力

    Args:
        nn (Module): PyTorch Module
    """
    def __init__(self, d_in: int, d_out: int, context_length: int, dropout_rate: float, num_heads: int, qkv_bias: bool=False):
        """初始化

        Args:
            d_in (int): 权重参数输入维度
            d_out (int): 权重参数输出维度
            context_length (int): 上下文长度（序列长度）
            dropout_rate (float): 暂退率
            num_heads (int): 头数
            qkv_bias (bool, optional): 启用Query、Key、Value偏置. Defaults to False.
        """
        super().__init__()
        self.heads = nn.ModuleList(
            [CausalAttention(d_in=d_in, d_out=d_out, context_length=context_length, dropout_rate=dropout_rate, qkv_bias=qkv_bias) for _ in range(num_heads)]
        ) # nn.ModuleList继承自nn.Module对象，是用于存储和管理多个子模型的容器类，类似于Python列表，但可自动将子模型注册到父模型中，使其支持训练和优化
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播

        Args:
            x (torch.Tensor): 输入张量

        Returns:
            torch.Tensor: 上下文张量
        """
        # 用于在指定维度上拼接多个张量，将一组张量沿着指定的维度连接在一起，形成一个新的张量
        # torch.cat与torch.stack不同，前者不会新增维度，而是在现有维度上进行拼接
        # dim=0 沿行方向拼接（垂直方向）、dim=1沿列方向拼接（水平方向）、dim=2沿第三维度拼接（通道或深度）
        # dim=-1：表示最后一个维度（最常用）、dim=-2：表示倒数第二个维度
        # dim=None：将所有输入张量视为一维向量进行拼接
        return torch.cat([head(x) for head in self.heads], dim=-1)

inputs = torch.tensor([
    [0.43, 0.15, 0.89], # Your(x^1)
    [0.55, 0.87, 0.66], # journey(x^2)
    [0.57, 0.85, 0.64], # starts(x^3)
    [0.22, 0.58, 0.33], # with(x^4)
    [0.77, 0.25, 0.10], # one(x^5)
    [0.05, 0.80, 0.55]  # step(x^6)
])

d_in = 3  # 输入维度
d_out = 2 # 输出维度

torch.manual_seed(123)
sa_v1 = SelfAttention_v1(d_in, d_out)
print(sa_v1(inputs))

torch.manual_seed(789)
sa_v2 = SelfAttention_v2(d_in, d_out)
print(sa_v2(inputs))

batch = torch.stack((inputs, inputs), dim=0) # 合成两个张量到一个张量中
print("Batch Shapes:", batch.shape)

context_length = batch.shape[1]
ca = CausalAttention(d_in, d_out, context_length, 0.0)
context_vec = ca(batch)
print("Context Vectors Shape:", context_vec.shape)

mha = MultiHeadAttentionWrapper(d_in=d_in, d_out=d_out, context_length=context_length, dropout_rate=0.0, num_heads=2)
mha_context_vec = mha(batch)
print("mha context vector:", mha_context_vec)
print("mha context vector shape:", mha_context_vec.shape)