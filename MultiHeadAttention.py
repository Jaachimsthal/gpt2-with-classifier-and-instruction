import torch.nn as nn
import torch

class MultiHeadAttention(nn.Module):
    """多头因果注意力

    Args:
        nn (nn.Module): 父类
    """
    
    def __init__(self, d_in: int, d_out: int, context_length: int, dropout_rate: float, num_heads: int, qkv_bias: bool = False):
        """初始化多头因果注意力

        Args:
            d_in (int): 词元输入维度
            d_out (int): 上下文向量输出维度
            context_length (int): 序列长度
            dropout_rate (float): 暂退率
            num_heads (int): 头数
            qkv_bias (bool, optional): qkv矩阵偏置. Defaults to False.
        """
        super().__init__()
        
        assert (d_out % num_heads == 0), \
            "d_out must be divisible by num_heads"
        
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads # 减少投影维度以匹配所需的输出维度
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out) # 使用一个线性层来组合头的输出
        self.dropout = nn.Dropout(dropout_rate)
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播

        Args:
            x (torch.Tensor): 输入张量

        Returns:
            torch.Tensor: 输出张量
        """
        b, num_tokens, d_in = x.shape # 形状：(b, num_tokens, d_out)=>(2, 6, 4)
        
        # 当前keys、queries、values的张量形状为(b, num_tokens, d_out)
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)
        
        # 重塑形状为：(b, num_tokens, self.num_heads, self.head_dim) => (2, 6, 2, 1)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        
        # 转置形状为：b, num_heads, num_tokens, head_dim（交换1轴和2轴）=> (2, 2, 6, 1)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)
        
        # !!! TIPS：包含多个轴的张量，会在张量最后两个维度上进行矩阵乘法
        attn_scores = queries @ keys.transpose(2, 3) # 计算每个头的点积 => (2, 2, 1, 6)，使用(2, 2, 6, 1) @ (2, 2, 1, 6) => (2, 2, 6, 6)
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        
        attn_scores.masked_fill_(mask_bool, -torch.inf)
        
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        context_vec = (attn_weights @ values).transpose(1, 2)
        
        context_vec = context_vec.contiguous().view(
            b, num_tokens, self.d_out
        )
        
        context_vec = self.out_proj(context_vec)
        return context_vec

inputs = torch.tensor([
    [0.43, 0.15, 0.89], # Your(x^1)
    [0.55, 0.87, 0.66], # journey(x^2)
    [0.57, 0.85, 0.64], # starts(x^3)
    [0.22, 0.58, 0.33], # with(x^4)
    [0.77, 0.25, 0.10], # one(x^5)
    [0.05, 0.80, 0.55]  # step(x^6)
])

A = torch.randn(2, 3, 4, 5, 6)
B = torch.randn(2, 3, 4, 6, 8)
print((A@B).shape)

batch = torch.stack((inputs, inputs), dim=0)
torch.manual_seed(123)
batch_size, context_length, d_in = batch.shape
print("Batch's Shape:", batch.shape)
print("Batch:", batch)

d_out = 2
mha = MultiHeadAttention(d_in, d_out, context_length, 0.0, num_heads=2)
context_vecs = mha(batch)
print("Context Vectors Shape:", context_vecs.shape)
print("Context Vectors:", context_vecs)
# 输入的Batch.Shape=(2,6,3)、输出的Batch.Shape=(2,6,2)
# 需要利用多维张量模拟出多个单头因果自注意力层