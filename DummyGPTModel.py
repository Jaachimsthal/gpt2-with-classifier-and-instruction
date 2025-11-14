import torch
import torch.nn as nn
import tiktoken
import matplotlib.pyplot as plt
from MultiHeadAttention import MultiHeadAttention

GPT_CONFIG_124M = {
    'vocab_size': 50527, # 词汇表大小
    'context_length': 1024, # 上下文维度
    'emb_dim': 768, # 嵌入维度
    'n_heads': 12, # 注意力头的数量
    'n_layers': 12, # 层数
    'drop_rate': 0.1, # 暂退率
    'qkv_bias': False # Q、K、V偏置
}

class DummpyGPTModel(nn.Module):
    """Dummpy GPT Model

    Args:
        nn (nn.Module): 衍生类
    """
    def __init__(self, cfg: dict[str, any]):
        """初始化模型

        Args:
            cfg (dict[str, any]): 模型参数
        """
        super().__init__()
        self.tok_emb    = nn.Embedding(cfg['vocab_size'], cfg['emb_dim'])
        self.pos_emb    = nn.Embedding(cfg['context_length'], cfg['emb_dim'])
        self.drop_emb   = nn.Dropout(cfg['drop_rate'])
        self.trf_blocks = nn.Sequential(
            *[DummyTransformerBlock(cfg) for _ in range(cfg['n_layers'])]
        )
        self.final_norm = DummyLayerNorm(cfg['emb_dim'])
        self.out_head   = nn.Linear(cfg['emb_dim'], cfg['vocab_size'], bias=False)
    
    def forward(self, in_idx: torch.Tensor) -> torch.Tensor:
        """前向传播

        Args:
            in_idx (torch.Tensor): 输入参数
        """
        batch_size, seq_len = in_idx.shape
        # 词嵌入+位置编码构成输入
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(
            torch.arange(seq_len, device=in_idx.device)
        )
        x = tok_embeds + pos_embeds
        # 暂退
        x = self.drop_emb(x)
        # 注意力块
        x = self.trf_blocks(x)
        # 层归一化
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits
        
class DummyTransformerBlock(nn.Module):
    """注意力块

    Args:
        nn (nn.Module): 衍生类
    """
    def __init__(self, cfg):
        super().__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播

        Args:
            x (torch.Tensor): _description_

        Returns:
            torch.Tensor: _description_
        """
        return x
    
class DummyLayerNorm(nn.Module):
    """层归一化

    Args:
        nn (nn.Module): 衍生类
    """
    def __init__(self, cfg):
        super().__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """_summary_

        Args:
            x (torch.Tensor): _description_

        Returns:
            torch.Tensor: _description_
        """
        return x

class LayerNorm(nn.Module):
    """层归一化

    Args:
        nn (nn.Module): 衍生类
    """
    def __init__(self, emb_dim: int):
        """初始化参数

        Args:
            emb_dim (int): 嵌入维度
        """
        super().__init__()
        self.eps = 1e-5 # 小常数（epsilon）在归一化过程中添加到方差上，防止除0错误
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))
        # scale和shift是两个可训练参数（与输入维度相同），若训练过程中发现调整参数可改善模型的训练任务表现，LLM会自动进行调整
        # 使得模型能够学习适合其数据处理的最佳缩放和偏移
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播（mean=0, var=1）

        Args:
            x (torch.Tensor): 待归一化张量

        Returns:
            torch.Tensor: 归一化张量
        """
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        
        norm_x = (x-mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift

class GELU(nn.Module):
    """GELU激活函数

    Args:
        nn (nn.Module): 父类
    """
    def __init__(self):
        """初始化函数
        """
        super().__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """GELU计算（更小计算量的近似实现）

        Args:
            x (torch.Tensor): 输入

        Returns:
            torch.Tensor: 激活输出
        """
        return 0.5 * x * (1+torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) * (x + 0.044715 * torch.pow(x, 3))
        ))

class FeedForward(nn.Module):
    """前馈神经网络模块

    Args:
        nn (nn.Module): 父类
    """
    def __init__(self, cfg):
        """初始化网络结构（输入层+隐藏层+输出层）（两个线性层和一个GELU激活函数组成）

        Args:
            cfg (dict): 参数配置
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
            x (torch.Tensor): 输入

        Returns:
            torch.Tensor: 输出
        """
        return self.layers(x)
    
class ResidualConnectionExample(nn.Module):
    """支持快捷连接的多层感知机

    Args:
        nn (nn.Module): PyTorch模块
    """
    def __init__(self, layer_sizes: list[int], use_shortcut: bool) -> None:
        """初始化网络结构

        Args:
            layer_sizes (list[int]): 层节矩阵维度
            use_shortcut (bool): 是否开启快捷连接
        """
        super().__init__()
        self.use_shortcut = use_shortcut
        self.layers = nn.ModuleList([
            nn.Sequential(nn.Linear(layer_sizes[0], layer_sizes[1]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[1], layer_sizes[2]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[2], layer_sizes[3]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[3], layer_sizes[4]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[4], layer_sizes[5]), GELU())
        ])
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播

        Args:
            x (torch.Tensor): 输入

        Returns:
            torch.Tensor: 输出
        """
        for layer in self.layers:
            layer_output = layer(x)
            if self.use_shortcut and x.shape == layer_output.shape:
                # 开启快捷连接并且输入张量形状与层
                x = x + layer_output
            else:
                x = layer_output
        return x

class TransformerBlock(nn.Module):
    """Transformer块

    Args:
        nn (nn.Module): PyTorch Module
    """
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg['emb_dim'],
            d_out=cfg['emb_dim'],
            context_length=cfg['context_length'],
            num_heads=cfg['n_heads'],
            dropout_rate=cfg['drop_rate'],
            qkv_bias=cfg['qkv_bias']
        )
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg['emb_dim'])
        self.norm2 = LayerNorm(cfg['emb_dim'])
        self.drop_shortcut = nn.Dropout(cfg['drop_rate'])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播

        Args:
            x (torch.Tensor): 输入序列

        Returns:
            torch.Tensor: 输出
        """
        # 注意力块中添加快捷连接
        shortcut = x
        x = self.norm1(x) # 层归一化（前层归一化）Pre-LayerNorm
        x = self.att(x) # 多头因果自注意力
        x = self.drop_shortcut(x) # 暂退
        x = x + shortcut
        
        # 前馈层中添加快捷连接
        shortcut = x
        x = self.norm2(x) # 层归一化（后层归一化）Post-LayerNorm
        x = self.ff(x) # 前馈神经网络
        x = self.drop_shortcut(x) # 暂退
        x = x + shortcut
        
        # 两次Dropout应用于多头因果自注意力层和前馈神经网络层之后，以通过正则化避免过拟合
        # 两层使用了快捷连接以避免梯度消失
        return x


# 创建分词器
tokenizer = tiktoken.get_encoding("gpt2")
batch = []
txt1 = "Every effort moves you"
txt2 = "Every day holds a"

batch.append(torch.tensor(tokenizer.encode(txt1)))
batch.append(torch.tensor(tokenizer.encode(txt2)))

batch = torch.stack(batch, dim=0) # List转tensor

# 初始化参数量为1.24亿的DummyGPTModel，并将分词后的批次数据传入
torch.manual_seed(123)
model = DummpyGPTModel(GPT_CONFIG_124M)
logits = model(batch)
print("Output shape:", logits.shape)
print(logits)

# 层归一化
batch_example = torch.randn(2, 5)
layer = nn.Sequential(nn.Linear(5, 6), nn.ReLU())
out = layer(batch_example)
print(out)
mean = out.mean(dim=-1, keepdim=True)
var = out.var(dim=-1, keepdim=True)
print("Mean:\n", mean)
print("Variance:\n", var)
# 对输出层输出进行归一化
out_norm = (out-mean) / torch.sqrt(var) # (输出-均值) / 标准差
torch.set_printoptions(sci_mode=False) # 关闭打印显示科学计数法
mean = out_norm.mean(dim=-1, keepdim=True)
var = out_norm.var(dim=-1, keepdim=True)
print("Mean:\n", mean)
print("Variance:\n", var)

ln = LayerNorm(emb_dim=5)
out_ln = ln(batch_example)
print("Mean:\n", out_ln.mean(dim=-1, keepdim=True))
print("Variance:\n", out_ln.var(dim=-1, keepdim=True, unbiased=False))

# 比较GELU和ReLU的区别
gelu, relu = GELU(), nn.ReLU()
x = torch.linspace(-3, 3, 100)
y_gelu, y_relu = gelu(x), relu(x)
plt.figure(figsize=(8, 3))

for i, (y, label) in enumerate(zip([y_gelu, y_relu], ["GELU", "ReLU"]), 1):
    plt.subplot(1, 2, i)
    plt.plot(x, y)
    plt.title(f"{label} activation function")
    plt.xlabel("x")
    plt.ylabel(f"{label} (x)")
    plt.grid(True)
plt.tight_layout()
plt.show()

# FFN TEST
ffn = FeedForward(GPT_CONFIG_124M)
x = torch.rand(2, 3, 768)
out = ffn(x)
print("Out:\n", out.shape)

torch.manual_seed(123)
model_without_shortcut = ResidualConnectionExample(layer_sizes=[3,3,3,3,3,1], use_shortcut=False)
sample_input = torch.tensor([[1., 0., -1.]])

def print_gradients(model: nn.Module, x: torch.Tensor) -> None:
    """反向传播计算梯度

    Args:
        model (nn.Module): PyTorch Module
        x (torch.Tensor): Input Tensor
    """
    output = model(x)
    target = torch.tensor([[0.]])
    
    loss = nn.MSELoss()
    loss = loss(output, target)
    
    # 计算偏导->梯度
    loss.backward()
    for name, param in model.named_parameters():
        if 'weight' in name:
            print(f"{name} has gradient mean of {param.grad.abs().mean().item()}")

print_gradients(model_without_shortcut, sample_input)

model_with_shortcut = ResidualConnectionExample(layer_sizes=[3,3,3,3,3,1], use_shortcut=True)
print_gradients(model_with_shortcut, sample_input)

# Transformer块测试
torch.manual_seed(123)
x = torch.rand(2, 4, 768) # batch=2、row=4、dim/row=768
block = TransformerBlock(cfg=GPT_CONFIG_124M)
output = block(x)

print("Input Shape:\n", x.shape)
print("Ouput Shape:\n", output.shape)