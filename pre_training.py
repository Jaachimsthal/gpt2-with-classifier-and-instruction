import torch
import torch.nn as nn
from GPTModel import GPTModel2, GPT_CONFIG_124M
import tiktoken
from generate_text import generate_text_simple
from tiktoken.core import Encoding
from GPTDataset import create_dataloader_v1

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

preplexity=torch.exp(loss)

# 计算训练集和验证集损失
file_path = "the-verdict.txt"
with open(file=file_path, mode="r", encoding="utf-8") as file:
    text_data = file.read()

total_characters = len(text_data)
total_tokens = len(tokenizer.encode(text_data))

print("Characters:", total_characters)
print("Tokens:", total_tokens)

# 生成测试集和验证集
train_ratio = 0.90
split_idx = int(train_ratio * len(text_data))
train_data = text_data[:split_idx]
valid_data = text_data[split_idx:]

# 加载数据
train_loader = create_dataloader_v1(
    train_data,
    batch_size=2,
    max_length=GPT_CONFIG_124M['context_length'],
    stride=GPT_CONFIG_124M['context_length'],
    drop_last=True,
    shuffle=True,
    num_workers=0
)

valid_loader = create_dataloader_v1(
    valid_data,
    batch_size=2,
    max_length=GPT_CONFIG_124M['context_length'],
    stride=GPT_CONFIG_124M['context_length'],
    drop_last=True,
    shuffle=True,
    num_workers=0
)

for x, y in train_loader:
    print(x.shape, y.shape)

from torch.utils.data import Dataset, DataLoader    

def calc_loss_batch(input_batch: torch.Tensor, target_batch: torch.Tensor, model: nn.Module, device: torch.device) -> torch.Tensor:
    """计算通过训练集加载器和验证集加载器返回的给定批次的交叉熵损失

    Args:
        input_batch (Dataset): _description_
        target_batch (Dataset): _description_
        model (nn.Module): _description_
        device (torch.device): _description_

    Returns:
        torch.Tensor: _description_
    """
    # 通过to(device)将数据转移到GPU上运算
    input_batch  = input_batch.to(device=device)
    target_batch = target_batch.to(device=device)
    logits: torch.Tensor = model(input_batch)
    
    # 计算交叉熵损失
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
    return loss

def calc_loss_loader(data_loader: DataLoader, model: nn.Module, device: torch.device, num_batchs=None) -> torch.Tensor:
    """计算损失

    Args:
        data_loader (DataLoader): _description_
        model (nn.Module): _description_
        device (_type_): _description_
        num_batchs (_type_, optional): _description_. Defaults to None.

    Returns:
        torch.Tensor: _description_
    """
    total_loss = 0.
    if len(data_loader) == 0:
        # 如果没有数据集直接返回float Nan
        return float("nan")
    elif num_batchs is None:
        # 没传入批次数量则根据数据集计算
        num_batchs = len(data_loader)
    else:
        # 如果设置的批次数量超过了数据加载器中的批次数，则需要减少批次，匹配数据加载器中的总批次
        num_batchs = min(num_batchs, len(data_loader))
    
    for i ,(input_batch, target_batch) in enumerate(data_loader):
        if i < num_batchs:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            # 每个批次的损失总和
            total_loss += loss.item()
        else:
            break
    # 计算平均损失
    return total_loss / num_batchs

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device=device)
with torch.no_grad():
    train_loss = calc_loss_loader(train_loader, model, device)
    valid_loss = calc_loss_loader(valid_loader, model, device)
print("Training loss:", train_loss)
print("Validation loss:", valid_loss)

# 训练LLM
def train_model_simple(
    mode: nn.Module,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_epochs: int,
    eval_freq, 
    eval_iter,
    start_context,
    tokenizer: Encoding
) -> tuple[torch.Tensor, torch.Tensor, str]:
    """训练函数

    Args:
        mode (nn.Module): _description_
        train_loader (DataLoader): _description_
        valid_loader (DataLoader): _description_
        optimizer (_type_): _description_
        device (torch.device): _description_
        num_epochs (int): _description_
        eval_freq (_type_): _description_
        eval_iter (_type_): _description_
        start_context (_type_): _description_
        tokenizer (Encoding): _description_

    Returns:
        tuple[torch.Tensor, torch.Tensor, str]: _description_
    """
    # 初始化列表以跟踪损失和所见的词元
    train_losses, valid_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1
    
    # 训练主循环
    for epoch in range(num_epochs):
        # 开启模型训练模式
        model.train()
        # 从train_loader中加载inputs和targets（均为批次，train_loader的一次迭代为一批数据）
        for input_batch, target_batch in train_loader:
            # 重置上一个批次迭代中的损失梯度
            optimizer.zero_grad()
            # 计算当前批次损失
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            # 反向传播计算梯度值（保存在计算图中）
            loss.backward()
            # 使用损失梯度更新模型权重（使用计算图中梯度更新）
            optimizer.step()
            
            tokens_seen += input_batch.numel()
            global_step += 1
            # 评估
            if global_step % eval_freq == 0:
                train_loss, valid_loss = evaluate_model(model, train_loader, valid_loader, device, eval_iter)
                train_losses.append(train_loss)
                valid_losses.append(valid_loss)
                track_tokens_seen.append(tokens_seen)
                print(
                    f"Ep {epoch+1} (Step {global_step:06d})"
                    f"Train Loss {train_loss:.3f}"
                    f"Valid Loss {valid_loss:.3f}"
                )
            # 每轮迭代结束后打印一个文本样本
            generate_and_print_sample(model, tokenizer, device, start_context)
    
    return train_losses, valid_losses, track_tokens_seen

def evaluate_model(model: nn.Module, train_loader: DataLoader, valid_loader: DataLoader, device: torch.device, eval_iter) -> tuple[torch.Tensor, torch.Tensor]:
    """打印训练集和验证集的损失，以便评估训练是否改善了模型性能

    Args:
        model (nn.Module): _description_
        train_loader (DataLoader): _description_
        valid_loader (DataLoader): _description_
        device (torch.device): _description_
        eval_iter (_type_): _description_
    """
    # 评估阶段禁用dropout以产出稳定且可复现的结果
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batchs=eval_iter)
        valid_loss = calc_loss_loader(valid_loader, model, device, num_batchs=eval_iter)
    # 评估结束重启模型训练模式
    model.train()
    return train_loss, valid_loss

def generate_and_print_sample(model: GPTModel2, tokenizer: Encoding, device: torch.device, start_context: str) -> None:
    """用于跟踪模型在训练中是否有改进

    Args:
        model (nn.Module): _description_
        tokenizer (Encoding): _description_
        device (torch.device): _description_
        start_context (_type_): _description_
    """
    model.eval()
    # 上下文长度根据模型位置嵌入层维度设置
    context_size = model.position_embeddings.weight.shape[0]
    # 以文本片段作为输入，先将其转换为词元ID
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        # 启用模型逐词生成文本（最多生成50个新token），根据位置嵌入控制输入上下文长度（tokens长度）（softmax+argmax）
        token_ids = generate_text_simple(
            model=model,
            idx=encoded,
            max_new_tokens=50,
            context_size=context_size
        )
    # 预测词元ID转文本
    decoded_text = token_ids_to_text(token_ids, tokenizer)
    print(decoded_text.replace("\n", " "))
    model.train()

torch.manual_seed(123)
model = GPTModel2(GPT_CONFIG_124M)
model.to(device=device)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=0.0004,
    weight_decay=0.1
)

num_epochs = 10
train_losses, valid_losses, tokens_seen = train_model_simple(
    model,
    train_loader,
    valid_loader,
    optimizer,
    device,
    num_epochs,
    eval_freq=5,
    eval_iter=5,
    start_context="Every effort moves you",
    tokenizer=tokenizer
)

# 绘制图表
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
def plot_losses(epoches_seen, token_seen, train_losses, val_losse):
    fig, ax1 = plt.subplots(figsize=(5,3))
    ax1.plot(epoches_seen, train_losses, label="Training loss")
    ax1.plot(epoches_seen, val_losse, linestyle="-.", label="Validation loss")
    
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax2 = ax1.twiny() # 创建共享同一个y轴的第二个x轴
    ax2.plot(token_seen, train_losses, alpha=0) # 对其刻度线的隐藏图标
    ax2.set_xlabel("Token Seen")
    fig.tight_layout()
    plt.show()
    
epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
plot_losses(epochs_tensor, tokens_seen, train_losses, valid_losses)

model.to("cpu")
model.eval()
tokenizer = tiktoken.get_encoding("gpt2")
token_ids = generate_text_simple(
    model=model,
    idx=text_to_token_ids("Every effort moves you", tokenizer=tokenizer),
    max_new_tokens=25,
    context_size=GPT_CONFIG_124M['context_length']
)

print("Output text:\n", token_ids_to_text(token_ids=token_ids, tokenizer=tokenizer))