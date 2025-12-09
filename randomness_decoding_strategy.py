import torch
import matplotlib.pyplot as plt
# 初始化示例词汇表
vocab = {
    'closer': 0,
    'every' : 1,
    'effort': 2,
    'forward':3,
    'inches': 4,
    'moves' : 5,
    'pizza' : 6,
    'toward': 7,
    'you'   : 8
}
inverse_vocab = {v: k for k, v in vocab.items()}
# 假设LLM起始上下文为"every effort moves you"，并生成了以下词元的logits
next_token_logits = torch.tensor([4.51, 0.89, -1.90, 6.75, 1.63, -1.62, -1.89, 6.28, 1.79])

# 若采用贪婪解码（argmax）面对相同的上下文会永远生成相同的预测词元
# 1.先通过softmax函数将logits转换成概率
probas = torch.softmax(next_token_logits, dim=0)
# 2.再通过argmax函数获取与生成的词元对应的词元ID
next_token_id = torch.argmax(probas).item()
# 3.最后通过反向词汇表将其映射回文本
print(inverse_vocab[next_token_id])

# 采用概率采样过程，使用PyTorch中的multinomial替换argmax
# multinomial函数按照其概率分数采样下一个词元
def print_sampled_tokens(probas):
    torch.manual_seed(123)
    sample = [torch.multinomial(probas, num_samples=1).item() for i in range(1_000)]
    sampled_ids = torch.bincount(torch.tensor(sample))
    for i, freq in enumerate(sampled_ids):
        print(f"{freq} × {inverse_vocab[i]}")

print_sampled_tokens(probas=probas)

def softmax_with_temperature(logits, temperature):
    scaled_logits = logits / temperature
    return torch.softmax(scaled_logits, dim=0)

temperatures = [1, 0.1, 5]
scaled_probas = [softmax_with_temperature(next_token_logits, T) for T in temperatures]
x = torch.arange(len(vocab))

bar_width = 0.15 # 设置条形图的宽度
fig, ax = plt.subplots(figsize=(5, 3)) # 一次性创建一个图形（Figure）和多个坐标轴（Axes）对象
for i, T in enumerate(temperatures):
    rects = ax.bar(x+i * bar_width, scaled_probas[i], bar_width, label=f'Temperature={T}')
ax.set_ylabel('Probability')
ax.set_xticks(x)
ax.set_xticklabels(vocab.keys(), rotation=90)
ax.legend()
plt.tight_layout()
plt.show()