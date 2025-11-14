import torch

inputs = torch.tensor([
    [0.43, 0.15, 0.89], # Your(x^1)
    [0.55, 0.87, 0.66], # journey(x^2)
    [0.57, 0.85, 0.64], # starts(x^3)
    [0.22, 0.58, 0.33], # with(x^4)
    [0.77, 0.25, 0.10], # one(x^5)
    [0.05, 0.80, 0.55]  # step(x^6)
])

keys = inputs.T
print(keys)

# 实现自注意力的第一步是计算中间值W，即注意力分数，将x^2作为查询，通过点积来计算查询（Query）x^2与其他所有输入元素之间的注意力分数W（分别将Query与其他元素的输入嵌入计算点击，得到分数）
query = inputs[1]
attn_scores_2 = torch.empty(inputs.shape[0])
for i, x_i in enumerate(inputs):
    attn_scores_2[i] = torch.dot(x_i, query)
print(attn_scores_2)

# 第二步需要对注意力分数进行归一化处理以维持LLM训练的稳定性（获得总和为1的注意力分数）
attn_weights_2_tmp = attn_scores_2 / attn_scores_2.sum()
print("Attention Weights:", attn_weights_2_tmp)
print("Sum:", attn_weights_2_tmp.sum())

def softmax_naive(x: torch.Tensor) -> torch.Tensor:
    """Softmax归一化（这种简单的softmax实现在处理大输入值或小输入值时可能遇到数值稳定性问题，如上溢或下溢，建议使用PyTorch实现的Softmax）

    Args:
        x (torch.Tensor): 注意力分数

    Returns:
        torch.Tensor: 归一化后的注意力分数
    """
    return torch.exp(x) / torch.exp(x).sum(dim=0)

# Softmax的PyTorch实现
attn_weights_2 = torch.softmax(attn_scores_2, dim=0)
print("Attention weights:", attn_weights_2)
print("Sum:", attn_weights_2.sum())

# 计算context vector
context_vec_2 = torch.zeros(query.shape)
for i, x_i in enumerate(inputs):
    context_vec_2 += attn_weights_2[i] * x_i
print(context_vec_2)

# 对全部输入计算context vector
atten_scores = torch.empty(6, 6)
for i, x_i in enumerate(inputs):
    for j, x_j in enumerate(inputs):
        atten_scores[i, j] = torch.dot(x_i, x_j)
print(atten_scores)

# 矩阵加速运算
atten_scores = inputs @ inputs.T
print(atten_scores)
atten_weights = torch.softmax(atten_scores, dim=-1)
print(atten_weights)
all_context_vector = atten_weights @ inputs
print(all_context_vector)