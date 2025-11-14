import torch

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
x_2 = inputs[1]

torch.manual_seed(123)
# 初始化Query、Key、Value权重
W_query = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_key = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_value = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)

# 计算查询向量、键向量和值向量
query_2 = x_2 @ W_query
key_2 = x_2 @ W_key
value_2 = x_2 @ W_value
print(query_2)

# 将6个输入词元从3维空间映射到2维嵌入空间
keys = inputs @ W_key     # 计算序列中所有词元的键向量
values = inputs @ W_value # 计算序列中所有词元的值向量
print("keys.shape:", keys.shape)
print("values.shape:", values.shape)

# 计算注意力分数（未归一化）
keys_2 = keys[1]
# inputs_2与W_query计算后与自身key vector计算注意力分数
atten_socres_22 = query_2.dot(keys_2)
print(atten_socres_22)

atten_scores_2 = query_2 @ keys.T
print(atten_scores_2) # x_2与全部序列词元嵌入key vector的注意力分数

# 缩放+Softmax归一化
d_k = keys.shape[-1]
# 缩放通过将注意力分数除以键向量嵌入维度的平方根
# 缩放是因为自注意力分数通过keys vector计算得到，在进入softmax运算时，如果dk维度较大会导致分数较大，从而使softmax函数进入梯度极小区域，产生梯度消失
attn_weight_2 = torch.softmax(atten_scores_2 / d_k**0.5, dim=-1)
print(attn_weight_2)

# 计算context vector
context_vec_2 = attn_weight_2 @ values
print(context_vec_2)