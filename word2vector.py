import torch
from GPTDataset import create_dataloader_v1

# 假设存在4个ID分别是2、3、5、1的输入词元，将id转换为张量
input_ids = torch.tensor([2, 3, 5, 1])
# 设置一个仅包含6个单词的词汇表，创建维度为3的嵌入
vocab_size = 6
output_dim = 3
# 使用词汇表大小和嵌入维度实例化一个嵌入层
torch.manual_seed(123)
# num_embeddings表示行，共6行，embedding_dim表示嵌入维度；每行对应词汇表的一个词元，每一列对应一个嵌入维度
embedding_layer = torch.nn.Embedding(num_embeddings=vocab_size, embedding_dim=output_dim)
# 嵌入层权重由小的随机数构成，其权重将在训练过程被优化
print(embedding_layer.weight)
# 应用词元ID到嵌入层上，以便获取嵌入向量，可以看到嵌入ID为3的嵌入向量与嵌入矩阵中的第4行完全一样（Python索引从0开始），嵌入层实质上执行的是一种查找操作，根据词元ID从嵌入层的权重矩阵中检索出相应的行
print(embedding_layer(torch.tensor([3])))

# 更实际、更实用的嵌入维度，将容量为50257的词表编码为256维的向量表示
vocab_size = 50257
output_dim = 256
# 假设输入批次为8，每批包含4个词元的输入，结果将是8*4*256的张量
token_embedding_layer = torch.nn.Embedding(num_embeddings=vocab_size, embedding_dim=output_dim)
# 实例化数据加载器
max_length = 4
filepath = "the-verdict.txt"
# 读取短篇小说文本
with open(file=filepath, mode='r', encoding='utf-8') as file:
    raw_text = file.read()
# 加载输入数据，词元ID张量为8*4，数据批次包含8个样本、每个样本由4个词元组成
dataloader = create_dataloader_v1(raw_text, batch_size=8, max_length=max_length, stride=max_length, shuffle=False)
data_iter = iter(dataloader)
inputs, targets = next(data_iter)
print("Token IDs:\n", inputs)
print("\nInputs shape:\n", inputs.shape)
# 执行嵌入，将每个词元ID嵌入到一个256维的向量中，当前张量的维度是8*4*256
token_embeddings = token_embedding_layer(inputs)
print(token_embeddings.shape)
# 绝对值位置嵌入，只需要创建一个维度与token_embedding_layer相同的嵌入层即可，因为绝对位置嵌入只需要为每一次嵌入添加一个不同的位置信息即可，位置信息依靠embedding层的不同权重生成，并在训练中更新权重
context_length = max_length
# 形状与词元ID嵌入层一致
pos_embedding_layer = torch.nn.Embedding(num_embeddings=context_length, embedding_dim=output_dim)
# torch.arange将生成一个从0到context_length的张量1*[0:context_length]
pos_embeddings = pos_embedding_layer(torch.arange(context_length))
print(pos_embeddings.shape)
# 创建输入张量
input_embeddings = token_embeddings + pos_embeddings