import urllib.request
import re
import SimpleTokenizer
import tiktoken
from GPTDataset import create_dataloader_v1

# 下载文本
# url = ("https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt")
filepath = "the-verdict.txt"
# urllib.request.urlretrieve(url=url, filename=filepath)

# 读取短篇小说文本
with open(file=filepath, mode='r', encoding='utf-8') as file:
    raw_text = file.read()
print("Total number of character:", len(raw_text)) # 字符总数
print(raw_text[:99]) # 前100个字符

# 将包含20479个字符的文本分割为独立的单词和特殊字符，用于后续转换为嵌入向量并用于LLM训练
preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]
print(len(preprocessed))
print(preprocessed[:10])

# 创建包含唯一词元的列表，并按照字母排序
preprocessed.extend(['<|unk|>', '<|endoftext|>'])
all_words = sorted(set(preprocessed))
vocab_size = len(all_words)
print(vocab_size)

# 创建词汇表
vocab = {token: integer for integer, token in enumerate(all_words)}
# 打印词汇表前50行
for i, item in enumerate(vocab.items()):
    print(item)
    if i >= 50:
        break
    
tokenizerV1 = SimpleTokenizer.SimpleTokenizerV1(vocab=vocab)
# str to int
text = """"It's the last he painted, you know," Mrs. Gisburn said with pardonable pride."""
ids = tokenizerV1.encode(text=text)
print(ids)
# int to str
print(tokenizerV1.decode(ids=ids))

tokenizerV2 = SimpleTokenizer.SimpleTokenizerV2(vocab=vocab)
text1 = "Hello, do you like tea?"
text2 = "In the sunlit terraces of the palace."
text = "<|endoftext|>".join((text1, text2))
print(text)

# 对文本样本进行分词
print(tokenizerV2.decode(tokenizerV2.encode(text=text)))

# 实例化分词器
tokenizer = tiktoken.get_encoding("gpt2")
text = (
    "Hello, do you like tea? <|endoftext|> In the sunlit terraces"
    "of someunknownPlace."
)
# 编码text
integers = tokenizer.encode(text=text, allowed_special={"<|endoftext|>"})
print(integers)
# 解码text
strings = tokenizer.decode(tokens=integers)
print(strings)

unk_integer = tokenizer.encode(text="Akwirw ier")
print(unk_integer)
unk_strings = tokenizer.decode(tokens=unk_integer)
print(unk_strings)

# 滑动窗口数据采样
with open(filepath, 'r', encoding="utf-8") as file:
    raw_text = file.read()

# BPE分词器后训练集中的词元总数
enc_text = tokenizer.encode(raw_text)
print(len(enc_text))

# 从数据集中移除前50个单词用于生成文本段落
enc_sample = enc_text[50:]
context_size = 4
x = enc_sample[:context_size]
y = enc_sample[1:context_size+1]
print(f"x:{x}")
print(f"y:     {y}")

for i in range(1, context_size+1):
    context = enc_sample[:i]
    desired = enc_sample[i]
    print(context, "----->", desired)
    
dataloader = create_dataloader_v1(raw_text, batch_size=8, max_length=8, stride=4, shuffle=False)
data_iter = iter(dataloader)
inputs, targets = next(data_iter)
print("Inputs:\n", inputs)
print("Targets:\n", targets)