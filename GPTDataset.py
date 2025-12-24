import torch
from torch.utils.data import Dataset, DataLoader
import tiktoken
from tiktoken.core import Encoding

class GPTDatasetV1(Dataset):
    def __init__(self, txt: str, tokenizer: Encoding, max_length: int, stride: int):
        """从数据集分词并生成token（词元ID）包括input和target两组，使用滑动窗口创建input_token_ids以及target_token_ids

        Args:
            txt (str): 输入文本
            tokenizer (tiktoken): 分词和词元ID创建
            max_length (int): 最大长度
            stride (int): _description_
        """
        self.input_ids = []
        self.target_ids = []
        
        token_ids = tokenizer.encode(txt)
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i+max_length]
            target_chunk = token_ids[i+1:i+max_length+1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))
    
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        """返回数据集的指定行

        Args:
            idx (int): 索引

        Returns:
            torch.Tensor: 张量
        """
        return self.input_ids[idx], self.target_ids[idx]
    
def create_dataloader_v1(txt: str, batch_size: int=4, max_length: int=256, stride: int=128, shuffle: bool=True, drop_last: bool=True, num_workers: int=0) -> DataLoader:
    """初始化分词器

    Args:
        txt (str): _description_
        batch_size (int, optional): _description_. Defaults to 4.
        max_length (int, optional): _description_. Defaults to 256.
        stride (int, optional): _description_. Defaults to 128.
        shuffle (bool, optional): _description_. Defaults to True.
        drop_last (bool, optional): _description_. Defaults to True.
        num_workers (int, optional): _description_. Defaults to 0.

    Returns:
        DataLoader: _description_
    """
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )