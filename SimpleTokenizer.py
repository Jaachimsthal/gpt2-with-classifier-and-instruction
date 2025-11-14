import re

class SimpleTokenizerV1:
    """词元及词元ID的encode和decode
    """
    def __init__(self, vocab):
        """构造方法 将词汇表保存为类属性方便访问，创建词汇表和逆向词汇表

        Args:
            vocab (set): 词汇表集合
        """
        self.str_to_int = vocab # 词汇表
        self.int_to_str = {i: s for s, i in vocab.items()} # 逆向词汇表
        
    def encode(self, text: str) -> list[int]:
        """处理输入文本，将其转换为词元ID

        Args:
            text (str): 文本

        Returns:
            list[int]: 词元ID列表
        """
        preprocessed = re.split(r'([,.?_!"()\']|--|\s)', text)
        preprocessed = [
            item.strip() for item in preprocessed if item.strip()
        ]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids
    
    def decode(self, ids: list[int]) -> str:
        """将词元ID转换回文本

        Args:
            ids (list[int]): 词元ID

        Returns:
            str: 文本
        """
        text = " ".join([self.int_to_str[i] for i in ids])
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text
    
class SimpleTokenizerV2:
    """能够处理未知词元及词元ID的encode和decode
    """
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = { i:s for s,i in vocab.items() }
    
    def encode(self, text: str) -> list[int]:
        """将文本转换为词元ID

        Args:
            text (str): 文本

        Returns:
            list[int]: 词元ID列表
        """
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        # 使用<|unk|>词元替换未知单词
        preprocessed = [item if item in self.str_to_int else "<|unk|>" for item in preprocessed]
        
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids
    
    def decode(self, ids: list[int]) -> str:
        """词元ID列表转文本

        Args:
            ids (list[int]): 词元ID列表

        Returns:
            str: 文本
        """
        text = " ".join([self.int_to_str[i] for i in ids])
        text = re.sub(r'\s+([,.:;?!"()\'])', r'\1', text)
        return text