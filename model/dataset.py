import json
import random
import re

import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
from sklearn.model_selection import train_test_split
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class PretrainDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=512):
        """ 初始化PretrainDataset类
        :param df: 数据框
        :param tokenizer: 分词器
        :param max_length: 最大序列长度
        """
        super().__init__()
        self.df = df  # 数据
        self.tokenizer = tokenizer  # 分词器
        self.max_length = max_length  # 最大序列长度
        self.padding = 0  # 填充 token ID

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index: int):
        """
        根据给定的索引获取数据集中的一个样本，并对其进行预处理。

        :param index: 样本在数据集中的索引位置
        :return: 返回一个元组，包含输入序列X、目标序列Y和损失掩码loss_mask
        """
        sample = self.df.iloc[index]

        # 将样本文本前后添加特殊标记（BOS开始标记和EOS结束标记）
        text = f"{self.tokenizer.bos_token}{str(sample['text'])}{self.tokenizer.eos_token}"

        # 使用分词器对文本进行编码，获取输入ID
        input_id = self.tokenizer(text).data['input_ids'][:self.max_length]

        # 计算实际文本长度
        text_len = len(input_id)

        # 计算需要填充的长度以达到最大长度
        padding_len = self.max_length - text_len

        # 对输入ID进行填充，使其长度等于最大长度
        input_id = input_id + [self.padding] * padding_len

        # 创建损失掩码，实际文本部分为1，表示计算损失；填充部分为0，表示不计算损失
        loss_mask = [1] * text_len + [0] * padding_len
        # TODO  这里有个很好的问题：为什么loss_mask不使用self.tokenizer(text).data['attention_mask'],
        # 而是需要通过计算得到？  提示：通常这两个是相等的，但是在进行如序列标注任务时，会不相等。

        # 将输入ID转换为NumPy数组，并拆分为输入序列X和目标序列Y
        input_id = np.array(input_id)
        X = np.array(input_id[:-1]).astype(np.int64)  # 输入序列X为除去最后一个元素的序列
        Y = np.array(input_id[1:]).astype(np.int64)   # 目标序列Y为除去第一个元素的序列

        # 损失掩码同样去除第一个元素，与X和Y的长度保持一致
        loss_mask = np.array(loss_mask[1:]).astype(np.int64)

        # 将NumPy数组转换为PyTorch张量并返回
        return torch.from_numpy(X), torch.from_numpy(Y), torch.from_numpy(loss_mask)



class SFTDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=1024, prompt_max_len=512, answer_max_len=256):
        super().__init__()
        self.df = df
        self.max_length = max_length
        self.prompt_max_len = prompt_max_len
        self.answer_max_len = answer_max_len
        #
        self.tokenizer = tokenizer
        self.padding = 0
        self.bos_id = self.tokenizer('<s>assistant').data['input_ids']

    def __len__(self):
        return self.df.shape[0]

    def find_sublist_index(self, main_list, sub_list) -> int:

        last_index = -1
        for i in range(len(main_list) - len(sub_list) + 1):
            if main_list[i:i + len(sub_list)] == sub_list:
                last_index = i
        return last_index

    def safe_eval(self, s):
        try:
            res = eval(s)
        except Exception as e:
            return []
        return res

    def __getitem__(self, index: int):
        #
        sample = self.df.iloc[index]
        history = self.safe_eval(sample['history'])
        q = str(sample['q'])
        a = str(sample['a'])

        messages = []
        for history_message in history:
            if len(history_message) <= 1:
                continue
            messages.append(
                {"role": 'user', "content": str(history_message[0])[:self.max_length // 2]}
            )
            messages.append(
                {"role": 'assistant', "content": str(history_message[1])[:self.max_length // 2]}
            )

        messages += [
            {"role": "user", "content": q},
            {"role": "assistant", "content": a},
        ]
        new_prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        input_id = self.tokenizer(new_prompt).data['input_ids'][:self.max_length]

        # 实际长度
        question_length = self.find_sublist_index(input_id, self.bos_id) + len(self.bos_id)
        # 没满最大长度的剩余部分
        padding_len = self.max_length - len(input_id)
        input_id = input_id + [self.padding] * padding_len
        mask_len = len(input_id) - question_length - padding_len
        # 0表示不计算损失
        loss_mask = [0] * question_length + [1] * (mask_len) + [0] * padding_len

        input_id = np.array(input_id)
        X = np.array(input_id[:-1]).astype(np.int64)
        Y = np.array(input_id[1:]).astype(np.int64)
        loss_mask = np.array(loss_mask[1:]).astype(np.int64)

        X_tensor = torch.from_numpy(X)
        Y_tensor = torch.from_numpy(Y)
        loss_mask_tensor = torch.from_numpy(loss_mask)

        return X_tensor, Y_tensor, loss_mask_tensor


if __name__ == "__main__":
    pass
