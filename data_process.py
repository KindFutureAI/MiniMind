import csv
import itertools
import re
import json
import jsonlines
import psutil
import ujson
import numpy as np
import pandas as pd
from transformers import AutoTokenizer
from datasets import load_dataset

bos_token = "<s>"
eos_token = "</s>"


def pretrain_process(chunk_size=50000):
    """ 将大语料保存到csv文件 """
    # 初始化数据块索引
    chunk_idx = 0

    # 打开 JSON Lines 文件
    with jsonlines.open('./dataset/mobvoi_seq_monkey_general_open_corpus.jsonl') as reader:
        # 打开 CSV 文件，准备写入
        with open('./dataset/pretrain_data.csv', 'w', newline='', encoding='utf-8') as csvfile:
            # 创建 CSV writer 对象
            writer = csv.writer(csvfile)
            # 写入 CSV 文件的表头
            writer.writerow(['text'])

            while True:
                # 从 reader 中读取 chunk_size 行数据
                chunk = list(itertools.islice(reader, chunk_size))

                # 如果 chunk 为空，说明文件已读取完毕
                if not chunk:
                    break

                # 遍历当前数据块中的每一行
                for idx, obj in enumerate(chunk):
                    try:
                        # 获取当前行的 'text' 字段
                        content = obj.get('text', '')

                        # 如果文本长度超过 512，跳过该行
                        if len(content) > 512:
                            continue

                        # 将有效文本写入 CSV 文件
                        writer.writerow([content])
                    except UnicodeDecodeError as e:
                        # 如果出现 UnicodeDecodeError，打印错误信息并跳过该行
                        print(f"Skipping invalid line {chunk_idx * chunk_size + idx + 1}: {e}")
                        continue

                # 更新数据块索引
                chunk_idx += 1

                # 打印当前处理的数据块范围和结束信息
                print('chunk:', ((chunk_idx - 1) * chunk_size, chunk_idx * chunk_size), 'process end')


def sft_process(contain_history=False):
    """
    处理并写入数据到CSV文件。

    参数:
    - contain_history (bool): 是否包含历史对话记录，默认为False。
    """
    file_name = 'sft_data.csv'
    if not contain_history:
        file_name = 'sft_data_single.csv'

    def chinese_ratio(text):
        """
        计算文本中中文字符的比例。

        参数:
        - text (str): 输入文本。

        返回:
        - float: 中文字符比例。
        """
        # 匹配所有中文字符
        chinese_chars = re.findall(r'[\u4e00-\u9fff]', text)
        # 中文字符数量占比
        return len(chinese_chars) / len(text) if text else 0

    def process_and_write_data(data):
        """
        过滤并写入数据到CSV文件。

        参数:
        - data (list): 数据列表。
        """
        q_lst, a_lst, history_lst = [], [], []  # 初始化列表以存储问题、答案和历史记录

        for per in data:
            history, q, a = per['history'], per['q'], per['a']

            # 跳过不符合条件的记录
            if (contain_history and not history) or not q or not a:
                continue
            if len(q) < 10 or len(a) < 5:
                continue
            if len(q) > 256 or len(a) > 256:
                continue
            # 判断问题和答案中中文字符占比是否超过70%
            if not (chinese_ratio(q) > 0.9 and chinese_ratio(a) > 0.9):
                continue

            q_lst.append(q)
            a_lst.append(a)
            if contain_history:
                history_lst.append(history)
            else:
                history_lst.append([])

        # 创建DataFrame并追加到CSV文件
        df = pd.DataFrame({'history': history_lst, 'q': q_lst, 'a': a_lst})
        df.to_csv(f'./dataset/{file_name}', mode='a', header=False, index=False, lineterminator='\r\n')

    chunk_size = 1000  # 每次处理的记录数
    data = []

    # 初始化CSV文件
    with open(f'./dataset/{file_name}', 'w', encoding='utf-8') as f:
        f.write('history,q,a\n')

    sft_datasets = ['./dataset/sft_data_zh.jsonl']
    if not contain_history:
        sft_datasets = ['./dataset/sft_data_zh.jsonl']

    for path in sft_datasets:
        with jsonlines.open(path) as reader:
            for idx, obj in enumerate(reader):
                try:
                    data.append({
                        'history': obj.get('history', ''),
                        'q': obj.get('input', '') + obj.get('q', ''),
                        'a': obj.get('output', '') + obj.get('a', '')
                    })

                    if len(data) >= chunk_size:
                        process_and_write_data(data)
                        data = []
                except jsonlines.InvalidLineError as e:
                    print(f"Skipping invalid JSON line {idx + 1}: {e}")
                    continue

            if data:
                process_and_write_data(data)
                data = []


def rl_process():
    ################
    # Dataset
    ################

    dataset_path = ['./dataset/dpo/dpo_zh_demo.json',
                    './dataset/dpo/train_data.json',
                    './dataset/dpo/huozi_rlhf_data.json', ]

    train_dataset = load_dataset('json', data_files=dataset_path)

    def process(row):
        row["chosen"] = tokenizer.apply_chat_template(row["chosen"], tokenize=False)
        row["reject"] = tokenizer.apply_chat_template(row["rejected"], tokenize=False)
        return row

    ds = train_dataset.map(
        process,
        load_from_cache_file=False,
    )

    output_dataset_path = './dataset/dpo/train_data.json'
    ds['train'].to_json(output_dataset_path, force_ascii=False, orient='records', lines=True)


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained('./model/minimind_tokenizer', use_fast=False)
    print('tokenizer词表大小：', len(tokenizer))

    ################
    # 1: pretrain
    # 2: sft
    # 3: RL
    ################
    process_type = 1

    if process_type == 1:
        pretrain_process()
    if process_type == 2:
        sft_process(contain_history=False)
    if process_type == 3:
        rl_process()
