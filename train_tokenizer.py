import random
from pprint import pprint
from tqdm import tqdm
from transformers import AutoTokenizer
import json
from datasets import load_dataset
from tokenizers import (
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    processors,
    trainers,
    Tokenizer,
)
import os

random.seed(42)  # 设置随机种子，确保结果可复现

def train_tokenizer():
    """
    训练自定义的分词器
    """
    # 0 基础参数
    data_path = './dataset/tokenizer_train.jsonl'  # 数据文件路径

    # 定义特殊token
    # 特殊token列表 <ubk>: unkown; <s>: start of sentence; </s>: end of sentence
    # TODO 对于unk, 需要记录下来，之后根据这个词的情况，重训练词表
    special_tokens = ["<unk>", "<s>", "</s>"]

    # 1 读取JSONL文件并提取文本数据
    def read_texts_from_jsonl(file_path):
        """ 从JSONL文件中逐行读取文本数据
        :param file_path: JSONL文件路径
        :yield: 文本数据
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                yield data['text']
    #   读取文本数据
    texts = read_texts_from_jsonl(data_path)  # 从JSONL文件中读取文本数据

    # 2 初始化tokenizer
    tokenizer = Tokenizer(models.BPE())  # 使用BPE模型初始化tokenizer
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)  # 设置前处理分词器
    print(f"debug: 第二步 初始化tokenizer完成")


    # 3 设置训练器并添加特殊token
    trainer = trainers.BpeTrainer(
        vocab_size=15000,  # 6400,  # 词汇表大小, 我需要一个更大的词表
        special_tokens=special_tokens,  # 确保这三个token被包含
        show_progress=True,  # 显示训练进度
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()  # 初始字母表
    )
    print(f"debug: 第三步 添加特殊token完成")


    # 4 训练tokenizer
    # 使用tqdm包裹texts，显示训练进度
    texts_with_progress = tqdm(texts, desc="Training Tokenizer", unit=" lines")
    tokenizer.train_from_iterator(texts_with_progress, trainer=trainer)  # 使用训练器训练tokenizer
    print(f"debug: 第四步 训练tokenizer完成"
          )
    # 5 设置解码器
    tokenizer.decoder = decoders.ByteLevel()  # 设置ByteLevel解码器
    print(f"debug: 第五步 设置解码器完成")

    # 6 检查特殊token的索引: 这是常见的约定, 它们位于前三个
    try:
        assert tokenizer.token_to_id("<unk>") == 0  # 检查<unk>的索引
        assert tokenizer.token_to_id("<s>") == 1  # 检查<s>的索引
        assert tokenizer.token_to_id("</s>") == 2  # 检查</s>的索引
    except AssertionError as e:
        print(f"Special tokens not found in the tokenizer. details:\n {e}")

    # 7 保存tokenizer
    tokenizer_dir = "model/minimind_tokenizer"  # 保存目录
    if not os.path.exists(tokenizer_dir):
        os.makedirs(tokenizer_dir, exist_ok=True)  # 创建保存目录

    tokenizer.save("./model/minimind_tokenizer/tokenizer.json")  # 保存tokenizer配置
    tokenizer.model.save(tokenizer_dir)  # 保存tokenizer模型
    print(f"debug: 第七步 保存tokenizer模型完成")

    # 8 手动创建配置文件
    config = {
        "add_bos_token": False,  # 是否添加开始标记
        "add_eos_token": False,  # 是否添加结束标记
        "add_prefix_space": True,  # 是否在每个token前添加空格
        "added_tokens_decoder": {  # 添加的特殊token解码器
            "0": {
                "content": "<unk>",  # 内容
                "lstrip": False,  # 左侧是否去除空格
                "normalized": False,  # 是否归一化
                "rstrip": False,  # 右侧是否去除空格
                "single_word": False,  # 是否单个词
                "special": True  # 是否特殊token
            },
            "1": {
                "content": "<s>",
                "lstrip": False,
                "normalized": False,
                "rstrip": False,
                "single_word": False,
                "special": True
            },
            "2": {
                "content": "</s>",
                "lstrip": False,
                "normalized": False,
                "rstrip": False,
                "single_word": False,
                "special": True
            }
        },
        "additional_special_tokens": [],  # 额外的特殊token
        "bos_token": "<s>",  # 开始标记
        "clean_up_tokenization_spaces": False,  # 是否清理分词后的空格
        "eos_token": "</s>",  # 结束标记
        "legacy": True,  # 是否使用旧版
        "model_max_length": 1000000000000000019884624838656,  # 模型最大长度
        "pad_token": None,  # 填充标记
        "sp_model_kwargs": {},  # SentencePiece模型参数
        "spaces_between_special_tokens": False,  # 特殊token之间是否有空格
        "tokenizer_class": "PreTrainedTokenizerFast",  # 分词器类
        "unk_token": "<unk>",  # 未知标记
        "use_default_system_prompt": False,  # 是否使用默认系统提示
        "chat_template": "{% if messages[0]['role'] == 'system' %}{% set system_message = messages[0]['content'] %}"
                         "{% endif %}{% if system_message is defined %}{{ system_message }}{% endif %}"
                         "{% for message in messages %}{% set content = message['content'] %}{% if message['role'] == 'user' %}"
                         "{{ '<s>user\\n' + content + '</s>\\n<s>assistant\\n' }}{% elif message['role'] == 'assistant' %}"
                         "{{ content + '</s>' + '\\n' }}{% endif %}{% endfor %}"  # 聊天模板
    }

    # 保存配置文件
    with open(os.path.join(tokenizer_dir, "tokenizer_config.json"), "w", encoding="utf-8") as config_file:
        json.dump(config, config_file, ensure_ascii=False, indent=4)  # 保存配置文件

    print("Tokenizer training completed and saved!")  # 打印完成信息


def eval_tokenizer():
    from transformers import AutoTokenizer

    # 加载预训练的tokenizer
    tokenizer = AutoTokenizer.from_pretrained("model/minimind_tokenizer")

    # 定义对话消息
    messages = [
        {"role": "system", "content": "你是一个优秀的聊天机器人，总是给我正确的回应！"},
        {"role": "user", "content": '是椭圆形的'},
        {"role": "assistant", "content": '456'},
        {"role": "user", "content": '456'},
        {"role": "assistant", "content": '789'}
    ]

    # 应用聊天模板
    new_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False  # 不进行分词，直接应用模板
    )

    print(new_prompt)  # 打印应用模板后的提示

    # 获取词汇表大小（不包括特殊符号）
    print('tokenizer词表大小：', tokenizer.vocab_size)

    # 获取实际词汇表长度（包括特殊符号）
    actual_vocab_size = len(tokenizer)
    print('qwen实际词表长度：', actual_vocab_size)

    # 定义新的提示
    new_prompt = 'wenjie，椭圆和⚪的关系是什么呢？因为明天下午要带家人去下医院，所以申请上午在家办公，因为明天下午要带家人去下医院，所以申请上午在家办公，因为明天下午要带家人去下医院，所以申请上午在家办公，下午请半天假~@LWJWe '
    print(new_prompt)  # 打印新的提示

    # 对新的提示进行分词
    model_inputs = tokenizer(new_prompt)

    print(model_inputs)  # 打印分词后的输入
    print('长度：', len(model_inputs['input_ids']))  # 打印输入ID的长度

    # 获取输入ID
    input_ids_ = model_inputs['input_ids']

    # 解码输入ID
    response = tokenizer.decode(input_ids_)
    print(response, end='')  # 打印解码后的文本

def read_head_texts_from_jsonl(file_path, head_text=10):
    """ 从JSONL文件中逐行读取文本数据
    :param file_path: JSONL文件路径
    :yield: 文本数据
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        count = 0
        for line in f:
            data = json.loads(line)
            count += 1
            print(f'第{count}次读取数据...')

            if count < head_text:
                pprint(data)
            else:
                return

def main():
    try:
        # train_tokenizer()  # 注释掉训练函数，如果需要训练则取消注释
        eval_tokenizer()  # 调用评估函数
    except Exception as e:
        msg = f"An error occurred: \n{e}"
        with open('error241022.txt', 'w') as f:
            f.write(msg)

if __name__ == '__main__':
    main()  # 运行主函数
    #
    # tokenizer_dir = "./model/minimind_tokenizer"  # 保存目录
    # dd = os.path.join(tokenizer_dir, "tokenizer.json")
    # cc = "/".join([tokenizer_dir, "tokenizer.json"])
    # print(dd)
    # print(cc)



    # data_path = './dataset/tokenizer_train.jsonl'  # 数据文件路径
    # read_head_texts_from_jsonl(data_path)