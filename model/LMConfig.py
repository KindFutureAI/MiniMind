from transformers import PretrainedConfig
from typing import List


class LMConfig(PretrainedConfig):
    model_type = "minimind"

    def __init__(
            self,
            dim: int = 768,                 # 模型维度，默认为 768
            n_layers: int = 3,              # Transformer 层数，默认为 3
            n_heads: int = 24,              # 注意力头数，默认为 24
            n_kv_heads: int = 8,            # KV 头数，默认为 8
            vocab_size: int = 15000,         # 词汇表大小，默认为 15000
            hidden_dim: int = None,         # 隐藏层维度，默认为 None
            multiple_of: int = 64,          # 隐藏层维度的倍数，默认为 64
            norm_eps: float = 1e-5,         # 归一化层的 epsilon 值，默认为 1e-5
            max_seq_len: int = 512,         # 最大序列长度，默认为 512
            dropout: float = 0.0,           # Dropout 概率，默认为 0.0
            flash_attn: bool = True,        # 是否使用 Flash Attention，默认为 True
            ####################################################
            # Here are the specific configurations of MOE
            # When use_moe is false, the following is invalid
            ####################################################
            use_moe: bool = False,          # 是否使用混合专家模型（MOE），默认为 False
            num_experts_per_tok: int = 2,   # 每个 token 选择的专家数量，默认为 2
            n_routed_experts: int = 4,      # 总的专家数量，默认为 4
            n_shared_experts: bool = True,  # 是否共享专家，默认为 True
            scoring_func: str = 'softmax',  # 评分函数，默认为 'softmax'
            aux_loss_alpha: float = 0.01,   # 辅助损失的 alpha 参数，默认为 0.01
            seq_aux: bool = True,           # 是否在序列级别上计算辅助损失，默认为 True
            norm_topk_prob: bool = True,    # 是否标准化 top-k 概率，默认为 True
            **kwargs,  # 其他关键字参数
    ):
        """
        初始化 LMConfig 类。

        :param dim: 模型维度，默认为 768。
        :param n_layers: Transformer 层数，默认为 8。
        :param n_heads: 注意力头数，默认为 16。
        :param n_kv_heads: KV 头数，默认为 8。
        :param vocab_size: 词汇表大小，默认为 6400。
        :param hidden_dim: 隐藏层维度，默认为 None。
        :param multiple_of: 隐藏层维度的倍数，默认为 64。
        :param norm_eps: 归一化层的 epsilon 值，默认为 1e-5。
        :param max_seq_len: 最大序列长度，默认为 512。
        :param dropout: Dropout 概率，默认为 0.0。
        :param flash_attn: 是否使用 Flash Attention，默认为 True。
        :param use_moe: 是否使用 Mixture of Experts (MoE)，默认为 False。
        :param num_experts_per_tok: 每个 token 选择的专家数量，默认为 2。
        :param n_routed_experts: 总的专家数量，默认为 4。
        :param n_shared_experts: 是否共享专家，默认为 True。
        :param scoring_func: 评分函数，默认为 'softmax'。
        :param aux_loss_alpha: 辅助损失的 alpha 参数，默认为 0.01。
        :param seq_aux: 是否在序列级别上计算辅助损失，默认为 True。
        :param norm_topk_prob: 是否标准化 top-k 概率，默认为 True。
        :param kwargs: 其他关键字参数，传递给父类 `PretrainedConfig`。
        """
        self.dim = dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.multiple_of = multiple_of
        self.norm_eps = norm_eps
        self.max_seq_len = max_seq_len
        self.dropout = dropout
        self.flash_attn = flash_attn
        ####################################################
        # Here are the specific configurations of MOE
        # When use_moe is false, the following is invalid
        ####################################################
        self.use_moe = use_moe
        self.num_experts_per_tok = num_experts_per_tok
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.scoring_func = scoring_func
        self.aux_loss_alpha = aux_loss_alpha
        self.seq_aux = seq_aux
        self.norm_topk_prob = norm_topk_prob
        super().__init__(**kwargs)
