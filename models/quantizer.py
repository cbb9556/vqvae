import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class VectorQuantizer(nn.Module):
    """
    Discretization bottleneck part of the VQ-VAE.

    Inputs:
    - n_e : number of embeddings
    - e_dim : dimension of embedding
    - beta : commitment cost used in loss term, beta * ||z_e(x)-sg[e]||^2
    """

    def __init__(self, n_e, e_dim, beta):
        super(VectorQuantizer, self).__init__()
        self.n_e = n_e # K = 512
        self.e_dim = e_dim # D = 64
        # 设置beta参数，用于控制量化损失的权重
        self.beta = beta

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        # 将张量中的每个元素按照均匀分布进行随机赋值
        # 取值范围 ([-1./n_embed, 1./n_embed]) 是一种常见的初始化策略，特别是对于嵌入层（embedding layer）。这里的 n_embed 通常表示嵌入向量的维度
        # 平滑梯度：较小的初始权重可以帮助梯度在反向传播过程中更加平滑，避免梯度爆炸或消失。
        # 对称性：负值和正值的对称分布有助于模型在训练初期更好地探索参数空间。
        # 经验法则：这种初始化方法在实践中被证明是有效的，特别是在处理高维嵌入时
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

    def forward(self, z):
        """
        Inputs the output of the encoder network z and maps it to a discrete 
        one-hot vector that is the index of the closest embedding vector e_j

        z (continuous) -> z_q (discrete)

        z.shape = (batch, channel, height, width)

        quantization pipeline:

            1. get encoder input (B,C,H,W)
            2. flatten input to (B*H*W,C)

        """
        # reshape z -> (batch, height, width, channel) and flatten
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.e_dim)

        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - 2 * \
            torch.matmul(z_flattened, self.embedding.weight.t())

        # find closest encodings
        # 寻找每个样本在编码空间中的最小距离索引
        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)
        # 初始化一个全零张量，用于存放最小编码的one-hot表示
        min_encodings = torch.zeros(
            min_encoding_indices.shape[0], self.n_e).to(device)
        # 使用scatter_方法，将最小编码索引处的值设为1，以形成one-hot向量
        min_encodings.scatter_(1, min_encoding_indices, 1)

        # get quantized latent vectors
        z_q = torch.matmul(min_encodings, self.embedding.weight).view(z.shape)

        # compute loss for embedding，固定zq 让 codebook的e靠近 zq
        # 这里，z的参数更新， 也只会与 zq的 codebook的 index e有关系，与解码器没有关系
        loss = torch.mean((z_q.detach()-z)**2) + self.beta * \
            torch.mean((z_q - z.detach()) ** 2) # 固定e，让zq靠近e， 这里会更新 self.embedding.weight的参数，量化层

        # preserve gradients
        # 保持梯度不变。通过这种方式，可以确保在反向传播时，
        # z_q的梯度会传递给z，而不会传递给z_q
        # z是编码器的输出， zq是 量化器的输出解码器的输入
        # 在反向传播时，只有 z 会接收到梯度，而 z_q 不会接收到额外的梯度。
        z_q = z + (z_q - z).detach()

        # perplexity
        # 计算最小编码的平均值，以了解编码的分布情况
        # 计算最小编码的平均值
        # 假设min_encodings是一个形状为(3, 5)的张量，表示3个样本的1-hot编码，每个样本有5个特征：
        # min_encodings = [[1, 0, 0, 0, 0],
        #                  [0, 1, 0, 0, 0],
        #                  [0, 0, 1, 0, 0]]
        # 使用torch.mean沿着第0维度（即样本维度）计算平均值，得到的结果e_mean是一个形状为(5,)的一维张量：
        # e_mean = [1/3, 1/3, 1/3, 0, 0]
        # 这表示在这3个样本中，每个特征的平均激活值分别为1/3, 1/3, 1/3, 0, 0。
        e_mean = torch.mean(min_encodings, dim=0)

        # 计算并返回困惑度（Perplexity）
        #
        # 参数:
        #   e_mean (Tensor): 模型输出的概率分布均值，形状为 [batch_size, vocab_size]
        #
        # 返回:
        #   Tensor: 计算得到的困惑度值
        #
        # 示例:
        #   假设 e_mean 是一个形状为 [1, 5] 的张量，表示一个样本在词汇表中的概率分布
        #   e_mean = torch.tensor([[1/3, 1/3, 1/3, 0, 0]])
        #   计算困惑度时，首先计算每个概率的对数值，然后乘以概率本身，求和后再取负指数
        #   对于这个例子，计算过程如下：
        #   - 对数部分：log(1/3) ≈ -1.0986
        #   - 乘以概率：1/3 * -1.0986 ≈ -0.3662
        #   - 求和：-0.3662 * 3 = -1.0986
        #   - 取负指数：exp(-(-1.0986)) ≈ 3
        #   最终得到的困惑度值为 3，表示模型在这个样本上的不确定性
        perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))

        # reshape back to match original input shape
        z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return loss, z_q, perplexity, min_encodings, min_encoding_indices
