import cv2
import numpy as np
from torch.utils.data import Dataset


class BlockDataset(Dataset):
    """
    Creates block dataset of 32X32 images with 3 channels
    requires numpy and cv2 to work
    """

    def __init__(self, file_path, train=True, transform=None):
        print('Loading block data')
        data = np.load(file_path, allow_pickle=True)
        print('Done loading block data')
        data = np.array([cv2.resize(x[0][0][:, :, :3], dsize=(
            32, 32), interpolation=cv2.INTER_CUBIC) for x in data])

        n = data.shape[0]
        cutoff = n//10
        self.data = data[:-cutoff] if train else data[-cutoff:]
        self.transform = transform

    def __getitem__(self, index):
        img = self.data[index]
        if self.transform is not None:
            img = self.transform(img)
        label = 0
        return img, label

    def __len__(self):
        return len(self.data)


class LatentBlockDataset(Dataset):
    """
    加载潜在块数据集
    """

    def __init__(self, file_path, train=True, transform=None):
        """
        初始化潜在块数据集

        参数:
        - file_path: 潜在数据文件的路径
        - train:     布尔值，表示是否加载训练数据或测试数据
        - transform: 可选的数据转换操作
        """
        # 加载潜在块数据
        print('加载潜在块数据')
        data = np.load(file_path, allow_pickle=True)
        print('完成加载潜在块数据')

        # 将数据划分为训练集和测试集
        self.data = data[:-500] if train else data[-500:]
        self.transform = transform

    def __getitem__(self, index):
        """
        获取数据集中的一项

        参数:
        - index: 要获取的项的索引

        返回:
        - img:   处理后的数据项
        - label: 数据项的标签，固定为0
        """
        # 从数据中获取单个项
        img = self.data[index]
        # 如果存在转换操作，则应用转换
        if self.transform is not None:
            img = self.transform(img)
        # 分配固定的标签
        label = 0
        return img, label

    def __len__(self):
        """
        获取数据集中的总项数
        """
        return len(self.data)
