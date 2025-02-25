import torch
import torch.nn.functional as F
from utils.utils import bilinear_sampler, coords_grid

try:
    import alt_cuda_corr
except:
    # alt_cuda_corr is not compiled
    pass


class CorrBlock:
    def __init__(self, fmap1, fmap2, num_levels=4, radius=4):
        self.num_levels = num_levels
        self.radius = radius
        self.corr_pyramid = []

        # all pairs correlation
        corr = CorrBlock.corr(fmap1, fmap2)

        batch, h1, w1, dim, h2, w2 = corr.shape
        corr = corr.reshape(batch*h1*w1, dim, h2, w2)
        
        self.corr_pyramid.append(corr)
        for i in range(self.num_levels-1):
            corr = F.avg_pool2d(corr, 2, stride=2)
            self.corr_pyramid.append(corr)

    def __call__(self, coords):
        r = self.radius
        coords = coords.permute(0, 2, 3, 1)
        batch, h1, w1, _ = coords.shape

        out_pyramid = []
        for i in range(self.num_levels):
            corr = self.corr_pyramid[i]
            dx = torch.linspace(-r, r, 2*r+1, device=coords.device)
            dy = torch.linspace(-r, r, 2*r+1, device=coords.device)
            delta = torch.stack(torch.meshgrid(dy, dx), axis=-1)

            centroid_lvl = coords.reshape(batch*h1*w1, 1, 1, 2) / 2**i
            delta_lvl = delta.view(1, 2*r+1, 2*r+1, 2)
            coords_lvl = centroid_lvl + delta_lvl

            corr = bilinear_sampler(corr, coords_lvl)
            corr = corr.view(batch, h1, w1, -1)
            out_pyramid.append(corr)

        out = torch.cat(out_pyramid, dim=-1)
        return out.permute(0, 3, 1, 2).contiguous().float()

    @staticmethod
    def corr(fmap1, fmap2):
        """
        计算两个特征图之间的相关性。
        参数:
        fmap1 (torch.Tensor): 第一个特征图，形状为 (batch, dim, ht, wd)。
        fmap2 (torch.Tensor): 第二个特征图，形状为 (batch, dim, ht, wd)。
        返回:
        torch.Tensor: 相关性矩阵，形状为 (batch, ht, wd, 1, ht, wd)。
        该函数首先将两个特征图展平为 (batch, dim, ht*wd) 的形状，
        然后计算它们的转置矩阵乘积，得到的结果再重新调整形状为
        (batch, ht, wd, 1, ht, wd)。最后，相关性矩阵除以特征维度的平方根。
        """
        batch, dim, ht, wd = fmap1.shape
        fmap1 = fmap1.view(batch, dim, ht*wd)
        fmap2 = fmap2.view(batch, dim, ht*wd) 
        
        corr = torch.matmul(fmap1.transpose(1,2), fmap2)
        corr = corr.view(batch, ht, wd, 1, ht, wd)
        return corr  / torch.sqrt(torch.tensor(dim).float())


class AlternateCorrBlock:
    def __init__(self, fmap1, fmap2, num_levels=4, radius=4):
        """
        初始化 AlternateCorrBlock 类。
        
        参数:
        fmap1 (torch.Tensor): 第一个特征图，形状为 (batch, dim, ht, wd)。
        fmap2 (torch.Tensor): 第二个特征图，形状为 (batch, dim, ht, wd)。
        num_levels (int): 金字塔的层数，默认为 4。
        radius (int): 相关性计算的半径，默认为 4。
        """
        self.num_levels = num_levels
        self.radius = radius

        # 构建特征图金字塔(进行多次平均池化构建多层金字塔)
        self.pyramid = [(fmap1, fmap2)]
        for i in range(1, self.num_levels):
            # 对特征图进行平均池化，以构建金字塔
            fmap1 = F.avg_pool2d(self.pyramid[-1][0], 2, stride=2)
            fmap2 = F.avg_pool2d(self.pyramid[-1][1], 2, stride=2)
            self.pyramid.append((fmap1, fmap2))

    def __call__(self, coords):
        """
        计算给定坐标的相关性。
        
        参数:
        coords (torch.Tensor): 坐标张量，形状为 (batch, 2, ht, wd)。
        图像数量, 2个坐标, 高, 宽
        返回:
        torch.Tensor: 相关性矩阵，形状为 (batch, num_levels*radius*radius, ht, wd)。
        """
        # 将坐标张量的维度从 (batch, 2, ht, wd) 转换为 (batch, ht, wd, 2)
        coords = coords.permute(0, 2, 3, 1)
        B, H, W, _ = coords.shape
        dim = self.pyramid[0][0].shape[1]

        corr_list = []
        for i in range(self.num_levels):
            r = self.radius
            # 获取第 i 层金字塔的特征图，并调整维度为 (batch, ht, wd, dim)
            fmap1_i = self.pyramid[0][0].permute(0, 2, 3, 1).contiguous()
            fmap2_i = self.pyramid[i][1].permute(0, 2, 3, 1).contiguous()

            # 将坐标张量缩放到第 i 层金字塔的尺度，并调整维度为 (batch, 1, ht, wd, 2)
            coords_i = (coords / 2**i).reshape(B, 1, H, W, 2).contiguous()
            
            # 使用 alt_cuda_corr 计算相关性
            corr, = alt_cuda_corr.forward(fmap1_i, fmap2_i, coords_i, r)
            # 将相关性张量的第 1 维去掉，并添加到相关性列表中
            corr_list.append(corr.squeeze(1))

        # 将所有层的相关性张量堆叠起来，形成形状为 (batch, num_levels, ht, wd) 的张量
        corr = torch.stack(corr_list, dim=1)
        # 调整相关性张量的形状为 (batch, num_levels*radius*radius, ht, wd)
        corr = corr.reshape(B, -1, H, W)
        # 返回归一化后的相关性张量
        return corr / torch.sqrt(torch.tensor(dim).float())
