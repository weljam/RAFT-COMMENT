import torch
import torch.nn as nn
import torch.nn.functional as F


class FlowHead(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256):
        super(FlowHead, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, 2, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))

class ConvGRU(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=192+128):
        super(ConvGRU, self).__init__()
        self.convz = nn.Conv2d(hidden_dim+input_dim, hidden_dim, 3, padding=1)
        self.convr = nn.Conv2d(hidden_dim+input_dim, hidden_dim, 3, padding=1)
        self.convq = nn.Conv2d(hidden_dim+input_dim, hidden_dim, 3, padding=1)

    def forward(self, h, x):
        hx = torch.cat([h, x], dim=1)

        z = torch.sigmoid(self.convz(hx))
        r = torch.sigmoid(self.convr(hx))
        q = torch.tanh(self.convq(torch.cat([r*h, x], dim=1)))

        h = (1-z) * h + z * q
        return h

class SepConvGRU(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=192+128):
        super(SepConvGRU, self).__init__()
        # 初始化水平方向的GRU门控卷积层
        self.convz1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))
        self.convr1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))
        self.convq1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))

        # 初始化垂直方向的GRU门控卷积层
        self.convz2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))
        self.convr2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))
        self.convq2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))

    def forward(self, h, x):
        # 水平方向的GRU计算
        hx = torch.cat([h, x], dim=1)  # 拼接隐藏状态和输入
        z = torch.sigmoid(self.convz1(hx))  # 更新门
        r = torch.sigmoid(self.convr1(hx))  # 重置门
        q = torch.tanh(self.convq1(torch.cat([r*h, x], dim=1)))  # 候选隐藏状态
        h = (1-z) * h + z * q  # 更新隐藏状态

        # 垂直方向的GRU计算
        hx = torch.cat([h, x], dim=1)  # 拼接隐藏状态和输入
        z = torch.sigmoid(self.convz2(hx))  # 更新门
        r = torch.sigmoid(self.convr2(hx))  # 重置门
        q = torch.tanh(self.convq2(torch.cat([r*h, x], dim=1)))  # 候选隐藏状态
        h = (1-z) * h + z * q  # 更新隐藏状态

        return h

class SmallMotionEncoder(nn.Module):
    def __init__(self, args):
        super(SmallMotionEncoder, self).__init__()
        cor_planes = args.corr_levels * (2*args.corr_radius + 1)**2
        self.convc1 = nn.Conv2d(cor_planes, 96, 1, padding=0)
        self.convf1 = nn.Conv2d(2, 64, 7, padding=3)
        self.convf2 = nn.Conv2d(64, 32, 3, padding=1)
        self.conv = nn.Conv2d(128, 80, 3, padding=1)

    def forward(self, flow, corr):
        cor = F.relu(self.convc1(corr))
        flo = F.relu(self.convf1(flow))
        flo = F.relu(self.convf2(flo))
        cor_flo = torch.cat([cor, flo], dim=1)
        out = F.relu(self.conv(cor_flo))
        return torch.cat([out, flow], dim=1)

class BasicMotionEncoder(nn.Module):
    def __init__(self, args):
        super(BasicMotionEncoder, self).__init__()
        # 计算相关平面的数量
        cor_planes = args.corr_levels * (2*args.corr_radius + 1)**2
        # 初始化第一个卷积层，输入通道为cor_planes，输出通道为256，卷积核大小为1x1，无填充
        self.convc1 = nn.Conv2d(cor_planes, 256, 1, padding=0)
        # 初始化第二个卷积层，输入通道为256，输出通道为192，卷积核大小为3x3，填充1
        self.convc2 = nn.Conv2d(256, 192, 3, padding=1)
        # 初始化第一个光流卷积层，输入通道为2，输出通道为128，卷积核大小为7x7，填充3
        self.convf1 = nn.Conv2d(2, 128, 7, padding=3)
        # 初始化第二个光流卷积层，输入通道为128，输出通道为64，卷积核大小为3x3，填充1
        self.convf2 = nn.Conv2d(128, 64, 3, padding=1)
        # 初始化最终的卷积层，输入通道为64+192，输出通道为126，卷积核大小为3x3，填充1
        self.conv = nn.Conv2d(64+192, 128-2, 3, padding=1)

    def forward(self, flow, corr):
        # 对相关性特征进行第一次卷积和ReLU激活
        cor = F.relu(self.convc1(corr))
        # 对相关性特征进行第二次卷积和ReLU激活
        cor = F.relu(self.convc2(cor))
        # 对光流特征进行第一次卷积和ReLU激活
        flo = F.relu(self.convf1(flow))
        # 对光流特征进行第二次卷积和ReLU激活
        flo = F.relu(self.convf2(flo))

        # 将相关性特征和光流特征拼接在一起
        cor_flo = torch.cat([cor, flo], dim=1)
        # 对拼接后的特征进行卷积和ReLU激活
        out = F.relu(self.conv(cor_flo))
        # 将最终的特征和原始光流拼接在一起
        return torch.cat([out, flow], dim=1)

class SmallUpdateBlock(nn.Module):
    def __init__(self, args, hidden_dim=96):
        super(SmallUpdateBlock, self).__init__()
        self.encoder = SmallMotionEncoder(args)
        self.gru = ConvGRU(hidden_dim=hidden_dim, input_dim=82+64)
        self.flow_head = FlowHead(hidden_dim, hidden_dim=128)

    def forward(self, net, inp, corr, flow):
        motion_features = self.encoder(flow, corr)
        inp = torch.cat([inp, motion_features], dim=1)
        net = self.gru(net, inp)
        delta_flow = self.flow_head(net)

        return net, None, delta_flow

class BasicUpdateBlock(nn.Module):
    def __init__(self, args, hidden_dim=128, input_dim=128):
        super(BasicUpdateBlock, self).__init__()
        self.args = args
        # 初始化基本运动编码器
        self.encoder = BasicMotionEncoder(args)
        # 初始化分离卷积GRU
        self.gru = SepConvGRU(hidden_dim=hidden_dim, input_dim=128+hidden_dim)
        # 初始化光流头
        self.flow_head = FlowHead(hidden_dim, hidden_dim=256)

        # 初始化掩码卷积层
        self.mask = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),  # 卷积层，输入通道128，输出通道256，卷积核大小3x3，填充1
            nn.ReLU(inplace=True),  # ReLU激活函数
            nn.Conv2d(256, 64*9, 1, padding=0))  # 卷积层，输入通道256，输出通道64*9，卷积核大小1x1，无填充

    def forward(self, net, inp, corr, flow, upsample=True):
        motion_features = self.encoder(flow, corr)
        inp = torch.cat([inp, motion_features], dim=1)

        net = self.gru(net, inp)
        delta_flow = self.flow_head(net)

        # scale mask to balence gradients
        mask = .25 * self.mask(net)
        return net, mask, delta_flow



