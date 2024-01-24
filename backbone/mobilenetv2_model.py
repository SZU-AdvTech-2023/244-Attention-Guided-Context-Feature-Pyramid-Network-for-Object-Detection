from torch import nn
import torch
from torchvision.ops import misc

# 实现了MobileNetV2模型，是一个轻量级的深度学习模型，用于图像分类任务


"""
该函数用于确保通道数是 8 的倍数，以满足模型设计的要求。
函数定义了通道数的调整逻辑，确保通道数不会下降超过原通道数的 10%。
"""


def _make_divisible(ch, divisor=8, min_ch=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_ch is None:
        min_ch = divisor
    new_ch = max(min_ch, int(ch + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_ch < 0.9 * ch:
        new_ch += divisor
    return new_ch


"""
该类定义了一个卷积-批归一化-ReLU激活的序列。
用于构建MobileNetV2中的卷积块。
"""


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, groups=1, norm_layer=None):
        padding = (kernel_size - 1) // 2
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding, groups=groups, bias=False),
            norm_layer(out_channel),
            nn.ReLU6(inplace=True)
        )


"""
该类定义了MobileNetV2的倒残差块（Inverted Residual Block）。
由一系列卷积块组成，包括 1x1 的 pointwise 卷积、3x3 的 depthwise 卷积和另一个 1x1 的 pointwise 卷积。
可以选择是否使用 shortcut 连接。
"""


class InvertedResidual(nn.Module):
    def __init__(self, in_channel, out_channel, stride, expand_ratio, norm_layer=None):
        super(InvertedResidual, self).__init__()
        hidden_channel = in_channel * expand_ratio
        self.use_shortcut = stride == 1 and in_channel == out_channel
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        layers = []
        if expand_ratio != 1:
            # 1x1 pointwise conv
            layers.append(ConvBNReLU(in_channel, hidden_channel, kernel_size=1, norm_layer=norm_layer))
        layers.extend([
            # 3x3 depthwise conv
            ConvBNReLU(hidden_channel, hidden_channel, stride=stride, groups=hidden_channel, norm_layer=norm_layer),
            # 1x1 pointwise conv(linear)
            nn.Conv2d(hidden_channel, out_channel, kernel_size=1, bias=False),
            norm_layer(out_channel),
        ])

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_shortcut:
            return x + self.conv(x)
        else:
            return self.conv(x)


"""
该类是整个MobileNetV2模型的主体。
构建了一系列卷积块和倒残差块，形成了深层的神经网络结构。
包含了一个分类头部，用于图像分类任务。
可以选择加载预训练的权重，或者通过随机初始化权重。
输入通道数为3（RGB图像），输出为模型预测的类别数。
"""


class MobileNetV2(nn.Module):
    def __init__(self, num_classes=1000, alpha=1.0, round_nearest=8, weights_path=None, norm_layer=None):
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = _make_divisible(32 * alpha, round_nearest)
        last_channel = _make_divisible(1280 * alpha, round_nearest)

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        inverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        features = []
        # conv1 layer
        features.append(ConvBNReLU(3, input_channel, stride=2, norm_layer=norm_layer))
        # building inverted residual residual blockes
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * alpha, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t, norm_layer=norm_layer))
                input_channel = output_channel
        # building last several layers
        features.append(ConvBNReLU(input_channel, last_channel, 1, norm_layer=norm_layer))
        # combine feature layers
        self.features = nn.Sequential(*features)

        # building classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(last_channel, num_classes)
        )

        if weights_path is None:
            # weight initialization
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out')
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.zeros_(m.bias)
        else:
            self.load_state_dict(torch.load(weights_path))

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


"""
模型的主要结构包括卷积块、倒残差块以及最后的分类头部。
输入通过一系列的卷积块，其中包括一个深度可分离卷积（depthwise separable convolution）。
倒残差块（Inverted Residual Blocks）被用于增加非线性性和模型容量。
最后通过全局平均池化，将特征图的空间维度降为1，然后通过全连接层输出最终的分类结果。

如果未提供预训练权重路径，模型将使用一种特定的权重初始化方法（例如，卷积层使用 Kaiming 正态初始化，批归一化层使用特定的值初始化）。
"""
