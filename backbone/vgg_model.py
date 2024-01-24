import torch.nn as nn
import torch

# 构建VGG模型的过程，VGG是一种经典的卷积神经网络架构
# 通过调用vgg()函数，可以构建指定VGG模型，如VGG16，然后通过正向传播将输入数据传递到网络中。
# 如果提供了weights_path，模型将加载预训练权重。


# 该类定义了VGG模型的结构。VGG包含一个特征提取部分（features）和一个分类器部分（classifier）。
# 特征提取部分由一系列卷积层和池化层组成，可以通过make_features函数生成。
# 分类器部分由三个全连接层组成，其中包含ReLU激活和Dropout层。
class VGG(nn.Module):
    def __init__(self, features, class_num=1000, init_weights=False, weights_path=None):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, class_num)
        )
        if init_weights and weights_path is None:
            self._initialize_weights()

        if weights_path is not None:
            self.load_state_dict(torch.load(weights_path))

    def forward(self, x):
        # N x 3 x 224 x 224
        x = self.features(x)
        # N x 512 x 7 x 7
        x = torch.flatten(x, start_dim=1)
        # N x 512*7*7
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                # nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


# 该函数根据给定的配置（cfg）创建VGG的特征提取部分。
# cfg 是一个描述VGG卷积层和池化层的列表，其中 'M' 表示最大池化层，其他值表示卷积层的输出通道数。
# 返回一个包含卷积层和ReLU激活函数的Sequential模块。
def make_features(cfg: list):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            layers += [conv2d, nn.ReLU(True)]
            in_channels = v
    return nn.Sequential(*layers)


# 该字典定义了不同VGG模型（如VGG11、VGG13、VGG16、VGG19）的配置。
cfgs = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


# 该函数根据给定的VGG模型名称构建VGG模型。
# 可以选择性地提供预训练权重的路径（weights_path），以加载预训练权重。
def vgg(model_name="vgg16", weights_path=None):
    assert model_name in cfgs, "Warning: model number {} not in cfgs dict!".format(model_name)
    cfg = cfgs[model_name]

    model = VGG(make_features(cfg), weights_path=weights_path)
    return model
