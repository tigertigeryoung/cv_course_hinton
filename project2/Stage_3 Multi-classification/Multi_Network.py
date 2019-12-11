import torch.nn as nn
import torch.nn.functional as f


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(3, 3, 3)                     # 输入为 128 * 3 * 500 * 500， 输出为 128 * 3 * 249 * 249
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(3, 6, 3)                     # 输入为 128 * 3 * 249 * 249， 输出为 128 * 6 * 123 * 123
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.relu2 = nn.ReLU(inplace=True)

        self.fc1 = nn.Linear(6 * 123 * 123, 150)            # 输入为 128 * 6 * 123 * 123， 输出为 150
        self.relu3 = nn.ReLU(inplace=True)

        self.drop = nn.Dropout2d()

        self.fc2 = nn.Linear(150, 2)   # 用于动物纲分类，0代表哺乳纲，1代表鸟纲
        self.fc3 = nn.Linear(150, 3)   # 用于动物重分类，0代表兔子，1代表老鼠，2代表鸡

        self.softmax1 = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.relu2(x)

        x = x.view(-1, 6 * 123 * 123)   # flatten
        x = self.fc1(x)
        x = self.relu3(x)

        x = f.dropout(x, training=self.training)

        x_classes = self.fc2(x)
        x_classes = self.softmax1(x_classes)

        x_species = self.fc3(x)
        x_species = self.softmax1(x_species)

        return x_classes, x_species

    # Wout = 向下取整( (Win + 2 * Padding - (Dilation * (Kin - 1) + 1) / Stride + 1 )
