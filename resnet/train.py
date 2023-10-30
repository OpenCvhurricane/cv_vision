import torch.nn as nn
import torch
import torchvision
import torch.optim as optim
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BasicBlock(nn.Module):  #卷积两层，F（X）和X的维度相等
    #expansion是F（X）相对X维度拓展的倍数
    expansion=1   ## 残差映射F(X)的维度有没有发生变化，1表示没有变化，downsample=None

    #downsample是用来将残差数据和卷积数据的shape变得相同，可以直接进行相加操作
    def __init__(self,in_channel,out_channel,stride=1,downsample=None,**kwargs):
        super(BasicBlock,self).__init__()
        self.conv1=nn.Conv2d(in_channels=in_channel,out_channels=out_channel,kernel_size=3,stride=stride,padding=1,bias=False)
        self.bn1=nn.BatchNorm2d(out_channel)   # BN层在conv和relu层之间

        self.conv2=nn.Conv2d(in_channels=out_channel,out_channels=out_channel,kernel_size=3,stride=1,padding=1,bias=False)
        self.bn2=nn.BatchNorm2d(out_channel)

        self.relu=nn.ReLU(inplace=True)   #这里inplace=True 的意思是原地池化操作。就是在原来的内存地址池化，覆盖掉以前的数据。好处就是可以节省运算内存，不用多储存变量
        self.downsample=downsample

    def forward(self, x):
        identity=x
        if self.downsample is not None:
            identity=self.downsample(x)

        out=self.conv1(x)
        out=self.bn1(out)
        out=self.relu(out)

        out=self.conv2(out)
        out=self.bn2(out)
        # out=F(x)+x
        out+=identity
        out=self.relu(out)

        return out

class Bottleneck(nn.Module): # 卷积3层，F(X)和X的维度不等
    """
    注意：原论文中，在虚线残差结构的主分支上，第一个1x1卷积层的步距是2，第二个3x3卷积层步距是1。
    但在pytorch官方实现过程中是第一个1x1卷积层的步距是1，第二个3x3卷积层步距是2，
    这么做的好处是能够在top1上提升大概0.5%的准确率。
    """
    # expansion是F(X)相对X维度拓展的倍数
    expansion=4

    def __init__(self,in_channel,out_channel,stride=1,downsample=None):
        super(Bottleneck,self).__init__()

        self.conv1=nn.Conv2d(in_channels=in_channel,out_channels=out_channel,kernel_size=1,stride=1,bias=False)  #1*1降维
        self.bn1=nn.BatchNorm2d(out_channel)

        self.conv2=nn.Conv2d(in_channels=out_channel,out_channels=out_channel,kernel_size=3,stride=1,bias=False) #3*3特征提取
        self.bn2=nn.BatchNorm2d(out_channel)

        self.conv3=nn.Conv2d(in_channels=out_channel,out_channels=out_channel*self.expansion,kernel_size=1,stride=1,bias=False) #1*1恢复维度
        self.bn3=nn.BatchNorm2d(out_channel*self.expansion)

        self.relu=nn.ReLU(inplace=True)
        self.downsample=downsample

    def forward(self, x):
        identity=x
        # downsample是用来将残差数据和卷积数据的shape变的相同，可以直接进行相加操作。
        if self.downsample is not None:
            identity=self.downsample(x)

        out=self.conv1(x)
        out=self.bn1(out)
        out=self.relu(out)

        out=self.conv2(out)
        out=self.bn2(out)
        out=self.relu(out)

        out=self.conv3(out)
        out=self.bn3(out)

        # out=F(X)+X
        out+=identity
        out=self.relu(out)

        return out
class ResNet(nn.Module):
    def __init__(self,
                 block,  # 使用的残差块类型
                 blocks_num,  # 每个卷积层，使用残差块的个数
                 num_classes=10,  # 训练集标签的分类个数
                 include_top=True,  # 是否在残差结构后接上pooling、fc、softmax
                 ):

        super(ResNet,self).__init__()
        self.include_top=include_top
        self.in_channel=64   # 第一层卷积输出特征矩阵的深度，也是后面层输入特征矩阵的深度

        # 输入层有RGB三个分量，使得输入特征矩阵的深度是3
        self.conv1=nn.Conv2d(3,self.in_channel,kernel_size=7,stride=2,padding=3,bias=False)
        self.bn1=nn.BatchNorm2d(self.in_channel)
        self.relu=nn.ReLU(inplace=True)

        self.maxpool=nn.MaxPool2d(kernel_size=3,stride=2,padding=1)

        # _make_layer(残差块类型，残差块中第一个卷积层的卷积核个数，残差块个数，残差块中卷积步长)函数：生成多个连续的残差块的残差结构
        self.layer1=self._make_layer(block,64,blocks_num[0])
        self.layer2=self._make_layer(block,128,blocks_num[1],stride=2)
        self.layer3=self._make_layer(block,256,blocks_num[2],stride=2)
        self.layer4=self._make_layer(block,512,blocks_num[3],stride=2)

        if self.include_top:  # 默认为True，接上pooling、fc、softmax
            self.avgpool=nn.AdaptiveAvgPool2d((1,1)) # 自适应平均池化下采样，无论输入矩阵的shape为多少，output size均为的高宽均为1x1
            # 使矩阵展平为向量，如（W,H,C）->(1,1,W*H*C)，深度为W*H*C
            self.fc=nn.Linear(512*block.expansion,num_classes)  # 全连接层，512 * block.expansion为输入深度，num_classes为分类类别个数

        for m in self.modules():#初始化
            if isinstance(m,nn.Conv2d): #isinstance() 函数来判断一个对象(m)是否是一个已知的类型(如int等，这儿为nn.Conv2d)
                nn.init.kaiming_normal_(m.weight,mode='fan_out',nonlinearity='relu')  #相应的初始化方式

    # _make_layer()函数：生成多个连续的残差块，(残差块类型，残差块中第一个卷积层的卷积核个数，残差块个数，残差块中卷积步长)
    def _make_layer(self,block,channel,block_num,stride=1):
        downsample=None

        #寻找：卷积步长不为1或深度扩张有变化，导致F(X)与X的shape不同的残差块，就要对X定义下采样函数，使之shape相同
        if stride!=1 or self.in_channel!=channel*block.expansion:
            downsample=nn.Sequential(
                nn.Conv2d(self.in_channel,channel**block.expansion,kernel_size=1,stride=stride,bias=False),
                nn.BatchNorm2d(channel*block.expansion)
            )

        # layers用于顺序储存各连续残差块
        # 每个残差结构，第一个残差块均为需要对X下采样的残差块，后面的残差块不需要对X下采样
        layers=[]
        # 添加第一个残差块，第一个残差块均为需要对X下采样的残差块
        layers.append(block(self.in_channel,
                            channel,
                            downsample=downsample,
                            stride=stride
                            ))
        self.in_channel=channel*block.expansion
        # 后面的残差块不需要对X下采样
        for _ in range(1,block_num):
            layers.append(block(self.in_channel,
                                channel))
        # 以非关键字参数形式，将layers列表，传入Sequential(),使其中残差块串联为一个残差结构
        return nn.Sequential(*layers)

    def forward(self, x):
        x=self.conv1(x)
        x=self.bn1(x)
        x=self.relu(x)

        x=self.maxpool(x)

        x=self.layer1(x)
        x=self.layer2(x)
        x=self.layer3(x)
        x=self.layer4(x)

        if self.include_top:  #一般为True
            x=self.avgpool(x)
            x=torch.flatten(x,1)
            x=self.fc(x)

        return x
# 至此ResNet的基本框架就写好了

# 下面定义不同层的ResNet

def ResNet18(num_classes=10,include_top=True):
    return ResNet(BasicBlock,[2,2,2,2],num_classes=num_classes,include_top=include_top)

# 设置超参数
num_epochs = 10
batch_size = 128
learning_rate = 0.001

# 数据预处理
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

# 加载数据集
print("正在加载数据集")
train_dataset = CIFAR10(root='./dataset', train=True, download=True, transform=transform_train)
test_dataset = CIFAR10(root='./dataset', train=False, download=True, transform=transform_test)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
print("数据集加载成功")

# 创建实例
model=ResNet18().to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 训练模型
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, labels)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))

# 在测试集上评估模型
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Accuracy of the model on the test images: {} %'.format(100 * correct / total))

# 保存模型
torch.save(model, './models/ResNet18.ckpt')
print("模型保存成功")
