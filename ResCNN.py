import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.cuda
import time


transform = transforms.Compose([
    transforms.Resize(40),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32),
    transforms.ToTensor()]
)
#parameters
batchSize = 100
learningRate = 0.05
epochs = 100
period = 30 #every period to change learning rate
changeScala = 3 #learning rate divide with changeScala every time
#load dataset
trainDataset = dsets.CIFAR10(
    root='./data',
    train=True,
    transform=transform,
    download=True
)
trainLoader = torch.utils.data.DataLoader(
    dataset=trainDataset,
    batch_size=batchSize,
    shuffle=True
)
testDataset = dsets.CIFAR10(
    root='./data',
    train=False,
    transform=transform
)
testLoader = torch.utils.data.DataLoader(
    dataset=testDataset,
    batch_size=batchSize,
    shuffle=False
)

print(testDataset.test_labels)

def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False
    )
#define residual block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

#define residual net model
class ResNet(nn.Module):
    def __init__(self, block, layers, numClasses=10):
        super(ResNet, self).__init__()
        self.in_channels = 16
        self.conv = conv3x3(3,16)
        self.bn = nn.BatchNorm2d(16)
        self.relu = nn.LeakyReLU(inplace=True)
        self.layer1 = self.makeLayer(block, 16, layers[0])
        self.layer2 = self.makeLayer(block, 32, layers[0], 2)
        self.layer3 = self.makeLayer(block, 64, layers[1], 2)
        self.avgPool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64, numClasses)

    def makeLayer(self, block, out_channels, blocks, stride=1):
        downsample = None

        if (stride != 1) or (self.in_channels != out_channels):
            downsample = nn.Sequential(
                conv3x3(self.in_channels, out_channels, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avgPool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

resnet = ResNet(ResidualBlock, [3, 3, 3]).cuda()

Criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(resnet.parameters(), lr=learningRate)


#start clock
startTime = time.time()
#training
for epoch in range(epochs):
    for i, (images, labels) in enumerate(trainLoader):
        images = Variable(images).cuda()
        labels = Variable(labels).cuda()
        outputs = resnet(images)

        optimizer.zero_grad()
        loss = Criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        if (i+1) % batchSize == 0:
            print('epoch [%3d/%3d]  |  iter [%4d/%4d]  |  loss:--->%.4f'%(epoch+1, epochs, i+1, len(trainDataset)// batchSize, loss.data.item()))
            # print(loss)
    if epoch % period == 0:
        learningRate /= changeScala
        optimizer = torch.optim.Adam(resnet.parameters(), lr=learningRate)

#stop clock
stopTime = time.time()
timeConsumption = stopTime - startTime
print('GPU consumption time --->%.2f seconds'%timeConsumption)
correct = 0
total = 0

for images, labels in testLoader:
    images = Variable(images).cuda()
    outputs = resnet(images)
    # print('outputs.data')
    # print(outputs.data)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct +=(predicted == labels.cuda()).sum()
accuracy = 100 * correct / total
print('accuracy:---> %.2f%%'%accuracy)


