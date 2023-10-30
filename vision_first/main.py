import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import torch

### 读入图像
img = cv.imread('C:/Users/Dell/Desktop/cv/horse.jpg')
### 对图像进行展示
plt.imshow(img)
plt.show()
### cv默认对图像的读取方式为BGR
print(type(img))
print(img.shape)
### 将格式转换为RGB
img_convert = cv.cvtColor(img, cv.COLOR_BGR2RGB)
plt.imshow(img_convert)

### 图像的RGB矩阵表示
image_array=np.array(img)
print(image_array)
### 转换为列向量表示
image = image_array.flatten()
print (image)

kind = 10
rate = 1e-3
weight = None

### L2正则化
def L2_Regularization () :
    return rate * np.sum(weight ** 2)

### 采用多类支撑向量机损失
def loss_function (outputs , labels) :
    num_labels = len(labels)
    corrects = outputs[range(num_labels), labels].unsqueeze(0).T
    # 最大间隔
    margin = 1.0
    margins = outputs - corrects + margin
    loss = torch.sum(torch.max(margins, 1)[0]) / len(labels)
    #正则化
    loss += rate * L2_Regularization()
    return loss

def train(self, X, y, learning_rate=1e-3, reg=1e-5, num_iters=100, batch_size=200, verbose=False):
     num_train, dim = X.shape31
     num_classes = np.max(y) + 1  # 假设有k类，y的取值为【0，k-1】且最大的下标一定会在训练数据中出现
     # 初始化权重矩阵
     if weight is None:
     weight = 0.001 * np.random.randn(dim, num_classes)
     loss_history = []
     # 在每次迭代，随机选择batch_size个数据
     for it in range(num_iters):
        mask = np.random.choice(num_train, batch_size, replace=True)
        X_batch = X[mask]
        y_batch = y[mask]
        # 计算损失和梯度
        loss, grad = self.loss(X_batch, y_batch, reg)
        loss_history.append(loss)
        # 更新参数
        weight -= grad * learning_rate51
        if verbose and it % 100 == 0:
            print('iteration %d / %d: loss %f' % (it, num_iters, loss))

     return loss_history










