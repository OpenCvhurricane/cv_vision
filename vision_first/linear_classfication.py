import numpy as np
from linear_svm import *
from softmax import *

class LinearClassifier(object):#线性分类器的基类
    def __init__(self):
        self.weight =None
    def train(self, X, y, learning_rate=1e-3, reg=1e-5, num_iters=100, batch_size=200, verbose=False):
        """使用SGD优化参数矩阵
        Inputs: - X (N, D) - y (N,) - learning_rate: 学习率. - reg: 正则参数. - num_iters: (int) 训练迭代的次数
         - batch_size: (int) 每次迭代使用的样本数量. - verbose: (boolean) 是否显示训练进度
        Outputs: 返回一个list保存了每次迭代的loss """
        num_train, dim =X.shape
        num_classes = np.max(y) + 1 #假设有k类，y的取值为【0，k-1】且最大的下标一定会在训练数据中出现
        #初始化权重矩阵
        if self.weight is None:
            self.weight = 0.001 *np.random.randn(dim, num_classes)
        loss_history =[]
        for it in range(num_iters): #在每次迭代，随机选择batch_size个数据
            mask=np.random.choice(num_train,batch_size,replace=True)
            X_batch =X[mask]
            y_batch =y[mask]
            #计算损失和梯度
            loss, grad =self.loss(X_batch, y_batch, reg)
            loss_history.append(loss)
            #更新参数
            self.weight -= grad* learning_rate
            if verbose and it % 100 ==0:
                print('iteration %d / %d: loss %f' %(it, num_iters, loss))

        return loss_history
    def predict(self, X):
        """ 使用训练好的参数来对输入进行预测60
        Inputs: - X (N, D) Returns: - y_pred (N，):预测的正确分类的下标 """
        y_pred=np.dot(X,self.weight)
        y_pred = np.argmax(y_pred, axis = 1)
        return y_pred
    def loss(self, X_batch, y_batch, reg):
        """这只是一个线性分类器的基类 不同的线性分类器loss的计算方式不同 所以需要在子类中重写 """
        pass

class LinearSVM(LinearClassifier):
    """使用SVM loss"""
    def loss(self, X_batch, y_batch, reg):
        return svm_loss_vectorized(self.weight, X_batch, y_batch, reg)
class Softmax(LinearClassifier):
    """使用交叉熵"""
    def loss(self, X_batch, y_batch, reg):
        return softmax_loss_vectorized(self.weight, X_batch, y_batch, reg)
