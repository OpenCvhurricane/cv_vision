import numpy as np
from random import shuffle

def svm_loss_naive(weight, X, y, reg):
    """用循环实现的SVM loss计算 这里的loss函数使用的是margin loss
    Inputs: - weight (D, C)： 权重矩阵. - X (N, D)： 批输入 - y (N,) 标签 - reg: 正则参数
    Returns : - loss float - weight的梯度 """
    dweight =np.zeros(weight.shape)
    num_classes = weight.shape[1]
    num_train =X.shape[0]
    loss = 0.0
    for i in range(num_train):
        scores = X[i].dot(weight)
        correct_class_score =scores[y[i]]
        for j in range(num_classes):
            if j ==y[i]:
                continue
            margin = scores[j] - correct_class_score + 1
            if margin >0:
                loss +=margin
                dweight[:,j]+=X[i].T
                dweight[:,y[i]]-=X[i].T
        loss /=num_train
        dweight/=num_train
        loss += reg * np.sum(weight * weight)
        dweight+=2* reg * weight
    return loss, dweight

def svm_loss_vectorized(weight, X, y, reg):
    """不使用循环，利用numpy矩阵运算的特性实现loss和梯度计算51 """
    loss = 0.0
    dweight =np.zeros(weight.shape)
    #计算loss
    num_classes = weight.shape[1]
    num_train =X.shape[0]
    scores=np.dot(X,weight)#得到得分矩阵(N，C)
    correct_socre=scores[range(num_train), list(y)].reshape(-1,1)#得到每个输入的正确分类的分数
    margins=np.maximum(0,scores-correct_socre+1)
    margins[range(num_train), list(y)] =0
    loss=np.sum(margins)/num_train+reg * np.sum(weight * weight)
    #计算梯度
    mask=np.zeros((num_train,num_classes))
    mask[margins>0]=1
    mask[range(num_train),list(y)]-=np.sum(mask,axis=1)
    dweight=np.dot(X.T,mask)
    dweight/=num_train
    dweight+=2* reg * weight
    return loss, dweight
