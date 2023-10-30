import numpy as np
from random import shuffle

def softmax_loss_naive(weight , X , y, reg):
    """用循环实现softmax损失函数 D,C,N分别表示数据维度，标签种类个数和数据批大小
     Inputs: - weight (D, C)：weights. - X (N, D)：data. - y (N,)： labels - reg: (float) regularization strength
    Returns : - loss - gradient """
    loss = 0.0
    dweight = np.zeros_like(weight)
    num_classes = weight.shape[1]
    num_train = X.shape[0]
    for i in range(num_train):
        scores = np.dot(X[i],weight)
        shift_scores = scores-max(scores)
        dom = np.log(np.sum(np.exp(shift_scores)))
        loss_i = -shift_scores[y[i]]+dom
        loss += loss_i
        for j in range(num_classes):
            softmax_output = np.exp(shift_scores[j])/sum(np.exp(shift_scores))
            if j == y[i]:
                dweight[:, j] += (-1 + softmax_output) * X[i].T
            else:
                dweight[:,j] += softmax_output * X[i].T
                loss /= num_train
                loss += reg * np.sum(weight * weight)
                dweight = dweight/num_train + 2*reg*weight

    return loss, dweight

def softmax_loss_vectorized(weight, X, y, reg):
    """无循环的实现"""
    loss = 0.0
    dweight = np.zeros_like(weight)
    num_classes = weight.shape[1]
    num_train = X.shape[0]
    scores = np.dot(X,weight)
    shift_scores = scores-np.max(scores,axis=1).reshape(-1,1)
    softmax_output = np.exp(shift_scores)/np.sum(np.exp(shift_scores), axis = 1).reshape(-1,1)
    loss=np.sum(-np.log(softmax_output[range(num_train),y]))
    loss=loss/num_train+reg * np.sum(weight * weight)
    dweight = softmax_output.copy()
    dweight[range(num_train), y] -= 1
    dweight=np.dot(X.T,dweight)
    dweight = dweight/num_train + 2*reg*weight

    return loss, dweight
