#实现hinge_loss和sotfmax_loss
import random
import numpy as np
from data_utils import load_CIFAR10
import matplotlib.pyplot as plt
from linear_svm import svm_loss_naive,svm_loss_vectorized
from softmax import softmax_loss_naive,softmax_loss_vectorized
import time
from linear_classfication import LinearSVM, Softmax

plt.rcParams['figure.figsize'] = (10.0, 8.0) #set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

###################################第一部分 载入数据并处理###############################
#载入CIFAR10数据.
cifar10_dir = 'C:/Users/Dell/Desktop/cv/cifar-10-python/cifar-10-batches-py'
X_train, y_train, X_test, y_test =load_CIFAR10(cifar10_dir)
print('Training data shape:', X_train.shape)
print('Training labels shape:', y_train.shape)
print('Test data shape:', X_test.shape)
print('Test labels shape:', y_test.shape)
#每个分类选几个图片显示观察一下
classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
num_classes =len(classes)
samples_per_class = 7
for y, cls in enumerate(classes):
     idxs = np.flatnonzero(y_train ==y)
     idxs = np.random.choice(idxs, samples_per_class, replace=False)
     for i, idx in enumerate(idxs):
         plt_idx = i * num_classes + y + 1
         plt.subplot(samples_per_class, num_classes, plt_idx)
         plt.imshow(X_train[idx].astype('uint8'))
         plt.axis('off')
         if i ==0:
            plt.title(cls)
            plt.show()
#把数据分为训练集，验证集和测试集。
#用一个小子集做测验，运行更快。
num_training = 49000
num_validation = 1000
num_test = 1000
num_dev = 500
#数据集本身没有给验证集，需要自己把训练集分成两部分
mask = range(num_training, num_training +num_validation)
X_val =X_train[mask]
y_val =y_train[mask]
mask =range(num_training)
X_train =X_train[mask]
y_train =y_train[mask]
mask = np.random.choice(num_training, num_dev, replace=False)
X_dev =X_train[mask]
y_dev =y_train[mask]
mask =range(num_test)
X_test =X_test[mask]
y_test =y_test[mask]
print('Train data shape:', X_train.shape)
print('Train labels shape:', y_train.shape)
print('Validation data shape:', X_val.shape)
print('Validation labels shape:', y_val.shape)
print('Test data shape:', X_test.shape)
print('Test labels shape:', y_test.shape)
X_train = np.reshape(X_train, (X_train.shape[0], -1))
X_val = np.reshape(X_val, (X_val.shape[0], -1))
X_test = np.reshape(X_test, (X_test.shape[0], -1))
X_dev = np.reshape(X_dev, (X_dev.shape[0], -1))
print('Training data shape:', X_train.shape)
print('Validation data shape:', X_val.shape)
print('Test data shape:', X_test.shape)
print('dev data shape:', X_dev.shape)
#预处理: 把像素点数据化成以0为中心
#第一步: 在训练集上计算图片像素点的平均值
mean_image = np.mean(X_train, axis=0)
print(mean_image.shape)
plt.figure(figsize=(4,4))
plt.imshow(mean_image.reshape((32,32,3)).astype('uint8')) #可视化一下平均值
plt.show()
#第二步: 所有数据都减去刚刚得到的均值
X_train -=mean_image
X_val -=mean_image
X_test -=mean_image
X_dev -=mean_image
#第三步: 给所有的图片都加一个位，并设为1，这样在训练权重的时候就不需要b了，只需要w
#相当于把b的训练并入了W中
X_train = np.hstack([X_train, np.ones((X_train.shape[0], 1))])
X_val = np.hstack([X_val, np.ones((X_val.shape[0], 1))])
X_test = np.hstack([X_test, np.ones((X_test.shape[0], 1))])
X_dev = np.hstack([X_dev, np.ones((X_dev.shape[0], 1))])
print(X_train.shape, X_val.shape, X_test.shape, X_dev.shape)

###################################第二部分 定义需要用到的函数###########################
def cmp_naiveANDvectorized(naive,vectorized):
    '''每个损失函数都用两种方式实现：循环和无循环(即利用numpy的特性)130 '''
    W = np.random.randn(3073, 10) * 0.0001
    #对比两张实现方式的计算时间
    tic =time.time()
    loss_naive, grad_naive = naive(W, X_dev, y_dev, 0.000005)
    toc =time.time()
    print('Naive computed in %fs' % ( toc -tic))
    tic =time.time()
    loss_vectorized, grad_vectorized = vectorized(W, X_dev, y_dev, 0.000005)
    toc =time.time()
    print('Vectorized computed in %fs' % ( toc -tic))
    #检验损失的实现是否正确，对于随机初始化的数据的权重，
    #softmax_loss应该约等于-log(0.1),svm_loss应该约等于9
    print('loss %f %f' %(loss_naive , loss_vectorized))
    #对比两种实现方式得到的结果是否相同
    print('difference loss %f' % (loss_naive -loss_vectorized))
    difference = np.linalg.norm(grad_naive - grad_vectorized, ord='fro')
    print('difference gradient: %f' %difference)

def cross_choose(Linear_classifier,learning_rates,regularization_strengths):
    '''选择超参数157 '''
    results = {} #存储每一对超参数对应的训练集和验证集上的正确率
    best_val = -1 #最好的验证集上的正确率
    best_model = None #最好的验证集正确率对应的svm类的对象
    best_loss_hist=None
    for rs in regularization_strengths:
        for lr in learning_rates:
            classifier =Linear_classifier
            loss_hist = classifier.train(X_train, y_train, lr, rs, num_iters=5)
            y_train_pred =classifier.predict(X_train)
            train_accuracy = np.mean(y_train ==y_train_pred)
            y_val_pred =classifier.predict(X_val)
            val_accuracy = np.mean(y_val ==y_val_pred)
            if val_accuracy >best_val:
                print("1")
                best_val =val_accuracy
                best_model =classifier
                best_loss_hist=loss_hist
                results[(lr,rs)] =train_accuracy, val_accuracy
    for lr, reg in sorted(results):
        train_accuracy, val_accuracy =results[(lr, reg)]
        print ('lr %e reg %e train accuracy: %f val accuracy: %f' %( lr, reg, train_accuracy, val_accuracy))
        print('best validation accuracy achieved during cross-validation: %f' %best_val)
    #可视化loss曲线
    plt.plot(best_loss_hist)
    plt.xlabel('Iteration number')
    plt.ylabel('Loss value')
    plt.show()
    return results, best_val, best_model

def show_weight(best_model):#看看最好的模型的效果
    #可视化学到的权重
    y_test_pred = best_model.predict(X_test)
    test_accuracy = np.mean(y_test ==y_test_pred)
    print('final test set accuracy: %f' %test_accuracy)
    w = best_model.weight[:-1,:] #去掉偏置参数
    w = w.reshape(32, 32, 3, 10)
    w_min, w_max =np.min(w), np.max(w)
    classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    for i in range(10):
        plt.subplot(2, 5, i + 1)
        #把权重转换到0-255
        wimg = 255.0 * (w[:, :, :, i].squeeze() - w_min) / (w_max -w_min)
        plt.imshow(wimg.astype('uint8'))
        plt.axis('off')
        plt.title(classes[i])
        plt.show()

##########################第三部分 应用和比较svm_loss和softmax_loss######################
cmp_naiveANDvectorized(svm_loss_naive,svm_loss_vectorized)
learning_rates = [(1+i*0.1)*1e-7 for i in range(-3,5)]
regularization_strengths = [(5+i*0.1)*1e3 for i in range(-3,3)]
#正则参数的选择要根据正常损失和W*W的大小的数量级来确定，初始时正常loss大概是9，W*W大概是1e-6
#可以观察最后loss的值的大小来继续调整正则参数的大小，使正常损失和正则损失保持合适的比例
results,best_val,best_model=cross_choose(LinearSVM(),learning_rates,regularization_strengths)
show_weight(best_model)
print("--------------------------------------------------------")
cmp_naiveANDvectorized(softmax_loss_naive,softmax_loss_vectorized)
learning_rates = [(2+i*0.1)*1e-7 for i in range(-2,2)]
regularization_strengths = [(7+i*0.1)*1e3 for i in range(-3,3)]
results,best_val,best_model=cross_choose(Softmax(),learning_rates,regularization_strengths)
show_weight(best_model)
