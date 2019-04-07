
import numpy as np
import timeit
import csv
from scipy.optimize import minimize
from scipy.io import loadmat
from math import sqrt
import pickle


def initializeWeights(n_in, n_out):
    """
    # initializeWeights return the random weights for Neural Network given the
    # number of node in the input layer and output layer

    # Input:
    # n_in: number of nodes of the input layer
    # n_out: number of nodes of the output layer
       
    # Output: 
    # W: matrix of random initial weights with size (n_out x (n_in + 1))"""

    epsilon = sqrt(6) / sqrt(n_in + n_out + 1)
    W = (np.random.rand(n_out, n_in + 1) * 2 * epsilon) - epsilon
    return W


def sigmoid(z):
    """# Notice that z can be a scalar, a vector or a matrix
    # return the sigmoid of input z"""
    # your code here
    
    sig = 1.0 / (1.0 + np.exp(-1.0 * z))
    return sig 
    

def featureSelection(data):
    features = []
    D = data.shape[1] #Total d features
    for x in range(D):
        if sum(data[:, x]) > 0:
            features += [x]
    return features 

def preprocess():
    """ Input:
     Although this function doesn't have any input, you are required to load
     the MNIST data set from file 'mnist_all.mat'.

     Output:
     train_data: matrix of training set. Each row of train_data contains 
       feature vector of a image
     train_label: vector of label corresponding to each image in the training
       set
     validation_data: matrix of training set. Each row of validation_data 
       contains feature vector of a image
     validation_label: vector of label corresponding to each image in the 
       training set
     test_data: matrix of training set. Each row of test_data contains 
       feature vector of a image
     test_label: vector of label corresponding to each image in the testing
       set

     Some suggestions for preprocessing step:
     - feature selection"""

    mat = loadmat('mnist_all.mat')  # loads the MAT object as a Dictionary

    # Pick a reasonable size for validation data

    #Initialize:
    
    train_size = 50000
    feature_size = 784
    test_size = 10000
    preprocess_train = np.zeros(shape=(50000, 784))
    preprocess_validation = np.zeros(shape=(10000, 784))
    preprocess_test = np.zeros(shape=(10000, 784))
    preprocess_train_label = np.zeros(shape=(50000,))
    preprocess_validation_label = np.zeros(shape=(10000,))
    preprocess_test_label = np.zeros(shape=(10000,))
    
    len_train = 0
    len_validation = 0
    len_test = 0
    len_train_label = 0
    len_validation_label = 0
    reduceBy=1000

    
    for i in mat:
        data = mat.get(i)
        length = len(data)
        if "train" in i:
            adjust = length - reduceBy
            # ---------------------adding data to training set-------------------------#
            preprocess_train,len_train,train_label,len_train_label = data_add(preprocess_train,len_train,adjust,data[np.random.permutation(range(length))[1000:], :],i,preprocess_train_label,len_train_label)

            # ---------------------adding data to validation set-------------------------#
            preprocess_validation,len_validation,preprocess_validation_label,len_validation_label = data_add(preprocess_validation,len_validation,1000,data[np.random.permutation(range(length))[0:1000],:],i,preprocess_validation_label,len_validation_label)

            # ---------------------adding data to test set-------------------------#
        elif "test" in i:
            preprocess_test_label[len_test:len_test + length] = i[len(i) - 1]
            preprocess_test[len_test:len_test + length] = data[np.random.permutation(range(length))]
            len_test += length
            # ---------------------Shuffle,double and normalize-------------------------#
            
    train_size = range(preprocess_train.shape[0])
    train_data,train_label = dsn(train_size,preprocess_train,preprocess_train_label)
    
    #print(train_preprocess[49999])
    validation_size = range(preprocess_validation.shape[0])
    validation_data,validation_label = dsn(validation_size,preprocess_validation,preprocess_validation_label)

    test_size = range(preprocess_test.shape[0])
    test_data,test_label = dsn(test_size,preprocess_test,preprocess_test_label)

    # Feature selection
    # Your code here.

    global features
    features = []
    features = featureSelection(train_data)                 
    train_data = train_data[:, features]
    validation_data = validation_data[:, features]
    test_data = test_data[:, features]

    # print(train_data.shape)
    # print(test_data.shape)
    # print(validation_data.shape)

    print('preprocess done')
    return train_data, train_label, validation_data, validation_label, test_data, test_label


def data_add(newData,datalen,adjust,data,key,datalabel,datalabellen):
  
    newData[datalen:datalen+adjust] = data
    datalen+= adjust
    
    datalabel[datalabellen:datalabellen + adjust] = key[len(key)-1]
    datalabellen += adjust
    return newData,datalen,datalabel,datalabellen

def dsn(size,pre_process,label_preprocess): #Double, Shuffle, Normalize
    
    train_perm = np.random.permutation(size)
    train_data = pre_process[train_perm]
    train_data = np.double(train_data)
    train_data = train_data / 255.0
    train_label = label_preprocess[train_perm]
    return train_data,train_label

def nnObjFunction(params, *args):
    """% nnObjFunction computes the value of objective function (negative log 
    %   likelihood error function with regularization) given the parameters 
    %   of Neural Networks, thetraining data, their corresponding training 
    %   labels and lambda - regularization hyper-parameter.

    % Input:
    % params: vector of weights of 2 matrices w1 (weights of connections from
    %     input layer to hidden layer) and w2 (weights of connections from
    %     hidden layer to output layer) where all of the weights are contained
    %     in a single vector.
    % n_input: number of node in input layer (not include the bias node)
    % n_hidden: number of node in hidden layer (not include the bias node)
    % n_class: number of node in output layer (number of classes in
    %     classification problem
    % training_data: matrix of training data. Each row of this matrix
    %     represents the feature vector of a particular image
    % training_label: the vector of truth label of training images. Each entry
    %     in the vector represents the truth label of its corresponding image.
    % lambda: regularization hyper-parameter. This value is used for fixing the
    %     overfitting problem.
       
    % Output: 
    % obj_val: a scalar value representing value of error function
    % obj_grad: a SINGLE vector of gradient value of error function
    % NOTE: how to compute obj_grad
    % Use backpropagation algorithm to compute the gradient of error function
    % for each weights in weight matrices.

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % reshape 'params' vector into 2 matrices of weight w1 and w2
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit j in input 
    %     layer to unit i in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit j in hidden 
    %     layer to unit i in output layer."""

    n_input, n_hidden, n_class, training_data, training_label, lambdaval = args

    w1 = params[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
    w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))
    obj_val = 0

    # Your code here
    N = training_label.shape[0]

    y = np.zeros((N, n_class))
    y[np.arange(N, dtype = "int"), training_label.astype(int)] = 1

    training_data = np.column_stack((np.array(training_data), np.array(np.ones(N))))

    z = np.dot(training_data, w1.T)
    Z = sigmoid(z)
    K = Z.shape[0]
    Z = np.column_stack((Z, np.ones(K)))
    o = np.dot(Z, w2.T)
    O = sigmoid(o)

    # deltaL = Ol - Yl
    error_obtained = O - y

    gradient_w1 = np.dot(((1 - Z) * Z * (np.dot(error_obtained, w2))).T, training_data)

    # Delete 0 type variables
    delete_var = 0

    gradient_w1 = np.delete(gradient_w1, n_hidden, delete_var)

    gradient_w2 = np.dot(error_obtained.T, Z)

    obj_val = (np.sum(-1 * (y * np.log(O) + (1 - y) * np.log(1 - O)))) / training_data.shape[0] + ((lambdaval / (2 * training_data.shape[0])) * (np.sum(np.square(w1)) + np.sum(np.square(w2))))

    # Make sure you reshape the gradient matrices to a 1D array. for instance if your gradient matrices are grad_w1 and grad_w2
    # you would use code similar to the one below to create a flat array

    gradient_w1 = (gradient_w1 + (lambdaval * w1)) / training_data.shape[0]
    gradient_w2 = (gradient_w2 + (lambdaval * w2)) / training_data.shape[0]

    obj_grad = np.array([])
    obj_grad = np.concatenate((gradient_w1.flatten(), gradient_w2.flatten()), 0)


    # print(obj_val)
    # print(obj_grad)

    # global print_obj_val
    # print_obj_val = obj_val

    return (obj_val, obj_grad)


def nnPredict(w1, w2, data):
    """% nnPredict predicts the label of data given the parameter w1, w2 of Neural
    % Network.

    % Input:
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit i in input 
    %     layer to unit j in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit i in input 
    %     layer to unit j in hidden layer.
    % data: matrix of data. Each row of this matrix represents the feature 
    %       vector of a particular image
       
    % Output: 
    % label: a column vector of predicted labels"""

    # Your code here
    N, D = data.shape
    
    # Bias
    B = np.ones(N)*1

    # Concatenating Bias
    data = np.column_stack((data, B))

    # Output at the hidden layer
    z = sigmoid(np.dot(data, w1.T))

    # Output at the output layer
    o = sigmoid(np.dot(np.column_stack((z, B.T)), w2.T))

    labels = o.argmax(axis = 1)

    # print(labels.shape)

    return labels


"""**************Neural Network Script Starts here********************************"""

train_data, train_label, validation_data, validation_label, test_data, test_label = preprocess()

#  Train Neural Network

# set the number of nodes in input unit (not including bias unit)
n_input = train_data.shape[1]

# set the number of nodes in hidden unit (not including bias unit)
# Setting the optimal value achieved by running the code as mentioned in the report which is 50
n_hidden = 50

# set the number of nodes in output unit
n_class = 10

'''
Looping to generate csv for tables

arr = [0, 5, 10, 15]
# arr = [50, ]
# arr2 = [4, 8, 12, 16, 20, 24, 28, 32]
file = open('test_1.csv', 'a') 
for i in range(4):
    lambdaval = arr[i]
'''

# initialize the weights into some random matrices
initial_w1 = initializeWeights(n_input, n_hidden)
initial_w2 = initializeWeights(n_hidden, n_class)

# unroll 2 weight matrices into single column vector
initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()), 0)

# set the regularization hyper-parameter
# Setting the optimal value achieved by running the code as mentioned in the report which is 5
lambdaval = 5

args = (n_input, n_hidden, n_class, train_data, train_label, lambdaval)

# Train Neural Network using fmin_cg or minimize from scipy,optimize module. Check documentation for a working example

opts = {'maxiter': 50}  # Preferred value.

# start = timeit.default_timer()

nn_params = minimize(nnObjFunction, initialWeights, jac=True, args=args, method='CG', options=opts)

# In Case you want to use fmin_cg, you may have to split the nnObjectFunction to two functions nnObjFunctionVal
# and nnObjGradient. Check documentation for this function before you proceed.
# nn_params, cost = fmin_cg(nnObjFunctionVal, initialWeights, nnObjGradient,args = args, maxiter = 50)


# Reshape nnParams from 1D vector into w1 and w2 matrices
w1 = nn_params.x[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
w2 = nn_params.x[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))

# Test the computed parameters

predicted_label = nnPredict(w1, w2, train_data)

# find the accuracy on Training Dataset

print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label == train_label).astype(float))) + '%')

predicted_label = nnPredict(w1, w2, validation_data)

# find the accuracy on Validation Dataset

print('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label == validation_label).astype(float))) + '%')

predicted_label = nnPredict(w1, w2, test_data)

# find the accuracy on Validation Dataset

print('\n Test set Accuracy:' + str(100 * np.mean((predicted_label == test_label).astype(float))) + '%')

obj_dump = [features, n_hidden, w1, w2, lambdaval]
pickle.dump(obj_dump, open("params.pickle", "wb"))

# stop = timeit.default_timer()
# total_time = stop - start
# file.write('{}, {}, {}, {}, {}, {}, {}\n'.format(lambdaval, n_hidden, print_obj_val, training_accuracy, validation_accuracy, testing_accuracy, total_time))
# file.close()

