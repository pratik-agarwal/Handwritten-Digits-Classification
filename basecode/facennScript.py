'''
Comparing single layer MLP with deep MLP (using TensorFlow)
'''

import numpy as np
import pickle

# Do not change this
def initializeWeights(n_in,n_out):
    """
    # initializeWeights return the random weights for Neural Network given the
    # number of node in the input layer and output layer

    # Input:
    # n_in: number of nodes of the input layer
    # n_out: number of nodes of the output layer
                            
    # Output: 
    # W: matrix of random initial weights with size (n_out x (n_in + 1))"""
    epsilon = sqrt(6) / sqrt(n_in + n_out + 1);
    W = (np.random.rand(n_out, n_in + 1)*2* epsilon) - epsilon;
    return W


# Replace this with your sigmoid implementation
def sigmoid(z):
    sig = 1.0 / (1.0 + np.exp(-1.0 * z))
    return sig 
    
# Replace this with your nnObjFunction implementation
def nnObjFunction(params, *args):
    n_input, n_hidden, n_class, training_data, training_label, lambdaval = args

    w1 = params[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
    w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))
    obj_val = 0

    # Your code here
    y = np.zeros((training_label.shape[0], n_class))
    y[np.arange(training_label.shape[0], dtype = "int"), training_label.astype(int)] = 1

    training_data = np.column_stack((np.array(training_data), np.array(np.ones(training_label.shape[0]))))

    K = (sigmoid(np.dot(training_data, w1.T))).shape[0]
    Z = np.column_stack((sigmoid(np.dot(training_data, w1.T)), np.ones((sigmoid(np.dot(training_data, w1.T))).shape[0])))
    O = sigmoid(np.dot(Z, w2.T))
    
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

    return (obj_val, obj_grad)

    
# Replace this with your nnPredict implementation
def nnPredict(w1,w2,data):
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
    

# Do not change this
def preprocess():
    pickle_obj = pickle.load(file=open('face_all.pickle', 'rb'))
    features = pickle_obj['Features']
    labels = pickle_obj['Labels']
    train_x = features[0:21100] / 255
    valid_x = features[21100:23765] / 255
    test_x = features[23765:] / 255

    labels = labels[0]
    train_y = labels[0:21100]
    valid_y = labels[21100:23765]
    test_y = labels[23765:]
    return train_x, train_y, valid_x, valid_y, test_x, test_y

"""**************Neural Network Script Starts here********************************"""
train_data, train_label, validation_data, validation_label, test_data, test_label = preprocess()
#  Train Neural Network
# set the number of nodes in input unit (not including bias unit)
n_input = train_data.shape[1]
# set the number of nodes in hidden unit (not including bias unit)
n_hidden = 256
# set the number of nodes in output unit
n_class = 2

# initialize the weights into some random matrices
initial_w1 = initializeWeights(n_input, n_hidden);
initial_w2 = initializeWeights(n_hidden, n_class);
# unroll 2 weight matrices into single column vector
initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()),0)
# set the regularization hyper-parameter
lambdaval = 10;
args = (n_input, n_hidden, n_class, train_data, train_label, lambdaval)

#Train Neural Network using fmin_cg or minimize from scipy,optimize module. Check documentation for a working example
opts = {'maxiter' :50}    # Preferred value.

nn_params = minimize(nnObjFunction, initialWeights, jac=True, args=args,method='CG', options=opts)
params = nn_params.get('x')
#Reshape nnParams from 1D vector into w1 and w2 matrices
w1 = params[0:n_hidden * (n_input + 1)].reshape( (n_hidden, (n_input + 1)))
w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))

#Test the computed parameters
predicted_label = nnPredict(w1,w2,train_data)
#find the accuracy on Training Dataset
print('\n Training set Accuracy:' + str(100*np.mean((predicted_label == train_label).astype(float))) + '%')
predicted_label = nnPredict(w1,w2,validation_data)
#find the accuracy on Validation Dataset
print('\n Validation set Accuracy:' + str(100*np.mean((predicted_label == validation_label).astype(float))) + '%')
predicted_label = nnPredict(w1,w2,test_data)
#find the accuracy on Validation Dataset
print('\n Test set Accuracy:' +  str(100*np.mean((predicted_label == test_label).astype(float))) + '%')