#!/usr/bin/env python3

#Jansen Long

# A python script that creates a CNN to classify fMRI images from the roi cc400 of the
#ABIDE dataset

import warnings
warnings.filterwarnings("ignore")
from nilearn import decomposition
import numpy as np
from nilearn.connectome import ConnectivityMeasure
from nilearn.connectome import sym_matrix_to_vec
from nilearn import plotting
from nilearn.plotting import plot_roi
import tensorflow as tf
from tensorflow.keras import layers, models,optimizers
from tensorflow.keras.layers import Dropout as Dropout
import matplotlib.pyplot as plt
import keras
from sklearn.preprocessing import normalize
from sklearn.preprocessing import OneHotEncoder
from nilearn.datasets import fetch_abide_pcp
from sklearn.model_selection import train_test_split, KFold
from tensorflow.keras import regularizers
from tensorflow.keras import backend


inputShape = (871,392,392,1)
seed = 0

def Create_Model(dropout_rate = 0.2, lr = 0.001, act = 'relu'):
    model = models.Sequential()
    model.add(layers.Conv2D(filters = 16,kernel_size = (2,2), kernel_initializer='random_normal', activation=act, kernel_regularizer=regularizers.l2(l=0.01), input_shape=inputShape[1:],  data_format="channels_last"))
    model.add(Dropout(dropout_rate))
    model.add(layers.MaxPooling2D((2, 2))) 
    #Dense Layers
    model.add(layers.Flatten())

    #Output Layer
    model.add(layers.Dense(2, activation = 'softmax'))
    
    model.compile(optimizer = optimizers.Adam(learning_rate=lr),
                  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
                  metrics=['accuracy', 'AUC', 'Precision', 'Recall'])
    return model

#This evaluates the models recall, specificity, and precision
def eval_model(m, X_test, y_test):
    x = m.predict(X_test)
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for i, j in zip(x,y_test):
        #if a negative prediction
        if i[0] > i[1]:
            #if true negative
            if j[0] == 1:
                tn += 1
            #if false negative
            elif j[0] == 0:
                fn += 1
        #if positive prediction
        else:
            #if true positive
            if j[1] == 1:
                tp += 1
            #if false positive
            elif j[1] == 0:
                fp += 1
    if (tp+fn) == 0:
        recall = 0
    else:
        recall = (tp)/(tp+fn)
        
    if (tp+fp) == 0:
        pre = 0
    else:
        pre = (tp)/(tp+fp)
        
    if (tn+fp) == 0:
        spec = 0
    else:
        spec = (tn)/(tn+fp)
        
    return recall, pre, spec

def KFold_Validation_CNN(X,Y, act_fun = 'relu', b_size = 64, ep = 25, learning_rate = 0.0005, dropout = 0.2, out_fold = 2, repeat = 5):
    #evaluation indeces: Loss = 0, acc = 1, auc = 2, precision = 3, recall = 4    
    loss = 0
    acc = 1
    auc = 2
    precision = 3
    recall = 4
    metrics = {"Accuracy":np.zeros((repeat, out_fold)),
              "Loss": np.zeros((repeat, out_fold)),
              "AUC":  np.zeros((repeat, out_fold)),
              "Precision": np.zeros((repeat, out_fold)),
              "Recall":  np.zeros((repeat, out_fold)),
              "Specificity": np.zeros((repeat, out_fold))}
    for i in range(repeat):
        k_folds = KFold(n_splits = out_fold, shuffle = True, random_state = i)
        j = 0
        for train_index, val_index in k_folds.split(X):
            X_train, X_test = X[train_index,:], X[val_index,:]
            y_train, y_test = Y[train_index,:], Y[val_index,:]
            model = Create_Model(dropout_rate = dropout, lr = learning_rate, act = act_fun)
            #print(X_train.shape, X_val.shape, y_train.shape, y_val.shape)
            history = model.fit(X_train,y_train, validation_data = (X_test,y_test), batch_size=b_size, epochs=ep, verbose = True)
            print(f"Fold {j+1} for iteration {i+1}")
            #plot_history(i,history)
            evaluation = model.evaluate(X_test, y_test)

            metrics["Accuracy"][i,j] = evaluation[acc]
            metrics["Loss"][i,j] = evaluation[loss]
            metrics["AUC"][i,j] = evaluation[auc]
            metrics["Recall"][i,j], metrics["Precision"][i,j], metrics["Specificity"][i,j] = eval_model(model, X_test, y_test)
            j += 1
        tf.keras.backend.clear_session()
    return metrics

# list all data in history
#print(history.history.keys())
def plot_history(iteration, history):
    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.ylim([0,1])
    plt.title('model accuracy for model ' + str(iteration))
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train','test'], loc='upper left')
    plt.show()
    # Summarize history for AUC
    plt.plot(history.history['auc'])
    plt.plot(history.history['val_auc'])
    plt.ylim([0,1])
    plt.title('model AUC for model ' + str(iteration))
    plt.ylabel('AUC')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.ylim([0,10])
    plt.title('model loss for model ' + str(iteration))
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

if __name__ == '__main__':
    
    print("Importing Data")
    abide_cc400 = abide_cc400 = fetch_abide_pcp(derivatives = ['rois_cc400'], pipeline = 'cpac', quality_checked = True, verbose = 0)
    y_cc400 = abide_cc400.phenotypic['DX_GROUP']
    
        #Compute Connectivity matrices
    print("Computing Matrices")
    corrMeasure = ConnectivityMeasure(kind = "correlation", vectorize = False)
    #tanMeasure = ConnectivityMeasure(kind = "tangent", vectorize = False)
    #covMeasure = ConnectivityMeasure(kind = 'covariance', vectorize = False)
    #partCorrMeasure = ConnectivityMeasure(kind = "partial correlation", vectorize = False)
    #preMeasure = ConnectivityMeasure(kind = "precision", vectorize = False)
    

    #rois_cc400
    corMatrix_cc400 = corrMeasure.fit_transform(abide_cc400.rois_cc400)
    #np.savetxt("./Connectivity_Data/corMatrix_cc400.csv", corMatrix_cc400, delimiter=",")
    #tanMatrix_cc400 = tanMeasure.fit_transform(abide_cc400.rois_cc400)
    #np.savetxt("./Connectivity_Data/tanMatrix_cc400.csv", tanMatrix_cc400, delimiter=",")
    #covMatrix_cc400 = covMeasure.fit_transform(abide_cc400.rois_cc400)
    #np.savetxt("./Connectivity_Data/covMatrix_cc400.csv", covMatrix_cc400, delimiter=",")
    #partCorrMatrix_cc400 = partCorrMeasure.fit_transform(abide_cc400.rois_cc400)
    #np.savetxt("./Connectivity_Data/partCorrMatrix_cc400.csv", partCorrMatrix_cc400, delimiter=",")
    #preMatrix_cc400 = preMeasure.fit_transform(abide_cc400.rois_cc400)
    #np.savetxt("./Connectivity_Data/preMatrix_cc400.csv", preMatrix_cc400, delimiter=",")
    
    #One hot encode output
    OneHot = OneHotEncoder()
    y_cc400 = y_cc400.reshape((-1,1))
    OneHot.fit(y_cc400)
    Y = OneHot.transform(y_cc400).toarray()
    #Reshaping the connectivity matrices
    corMatrix_cc400 = corMatrix_cc400.reshape(inputShape)
    #tanMatrix_cc400 = tanMatrix_cc400.reshape(inputShape)
    #covMatrix_cc400 = covMatrix_cc400.reshape(inputShape)
    #partCorrMatrix_cc400 = partCorrMatrix_cc400.reshape(inputShape)
    #preMatrix_cc400 = preMatrix_cc400.reshape(inputShape)
    
    print("Training models")
    
    print("Training Model with ReLU on Correlation")
    corRel = KFold_Validation_CNN(corMatrix_cc400, Y, act_fun = 'relu',  out_fold = 10,ep = 75, b_size = 100, dropout = 0.2, learning_rate = 0.001)
    print(corRel)
    print("##################################################################################################################################")
    print("Training Model with Tanh on Correlation")
    corTan = KFold_Validation_CNN(corMatrix_cc400, Y, act_fun = 'tanh',  out_fold = 10,ep = 75, b_size = 100, dropout = 0.2, learning_rate = 0.001)
    print(corTan)
    print("##################################################################################################################################")
    print("Training Model with Leaky Relu on Correlation")
    corLRel = KFold_Validation_CNN(corMatrix_cc400, Y, act_fun = 'leaky_relu',  out_fold = 10,ep = 75, b_size = 100, dropout = 0.2, learning_rate = 0.001)
    print(corLRel)
    
    print("##################################################################################################################################")
    '''print("Training Model with ReLu on Tangent Matrix")
    tanRel = KFold_Validation_CNN(tanMatrix_cc400, Y, act_fun = 'relu',  out_fold = 10,ep = 75, b_size = 100, dropout = 0.2, learning_rate = 0.001)
    print(tanRel)
    print("##################################################################################################################################")
    print("Training Model with Tanh on Tangent Matrix")
    tanTan = KFold_Validation_CNN(tanMatrix_cc400, Y, act_fun = 'tanh',  out_fold = 10,ep = 75, b_size = 100, dropout = 0.2, learning_rate = 0.001)
    print(tanTan)
    print("##################################################################################################################################")
    print("Training Model with Leaky Relu on Tangent Matrix")
    tanLRel = KFold_Validation_CNN(tanMatrix_cc400, Y, act_fun = 'leaky_relu',  out_fold = 10,ep = 75, b_size = 100, dropout = 0.2, learning_rate = 0.001)
    print(tanLRel)
    
    print("##################################################################################################################################")
    print("Training Model with ReLu on Covariance")
    covRel = KFold_Validation_CNN(covMatrix_cc400, Y, act_fun = 'relu',  out_fold = 10,ep = 75, b_size = 100, dropout = 0.2, learning_rate = 0.001)
    print(covRel)
    print("##################################################################################################################################")
    print("Training Model with Tanh on Covariance")
    covTan = KFold_Validation_CNN(covMatrix_cc400, Y, act_fun = 'tanh',  out_fold = 10,ep = 75, b_size = 100, dropout = 0.2, learning_rate = 0.001)
    print(covTan)
    print("##################################################################################################################################")
    print("Training Model with Leaky Relu on Covariance")
    covLRel = KFold_Validation_CNN(covMatrix_cc400, Y, act_fun = 'leaky_relu',  out_fold = 10,ep = 75, b_size = 100, dropout = 0.2, learning_rate = 0.001)
    print(covLRel)
    
    print("##################################################################################################################################")
    print("Training Model with ReLu on Partial Correlation")
    pCorRel = KFold_Validation_CNN(partCorrMatrix_cc400, Y, act_fun = 'relu',  out_fold = 10,ep = 75, b_size = 100, dropout = 0.2, learning_rate = 0.001)
    print(pCorRel)
    print("##################################################################################################################################")
    print("Training Model with Tanh on Partial Correlation")
    pCorTan = KFold_Validation_CNN(partCorrMatrix_cc400, Y, act_fun = 'tanh',  out_fold = 10,ep = 75, b_size = 100, dropout = 0.2, learning_rate = 0.001)
    print(pCorTan)
    print("##################################################################################################################################")
    print("Training Model with Leaky Relu on Partial Correlation")
    pCorLRel = KFold_Validation_CNN(partCorrMatrix_cc400, Y, act_fun = 'leaky_relu',  out_fold = 10,ep = 75, b_size = 100, dropout = 0.2, learning_rate = 0.001)
    print(pCorLRel)
    
    print("##################################################################################################################################")
    print("Training Model with ReLu on Precision")
    preRel = KFold_Validation_CNN(preMatrix_cc400, Y, act_fun = 'relu',  out_fold = 10,ep = 75, b_size = 100, dropout = 0.2, learning_rate = 0.001)
    print(preRel)
    print("##################################################################################################################################")
    print("Training Model with Tanh on Precision")
    preTan = KFold_Validation_CNN(preMatrix_cc400, Y, act_fun = 'tanh',  out_fold = 10,ep = 75, b_size = 100, dropout = 0.2, learning_rate = 0.001)
    print(preTan)
    print("##################################################################################################################################")
    print("Training Model with Leaky Relu on Precision")
    preLRel = KFold_Validation_CNN(preMatrix_cc400, Y, act_fun = 'leaky_relu',  out_fold = 10,ep = 75, b_size = 100, dropout = 0.2, learning_rate = 0.001)
    print(preLRel)'''
    
    
    
    
    
    
    
    
    




