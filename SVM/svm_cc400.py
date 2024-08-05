#!/usr/bin/env python3

import warnings
warnings.filterwarnings("ignore")
from nilearn.datasets import fetch_abide_pcp
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score, recall_score
# Linear SVM
from sklearn.svm import SVC
from nilearn.connectome import ConnectivityMeasure
from nilearn.connectome import sym_matrix_to_vec
import numpy as np
import time
from sklearn.model_selection import train_test_split, KFold
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score, balanced_accuracy_score, precision_score
from numba import jit, cuda
from sklearn.model_selection import RepeatedStratifiedKFold
import json
from json import JSONEncoder

class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)

#Function that trains and test a SVC model for each combination of parameters as laid out
#Returns the scores with the highest AUC and the parameters of the model with the best AUC
@jit
def OptimizeRegularization(test_kernel, X_data, labels, folds = 10):
    total_time = 0
    max_AUC = 0
    max_scores = None
    params ={"C":0, "Shrinking":True, "Gamma":"scale"}
    iteration = 1
    #Testing with gamma = Scale, Shrink = True
    for i in [10000, 5000, 1000, 100, 10, 1, 0.1, 0.01]:
        timer = time.perf_counter()
        svm = SVC(kernel=test_kernel,C = i, shrinking = True, gamma = 'scale')
        #print(type(svm))
        scores = Train_Test_Model(svm, X_data, labels, num_folds = folds)
        if np.average(scores['test_AUC']) > max_AUC:
            max_scores = scores
            max_AUC = np.average(scores['test_AUC'])
            params["C"] = i
            params["Shrinking"] = True
            params["Gamma"] = "scale"
        iter_time = time.perf_counter()-timer
        total_time += iter_time 
        print("Iteration", iteration, iter_time, "seconds")
        iteration += 1
    #Gama = auto, shrink = True
    for i in [10000, 5000, 1000, 100, 10, 1, 0.1, 0.01]:
        timer = time.perf_counter()
        svm = SVC(kernel=test_kernel,C = i, shrinking = True, gamma = 'auto')
        #print(type(svm))
        scores = Train_Test_Model(svm, X_data, labels, num_folds = folds)
        if np.average(scores['test_AUC']) > max_AUC:
            max_scores = scores
            max_AUC = np.average(scores['test_AUC'])
            params["C"] = i
            params["Shrinking"] = True
            params["Gamma"] = "scale" 
        iter_time = time.perf_counter()-timer
        total_time += iter_time 
        print("Iteration", iteration, iter_time, "seconds")
        iteration += 1
    #Gama = Scale, Shrinking = False
    for i in [10000, 5000, 1000, 100, 10, 1, 0.1, 0.01]:
        timer = time.perf_counter()
        svm = SVC(kernel=test_kernel,C = i, shrinking = False, gamma = 'scale')
        #print(type(svm))
        scores = Train_Test_Model(svm, X_data, labels, num_folds = folds)
        if np.average(scores['test_AUC']) > max_AUC:
            max_scores = scores
            max_AUC = np.average(scores['test_AUC'])
            params["C"] = i
            params["Shrinking"] = True
            params["Gamma"] = "scale" 
        iter_time = time.perf_counter()-timer
        total_time += iter_time 
        print("Iteration", iteration, iter_time, "seconds")
        iteration += 1
    #Gama = auto, Shrinking = False    
    for i in [10000, 5000, 1000, 100, 10, 1, 0.1, 0.01]:
        timer = time.perf_counter()
        svm = SVC(kernel=test_kernel,C = i,  shrinking = False,gamma = 'auto')
        #print(type(svm))
        scores = Train_Test_Model(svm, X_data, labels, num_folds = folds)
        if np.average(scores['test_AUC']) > max_AUC:
            max_scores = scores
            max_AUC = np.average(scores['test_AUC'])
            params["C"] = i
            params["Shrinking"] = True
            params["Gamma"] = "scale"
        iter_time = time.perf_counter()-timer
        total_time += iter_time 
        print("Iteration", iteration, iter_time, "seconds")
        iteration += 1
    print("Total Time =", total_time, "seconds\n")
    print("AUC:", np.average(max_scores['test_AUC']), "\nAccuracy: ", np.average(max_scores['test_accuracy']), "\nSpecificity:",np.average(max_scores['test_specificity']),"\nSensitivity:", np.average(max_scores['test_sensitivity']), sep = ' ')
    print("Optimal C:", params["C"], "\nShrinking:", params["Shrinking"], "\nGamma:", params["Gamma"])
    return max_scores, params

scoring = {
    'accuracy': make_scorer(accuracy_score),
    'sensitivity': make_scorer(recall_score),
    'specificity': make_scorer(recall_score,pos_label=0),
    'AUC': make_scorer(roc_auc_score),
    'Balanced Accuracy': make_scorer(balanced_accuracy_score),
    'precision': make_scorer(precision_score, zero_division = 0),
    'recall': make_scorer(recall_score, zero_division = 0)
}
#Function that trains the SVC models and returns a set of scores as defined above
@jit
def Train_Test_Model(model, X_data, labels, seed = 0, num_folds = 10):
    skf = RepeatedStratifiedKFold(n_splits = num_folds, n_repeats = 5, random_state = seed)
    scores = cross_validate(model, X_data, labels, scoring = scoring, cv = skf, n_jobs = 12, return_train_score = True, verbose = 0)
    return scores


if __name__ == '__main__':

    print("Importing Data")
    abide_cc400 = fetch_abide_pcp(derivatives = ['rois_cc400'], pipeline = 'cpac', quality_checked = True, verbose = 0)
    
    y_cc400 = abide_cc400.phenotypic['DX_GROUP']
    y_cc400[y_cc400 == 2] = 0

    corrMeasure = ConnectivityMeasure(kind = "correlation", vectorize = True)
    tanMeasure = ConnectivityMeasure(kind = "tangent", vectorize = True)
    covMeasure = ConnectivityMeasure(kind = 'covariance', vectorize = True)
    partCorrMeasure = ConnectivityMeasure(kind = "partial correlation", vectorize = True)
    preMeasure = ConnectivityMeasure(kind = "precision", vectorize = True)

    print("Transforming Data")
    '''corMatrix_cc400 = corrMeasure.fit_transform(abide_cc400.rois_cc400)
    print(corMatrix_cc400.shape)
    tanMatrix_cc400 = tanMeasure.fit_transform(abide_cc400.rois_cc400)
    print(tanMatrix_cc400.shape)
    covMatrix_cc400 = covMeasure.fit_transform(abide_cc400.rois_cc400)
    print(covMatrix_cc400.shape)'''
    partCorrMatrix_cc400 = partCorrMeasure.fit_transform(abide_cc400.rois_cc400)
    print(partCorrMatrix_cc400.shape)
    preMatrix_cc400 = preMeasure.fit_transform(abide_cc400.rois_cc400)
    print(preMatrix_cc400.shape)

    Names = ["./Results/SVM/cc400/Linear_Cor.json", "./Results/SVM/cc400/RBF-Cor.json", "./Results/SVM/cc400/Sigmoid-Cor.json", "./Results/SVM/cc400/Poly-Cor.json",
            "./Results/SVM/cc400/Linear-Tan.json", "./Results/SVM/cc400/RBF-Tan.json", "./Results/SVM/cc400/Sigmoid-Tan.json", "./Results/SVM/cc400/Poly-Tan.json",
            "./Results/SVM/cc400/Linear-Cov.json", "./Results/SVM/cc400/RBF-Cov.json", "./Results/SVM/cc400/Sigmoid-Cov.json", "./Results/SVM/cc400/Poly-Cov.json",
            "./Results/SVM/cc400/Linear-PCor.json", "./Results/SVM/cc400/RBF-PCor.json", "./Results/SVM/cc400/Sigmoid-PCor.json", "./Results/SVM/cc400/Poly-PCor.json",
            "./Results/SVM/cc400/Linear-Pre.json", "./Results/SVM/cc400/RBF-Pre.json", "./Results/SVM/cc400/Sigmoid-Pre.json", "./Results/SVM/cc400/Poly-Pre.json"]
    i = 0
    
    print("Running tests")
    cc400_linearKernel_cor_scores, cc400_linearKernel_corParams  = OptimizeRegularization('linear', corMatrix_cc400, y_cc400)
    out_file = open(Names[i], "w")
    print("Writing to ", Names[i])
    i+=1
    json.dump(cc400_linearKernel_cor_scores, out_file, cls = NumpyArrayEncoder)
    out_file.close()
    
    cc400_rbf_kernel_cor_scores, cc400_rbf_kernel_cor_Optimal_params = OptimizeRegularization('rbf', corMatrix_cc400, y_cc400)
    out_file = open(Names[i], "w")
    print("Writing to ", Names[i])
    i+=1
    json.dump(cc400_rbf_kernel_cor_scores, out_file, cls = NumpyArrayEncoder)
    out_file.close()
    
    cc400_sigmoidKernel_cor_scores, cc400_sigmoidKernel_cor_Optimal_C = OptimizeRegularization('sigmoid', corMatrix_cc400, y_cc400)
    out_file = open(Names[i], "w")
    print("Writing to ", Names[i])
    i+=1
    json.dump(cc400_sigmoidKernel_cor_scores, out_file, cls = NumpyArrayEncoder)
    out_file.close()

    cc400_polyKernel_cor_scores, cc400_polyKernel_cor_Optimal_C = OptimizeRegularization('poly', corMatrix_cc400, y_cc400)
    out_file = open(Names[i], "w")
    print("Writing to ", Names[i])
    i+=1
    json.dump(cc400_polyKernel_cor_scores,out_file, cls = NumpyArrayEncoder)
    out_file.close()
    
    cc400_linearKernel_tan_scores, cc400_linearKernel_tan_Optimal_C = OptimizeRegularization('linear', tanMatrix_cc400, y_cc400)
    out_file = open(Names[i], "w")
    print("Writing to ", Names[i])
    i+=1
    json.dump(cc400_linearKernel_tan_scores, out_file, cls = NumpyArrayEncoder)
    out_file.close()

    cc400_rbfKernel_tan_scores, cc400_rbfKernel_tan_Optimal_C = OptimizeRegularization('rbf', tanMatrix_cc400, y_cc400)
    out_file = open(Names[i], "w")
    print("Writing to ", Names[i])
    i+=1
    json.dump(cc400_rbfKernel_tan_scores, out_file, cls = NumpyArrayEncoder)
    out_file.close()
    
    cc400_SigKernel_tan_scores, cc400_SigKernel_tan_Optimal_C = OptimizeRegularization('sigmoid', tanMatrix_cc400, y_cc400)
    out_file = open(Names[i], "w")
    print("Writing to ", Names[i])
    i+=1
    json.dump(cc400_SigKernel_tan_scores, out_file, cls = NumpyArrayEncoder)
    out_file.close()
    
    cc400_PolyKernel_tan_scores, cc400_PolyKernel_tan_Optimal_C = OptimizeRegularization('poly', tanMatrix_cc400, y_cc400)
    out_file = open(Names[i], "w")
    print("Writing to ", Names[i])
    i+=1
    json.dump(cc400_PolyKernel_tan_scores, out_file, cls = NumpyArrayEncoder)
    out_file.close()

    cc400_LinearKernel_cov_scores, cc400_LinearKernel_cov_Optimal_C = OptimizeRegularization('linear', covMatrix_cc400, y_cc400)
    out_file = open(Names[i], "w")
    print("Writing to ", Names[i])
    i+=1
    json.dump(cc400_LinearKernel_cov_scores, out_file, cls = NumpyArrayEncoder)
    out_file.close()
    
    cc400_rbfKernel_cov_scores, cc400_rbfKernel_cov_Optimal_C = OptimizeRegularization('rbf', covMatrix_cc400, y_cc400)
    out_file = open(Names[i], "w")
    print("Writing to ", Names[i])
    i+=1
    json.dump(cc400_rbfKernel_cov_scores, out_file, cls = NumpyArrayEncoder)
    out_file.close()
    
    cc400_SigKernel_cov_scores, cc400_SigKernel_cov_Optimal_C = OptimizeRegularization('sigmoid', covMatrix_cc400, y_cc400)
    out_file = open(Names[i], "w")
    print("Writing to ", Names[i])
    i+=1
    json.dump(cc400_SigKernel_cov_scores, out_file, cls = NumpyArrayEncoder)
    out_file.close()
    
    cc400_PolyKernel_cov_scores, cc400_PolyKernel_cov_Optimal_C = OptimizeRegularization('poly', covMatrix_cc400, y_cc400)
    out_file = open(Names[i], "w")
    print("Writing to ", Names[i])
    i+=1
    json.dump(cc400_PolyKernel_cov_scores, out_file, cls = NumpyArrayEncoder)
    out_file.close()
    
    cc400_LinearKernel_PartCorr_scores, cc400_LinearKernel_PartCorr_Optimal_C = OptimizeRegularization('linear', partCorrMatrix_cc400, y_cc400)
    out_file = open(Names[i], "w")
    print("Writing to ", Names[i])
    i+=1
    json.dump(cc400_LinearKernel_PartCorr_scores, out_file, cls = NumpyArrayEncoder)
    out_file.close()
    
    cc400_rbfKernel_PartCorr_scores, cc400_rbfKernel_PartCorr_Optimal_C = OptimizeRegularization('rbf', partCorrMatrix_cc400, y_cc400)
    out_file = open(Names[i], "w")
    print("Writing to ", Names[i])
    i+=1
    json.dump(cc400_rbfKernel_PartCorr_scores, out_file, cls = NumpyArrayEncoder)
    out_file.close()
    
    cc400_SigKernel_PartCorr_scores, cc400_SigKernel_PartCorr_Optimal_C = OptimizeRegularization('sigmoid', partCorrMatrix_cc400, y_cc400)
    out_file = open(Names[i], "w")
    print("Writing to ", Names[i])
    i+=1
    json.dump(cc400_SigKernel_PartCorr_scores, out_file, cls = NumpyArrayEncoder)
    out_file.close()
    
    cc400_PolyKernel_PartCorr_scores, cc400_PolyKernel_PartCorr_Optimal_C = OptimizeRegularization('poly', partCorrMatrix_cc400, y_cc400)
    out_file = open(Names[i], "w")
    print("Writing to ", Names[i])
    i+=1
    json.dump(cc400_PolyKernel_PartCorr_scores, out_file, cls = NumpyArrayEncoder)
    out_file.close()
    
    cc400_LinearKernel_Pre_scores, cc400_LinearKernel_Pre_Optimal_C = OptimizeRegularization('linear', preMatrix_cc400, y_cc400)
    out_file = open(Names[i], "w")
    print("Writing to ", Names[i])
    i+=1
    json.dump(cc400_LinearKernel_Pre_scores, out_file, cls = NumpyArrayEncoder)
    out_file.close()
    
    cc400_rbfKernel_Pre_scores, cc400_rbfKernel_Pre_Optimal_C = OptimizeRegularization('rbf', preMatrix_cc400, y_cc400)
    out_file = open(Names[i], "w")
    print("Writing to ", Names[i])
    i+=1
    json.dump(cc400_rbfKernel_Pre_scores, out_file, cls = NumpyArrayEncoder)
    out_file.close()
    
    cc400_SigKernel_Pre_scores, cc400_SigKernel_Pre_Optimal_C = OptimizeRegularization('sigmoid', preMatrix_cc400, y_cc400)
    out_file = open(Names[i], "w")
    print("Writing to ", Names[i])
    i+=1
    json.dump(cc400_SigKernel_Pre_scores, out_file, cls = NumpyArrayEncoder)
    out_file.close()
    
    cc400_PolyKernel_Pre_scores, cc400_PolyKernel_Pre_Optimal_C = OptimizeRegularization('poly', preMatrix_cc400, y_cc400)
    out_file = open(Names[i], "w")
    print("Writing to ", Names[i])
    i+=1
    json.dump(cc400_PolyKernel_Pre_scores, out_file, cls = NumpyArrayEncoder)
    out_file.close()
