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
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score, balanced_accuracy_score
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

@jit(parallel = True)
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
        print(f"Iteration {iteration}: {iter_time} seconds")
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
        print(f"Iteration {iteration}: {iter_time} seconds")
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
        print(f"Iteration {iteration}: {iter_time} seconds")
        iteration += 1
    print(f"Total Time = {total_time} seconds\n")
    print(f"AUC: {np.average(max_scores['test_AUC'])} \nAccuracy:  {np.average(max_scores['test_accuracy'])} \nSpecificity:{np.average(max_scores['test_specificity'])} \nSensitivity: {np.average(max_scores['test_sensitivity'])}")
    print(f"Optimal C: {params['C']} \nShrinking: {params['Shrinking']} \nGamma: {params['Gamma']}")
    return max_scores, params

scoring = {
    'accuracy': make_scorer(accuracy_score),
    'sensitivity': make_scorer(recall_score),
    'specificity': make_scorer(recall_score,pos_label=0),
    'AUC': make_scorer(roc_auc_score),
    'Balanced Accuracy': make_scorer(balanced_accuracy_score)
}
#Function that trains the SVC models and returns a set of scores as defined above
@jit(parallel = True)
def Train_Test_Model(model, X_data, labels, seed = 0, num_folds = 10):
    skf = RepeatedStratifiedKFold(n_splits = num_folds, n_repeats = 5, random_state = seed)
    scores = cross_validate(model, X_data, labels, scoring = scoring, cv = skf, return_train_score = True, verbose = 0, n_jobs = 4)
    return scores


if __name__ == '__main__':
    
    print("Importing Data")
    abide_cc200 = fetch_abide_pcp(derivatives = ['rois_cc200'], pipeline = 'cpac', quality_checked = True, verbose = 0)
    
    y_cc200 = abide_cc200.phenotypic['DX_GROUP']
    y_cc200[y_cc200 == 2] = 0

    corrMeasure = ConnectivityMeasure(kind = "correlation", vectorize = True)
    tanMeasure = ConnectivityMeasure(kind = "tangent", vectorize = True)
    covMeasure = ConnectivityMeasure(kind = 'covariance', vectorize = True)
    partCorrMeasure = ConnectivityMeasure(kind = "partial correlation", vectorize = True)
    preMeasure = ConnectivityMeasure(kind = "precision", vectorize = True)

    print("Transforming Data")
    corMatrix_cc200 = corrMeasure.fit_transform(abide_cc200.rois_cc200)
    print(corMatrix_cc200.shape)
    tanMatrix_cc200 = tanMeasure.fit_transform(abide_cc200.rois_cc200)
    print(tanMatrix_cc200.shape)
    covMatrix_cc200 = covMeasure.fit_transform(abide_cc200.rois_cc200)
    print(covMatrix_cc200.shape)
    partCorrMatrix_cc200 = partCorrMeasure.fit_transform(abide_cc200.rois_cc200)
    print(partCorrMatrix_cc200.shape)
    preMatrix_cc200 = preMeasure.fit_transform(abide_cc200.rois_cc200)
    print(preMatrix_cc200.shape)

    print("Running tests")
    cc200_linearKernel_cor_scores, cc200_linearKernel_corParams  = OptimizeRegularization('linear', corMatrix_cc200, y_cc200)
    print(json.dumps(cc200_linearKernel_cor_scores, cls=NumpyArrayEncoder)) 
    
    cc200_rbf_kernel_cor_scores, cc200_rbf_kernel_cor_Optimal_params = OptimizeRegularization('rbf', corMatrix_cc200, y_cc200)
    print(json.dumps(cc200_rbf_kernel_cor_scores, cls=NumpyArrayEncoder))
    
    cc200_sigmoidKernel_cor_scores, cc200_sigmoidKernel_cor_Optimal_C = OptimizeRegularization('sigmoid', corMatrix_cc200, y_cc200)
    print(json.dumps(cc200_sigmoidKernel_cor_scores, cls=NumpyArrayEncoder))
    
    cc200_polyKernel_cor_scores, cc200_polyKernel_cor_Optimal_C = OptimizeRegularization('poly', corMatrix_cc200, y_cc200)
    print(json.dumps(cc200_polyKernel_cor_scores, cls=NumpyArrayEncoder))
    
    cc200_linearKernel_tan_scores, cc200_linearKernel_tan_Optimal_C = OptimizeRegularization('linear', tanMatrix_cc200, y_cc200)
    print(json.dumps(cc200_linearKernel_tan_scores, cls=NumpyArrayEncoder))
    
    cc200_rbfKernel_tan_scores, cc200_rbfKernel_tan_Optimal_C = OptimizeRegularization('rbf', tanMatrix_cc200, y_cc200)
    print(json.dumps(cc200_rbfKernel_tan_scores, cls=NumpyArrayEncoder))
    
    cc200_SigKernel_tan_scores, cc200_SigKernel_tan_Optimal_C = OptimizeRegularization('sigmoid', tanMatrix_cc200, y_cc200)
    print(json.dumps(cc200_SigKernel_tan_scores, cls=NumpyArrayEncoder))
    
    cc200_PolyKernel_tan_scores, cc200_PolyKernel_tan_Optimal_C = OptimizeRegularization('poly', tanMatrix_cc200, y_cc200)
    print(json.dumps(cc200_PolyKernel_tan_scores, cls=NumpyArrayEncoder))
    
    cc200_LinearKernel_cov_scores, cc200_LinearKernel_cov_Optimal_C = OptimizeRegularization('linear', covMatrix_cc200, y_cc200)
    print(json.dumps(cc200_LinearKernel_cov_scores, cls=NumpyArrayEncoder))
    
    cc200_rbfKernel_cov_scores, cc200_rbfKernel_cov_Optimal_C = OptimizeRegularization('rbf', covMatrix_cc200, y_cc200)
    print(json.dumps(cc200_rbfKernel_cov_scores, cls=NumpyArrayEncoder))
    
    cc200_SigKernel_cov_scores, cc200_SigKernel_cov_Optimal_C = OptimizeRegularization('sigmoid', covMatrix_cc200, y_cc200)
    print(json.dumps(cc200_SigKernel_cov_scores, cls=NumpyArrayEncoder))
    
    cc200_PolyKernel_cov_scores, cc200_PolyKernel_cov_Optimal_C = OptimizeRegularization('poly', covMatrix_cc200, y_cc200)
    print(json.dumps(cc200_PolyKernel_cov_scores, cls=NumpyArrayEncoder))
    
    cc200_LinearKernel_PartCorr_scores, cc200_LinearKernel_PartCorr_Optimal_C = OptimizeRegularization('linear', partCorrMatrix_cc200, y_cc200)
    print(json.dumps(cc200_LinearKernel_PartCorr_scores, cls=NumpyArrayEncoder))
    
    cc200_rbfKernel_PartCorr_scores, cc200_rbfKernel_PartCorr_Optimal_C = OptimizeRegularization('rbf', partCorrMatrix_cc200, y_cc200)
    print(json.dumps(cc200_rbfKernel_PartCorr_scores, cls=NumpyArrayEncoder))
    
    cc200_SigKernel_PartCorr_scores, cc200_SigKernel_PartCorr_Optimal_C = OptimizeRegularization('sigmoid', partCorrMatrix_cc200, y_cc200)
    print(json.dumps(cc200_SigKernel_PartCorr_scores, cls=NumpyArrayEncoder))
    
    cc200_PolyKernel_PartCorr_scores, cc200_PolyKernel_PartCorr_Optimal_C = OptimizeRegularization('poly', partCorrMatrix_cc200, y_cc200)
    print(json.dumps(cc200_PolyKernel_PartCorr_scores, cls=NumpyArrayEncoder))
    
    cc200_LinearKernel_Pre_scores, cc200_LinearKernel_Pre_Optimal_C = OptimizeRegularization('linear', preMatrix_cc200, y_cc200)
    print(json.dumps(cc200_LinearKernel_Pre_scores, cls=NumpyArrayEncoder))
    
    cc200_rbfKernel_Pre_scores, cc200_rbfKernel_Pre_Optimal_C = OptimizeRegularization('rbf', preMatrix_cc200, y_cc200)
    print(json.dumps(cc200_rbfKernel_Pre_scores, cls=NumpyArrayEncoder))
    
    cc200_SigKernel_Pre_scores, cc200_SigKernel_Pre_Optimal_C = OptimizeRegularization('sigmoid', preMatrix_cc200, y_cc200)
    print(json.dumps(cc200_SigKernel_Pre_scores, cls=NumpyArrayEncoder))
    
    cc200_PolyKernel_Pre_scores, cc200_PolyKernel_Pre_Optimal_C = OptimizeRegularization('poly', preMatrix_cc200, y_cc200)
    print(json.dumps(cc200_PolyKernel_Pre_scores, cls=NumpyArrayEncoder))

    print("Just in case I've done something wrong")
    cc200_scores = [cc200_linearKernel_cor_scores, cc200_rbf_kernel_cor_scores, cc200_sigmoidKernel_cor_scores, cc200_polyKernel_cor_scores, 
                  cc200_linearKernel_tan_scores, cc200_rbfKernel_tan_scores, cc200_SigKernel_tan_scores, cc200_PolyKernel_tan_scores,
                 cc200_LinearKernel_cov_scores, cc200_rbfKernel_tan_scores, cc200_SigKernel_cov_scores, cc200_PolyKernel_cov_scores,
                 cc200_LinearKernel_PartCorr_scores, cc200_rbfKernel_PartCorr_scores, cc200_SigKernel_PartCorr_scores, cc200_PolyKernel_PartCorr_scores,
                 cc200_LinearKernel_Pre_scores, cc200_rbfKernel_Pre_scores, cc200_SigKernel_Pre_scores, cc200_PolyKernel_Pre_scores]
    
    Names = ["Linear_Cor", "RBF-Cor", "Sigmoid-Cor", "Poly-Cor",
            "Linear-Tan", "RBF-Tan", "Sigmoid-Tan", "Poly-Tan",
            "Linear-Cov", "RBF-Cov", "Sigmoid-Cov", "Poly-Cov",
            "Linear-PCor", "RBF-PCor", "Sigmoid-PCor", "Poly-PCor",
            "Linear-Pre", "RBF-Pre", "Sigmoid-Pre", "Poly-Pre"]

    for i, j in zip(cc200_scores, Names):
        #json.dump(i, "./Results/SVM/"+"cc200-"+j+".json")
        print(json.dumps(i, cls=NumpyArrayEncoder))
        #np.savetxt("./Results/SVM/"+"cc200-"+j+".csv", delimiter=",")

