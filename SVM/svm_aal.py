#!/usr/bin/env python3

import warnings
warnings.filterwarnings("ignore")
from nilearn.datasets import fetch_abide_pcp
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score, recall_score
# Linear SVM
from sklearn.svm import SVC
#from roc import plotROC
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
        if isinstance(obj, numpy.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)

#Function that trains and test a SVC model for each combination of parameters as laid out
#Returns the scores with the highest AUC and the parameters of the model with the best AUC
@jit(target_backend = 'cpu')
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
    'Balanced Accuracy': make_scorer(balanced_accuracy_score)
}
#Function that trains the SVC models and returns a set of scores as defined above
def Train_Test_Model(model, X_data, labels, seed = 0, num_folds = 10):
    skf = RepeatedStratifiedKFold(n_splits = num_folds, n_repeats = 5, random_state = seed)
    scores = cross_validate(model, X_data, labels, scoring = scoring, cv = skf, return_train_score = True, verbose = 0, n_jobs = 4)
    return scores


if __name__ == '__main__':
    abide_aal = fetch_abide_pcp(derivatives = ['rois_aal'], pipeline = 'cpac', quality_checked = True, verbose = 0)

    
    
    y_aal = abide_aal.phenotypic['DX_GROUP']
    y_aal[y_aal == 2] = 0

    corrMeasure = ConnectivityMeasure(kind = "correlation", vectorize = True)
    tanMeasure = ConnectivityMeasure(kind = "tangent", vectorize = True)
    covMeasure = ConnectivityMeasure(kind = 'covariance', vectorize = True)
    partCorrMeasure = ConnectivityMeasure(kind = "partial correlation", vectorize = True)
    preMeasure = ConnectivityMeasure(kind = "precision", vectorize = True)

    #print("importing data")
    corMatrix_aal = corrMeasure.fit_transform(abide_aal.rois_aal)
    #print(corMatrix_aal.shape)
    tanMatrix_aal = tanMeasure.fit_transform(abide_aal.rois_aal)
    #print(tanMatrix_aal.shape)
    covMatrix_aal = covMeasure.fit_transform(abide_aal.rois_aal)
    #print(covMatrix_aal.shape)
    partCorrMatrix_aal = partCorrMeasure.fit_transform(abide_aal.rois_aal)
    #print(partCorrMatrix_aal.shape)
    preMatrix_aal = preMeasure.fit_transform(abide_aal.rois_aal)
    #print(preMatrix_aal.shape)
    
    #print("Running tests")
    aal_linearKernel_cor_scores, aal_linearKernel_corParams  = OptimizeRegularization('linear', corMatrix_aal, y_aal)
    aal_rbf_kernel_cor_scores, aal_rbf_kernel_cor_Optimal_params = OptimizeRegularization('rbf', corMatrix_aal, y_aal)
    aal_sigmoidKernel_cor_scores, aal_sigmoidKernel_cor_Optimal_C = OptimizeRegularization('sigmoid', corMatrix_aal, y_aal)
    aal_polyKernel_cor_scores, aal_polyKernel_cor_Optimal_C = OptimizeRegularization('poly', corMatrix_aal, y_aal)
    aal_linearKernel_tan_scores, aal_linearKernel_tan_Optimal_C = OptimizeRegularization('linear', tanMatrix_aal, y_aal)
    aal_rbfKernel_tan_scores, aal_rbfKernel_tan_Optimal_C = OptimizeRegularization('rbf', tanMatrix_aal, y_aal)
    aal_SigKernel_tan_scores, aal_SigKernel_tan_Optimal_C = OptimizeRegularization('sigmoid', tanMatrix_aal, y_aal)
    aal_PolyKernel_tan_scores, aal_PolyKernel_tan_Optimal_C = OptimizeRegularization('poly', tanMatrix_aal, y_aal)
    aal_LinearKernel_cov_scores, aal_LinearKernel_cov_Optimal_C = OptimizeRegularization('linear', covMatrix_aal, y_aal)
    aal_rbfKernel_cov_scores, aal_rbfKernel_cov_Optimal_C = OptimizeRegularization('rbf', covMatrix_aal, y_aal)
    aal_SigKernel_cov_scores, aal_SigKernel_cov_Optimal_C = OptimizeRegularization('sigmoid', covMatrix_aal, y_aal)
    aal_PolyKernel_cov_scores, aal_PolyKernel_cov_Optimal_C = OptimizeRegularization('poly', covMatrix_aal, y_aal)
    aal_LinearKernel_PartCorr_scores, aal_LinearKernel_PartCorr_Optimal_C = OptimizeRegularization('linear', partCorrMatrix_aal, y_aal)
    aal_rbfKernel_PartCorr_scores, aal_rbfKernel_PartCorr_Optimal_C = OptimizeRegularization('rbf', partCorrMatrix_aal, y_aal)
    aal_SigKernel_PartCorr_scores, aal_SigKernel_PartCorr_Optimal_C = OptimizeRegularization('sigmoid', partCorrMatrix_aal, y_aal)
    aal_PolyKernel_PartCorr_scores, aal_PolyKernel_PartCorr_Optimal_C = OptimizeRegularization('poly', partCorrMatrix_aal, y_aal)
    aal_LinearKernel_Pre_scores, aal_LinearKernel_Pre_Optimal_C = OptimizeRegularization('linear', preMatrix_aal, y_aal)
    aal_rbfKernel_Pre_scores, aal_rbfKernel_Pre_Optimal_C = OptimizeRegularization('rbf', preMatrix_aal, y_aal)
    aal_SigKernel_Pre_scores, aal_SigKernel_Pre_Optimal_C = OptimizeRegularization('sigmoid', preMatrix_aal, y_aal)
    aal_PolyKernel_Pre_scores, aal_PolyKernel_Pre_Optimal_C = OptimizeRegularization('poly', preMatrix_aal, y_aal)

    aal_scores = [aal_linearKernel_cor_scores, aal_rbf_kernel_cor_scores, aal_sigmoidKernel_cor_scores, aal_polyKernel_cor_scores, 
                  aal_linearKernel_tan_scores, aal_rbfKernel_tan_scores, aal_SigKernel_tan_scores, aal_PolyKernel_tan_scores,
                 aal_LinearKernel_cov_scores, aal_rbfKernel_tan_scores, aal_SigKernel_cov_scores, aal_PolyKernel_cov_scores,
                 aal_LinearKernel_PartCorr_scores, aal_rbfKernel_PartCorr_scores, aal_SigKernel_PartCorr_scores, aal_PolyKernel_PartCorr_scores,
                 aal_LinearKernel_Pre_scores, aal_rbfKernel_Pre_scores, aal_SigKernel_Pre_scores, aal_PolyKernel_Pre_scores]
    
    Names = ["Linear-Cor", "RBF-Cor", "Sigmoid-Cor", "Poly-Cor",
            "Linear-Tan", "RBF-Tan", "Sigmoid-Tan", "Poly-Tan",
            "Linear-Cov", "RBF-Cov", "Sigmoid-Cov", "Poly-Cov",
            "Linear-PCor", "RBF-PCor", "Sigmoid-PCor", "Poly-PCor",
            "Linear-Pre", "RBF-Pre", "Sigmoid-Pre", "Poly-Pre"]

    for i, j in zip(aal_scores, Names):
        #json.dump(i, "./Results/SVM/"+"aal-"+j+".json")
        print(json.dumps(i, cls=NumpyArrayEncoder))


