import datetime
import json
import os
import pickle

import numpy as np
import pandas as pd
import shap
from matplotlib import pyplot as plt
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_curve, roc_auc_score, recall_score, confusion_matrix, matthews_corrcoef
from sklearn.svm import SVC
from lightgbm import LGBMClassifier
from utils.utils import folds_split, mean_std
import xgboost as xgb
from representation import list2rep


def data_load(csv):
    data = pd.read_csv(csv)
    features_raw = data['SMILES'].tolist()
    label_raw = data['class'].tolist()

    smiles_list = []
    label_list = []

    for i in range(len(features_raw)):
        smiles = features_raw[i]
        label = label_raw[i]
        mol = Chem.MolFromSmiles(smiles)

        if mol is None:
            continue
        else:
            smiles = Chem.MolToSmiles(mol)
            smiles_list.append(smiles)
            label_list.append(label)
    return smiles_list, label_list


def architecture(X_train, Y_train, X_tesy, model_type, save=False, **kwargs):
    kwargs["model_type"] = model_type
    if model_type == 'RF':
        model = RandomForestClassifier(n_estimators=kwargs["n_estimators"], criterion=kwargs["criterion"],
                                       max_features=kwargs["max_features"], bootstrap=kwargs["bootstrap"],
                                       oob_score=kwargs["oob_score"], warm_start=kwargs["warm_start"],
                                       min_samples_leaf=kwargs["min_samples_leaf"])
        model.fit(X_train, Y_train)
        y_pred = model.predict(X_tesy)
        y_scores = model.predict_proba(X_tesy)[:, 1]
    elif model_type == 'SVM':
        model = SVC(probability=True)
        model.fit(X_train, Y_train)
        y_pred = model.predict(X_tesy)
        y_scores = model.predict_proba(X_tesy)[:, 1]
    elif model_type == 'LGB':
        model = LGBMClassifier()
        model.fit(X_train, Y_train)
        y_pred = model.predict(X_tesy)
        y_scores = model.predict_proba(X_tesy)[:, 1]
    elif model_type == 'XGB':
        model = xgb.XGBClassifier()
        model.fit(X_train, Y_train)
        y_pred = model.predict(X_tesy)
        y_scores = model.predict_proba(X_tesy)[:, 1]
    else:
        raise ValueError(f'Unknown model_type: {model_type}')

    time_now = str(datetime.datetime.now())
    time_now = time_now.replace(" ", "_")
    time_now = time_now.replace(":", "-")
    if save:
        pkl = os.path.join(kwargs["path"], time_now + ".pkl")
        config = os.path.join(kwargs["path"], time_now + ".config")
        with open(pkl, 'wb') as file:
            pickle.dump(model, file)
        with open(config, 'wb') as f:
            pickle.dump(kwargs, f)
    return y_pred, y_scores, model


def metrics_calc(test, pred, score):
    confusion = confusion_matrix(test, pred)
    TP = confusion[1, 1]
    TN = confusion[0, 0]
    FP = confusion[0, 1]
    FN = confusion[1, 0]
    Accuracy = (TP + TN) / float(TP + TN + FP + FN)
    Sensitivity = TP / float(TP + FN)
    Specificity = TN / float(TN + FP)

    # Matthews correlation coefficient
    MCC = matthews_corrcoef(test, pred)

    # AUC
    auc = roc_auc_score(test, score)

    # recall_score
    RE = recall_score(test, pred)

    return Accuracy, Sensitivity, Specificity, MCC, auc, RE


def model_train(fpMtx, label, model_type, save, **kwargs):
    X_train, X_valid, y_train, y_valid = train_test_split(fpMtx, label, train_size=0.8, random_state=42)

    folds = folds_split(X_train, y_train)
    Accuracy_list, Sensitivity_list, Specificity_list, MCC_list, auc_list, RE_list = [], [], [], [], [], []
    for fold in folds:
        X_train_fold, y_train_fold, X_test_fold, y_test_fold = fold
        y_pred_fold, y_scores_fold, _ = architecture(X_train_fold, y_train_fold, X_test_fold, model_type,
                                                     **kwargs)  
        Accuracy, Sensitivity, Specificity, MCC, auc, RE = metrics_calc(y_test_fold, y_pred_fold, y_scores_fold)
        Accuracy_list.append(Accuracy)
        Sensitivity_list.append(Sensitivity)
        Specificity_list.append(Specificity)
        MCC_list.append(MCC)
        auc_list.append(auc)
        RE_list.append(RE)

    Accuracy_fold = mean_std(Accuracy_list)
    Sensitivity_fold = mean_std(Sensitivity_list)
    Specificity_fold = mean_std(Specificity_list)
    MCC_fold = mean_std(MCC_list)
    auc_fold = mean_std(auc_list)
    RE_fold = mean_std(RE_list)
    print("Accuracy_fold:", Accuracy_fold,
          "Sensitivity_fold:", Sensitivity_fold,
          "Specificity_fold:", Specificity_fold,
          "MCC_fold:", MCC_fold,
          "auc_fold:", auc_fold,
          "RE_fold:", RE_fold)
    y_pred, y_scores, model = architecture(X_train, y_train, X_valid, model_type, save, **kwargs)  
    Accuracy_valid, Sensitivity_valid, Specificity_valid, MCC_valid, auc_valid, RE_valid = metrics_calc(y_valid, y_pred,
                                                                                                        y_scores)
    print("auc_valid:", ('{:.3f}'.format(auc_valid)),
          "Accuracy_valid:", ('{:.3f}'.format(Accuracy_valid)),
          "Sensitivity_valid:", ('{:.3f}'.format(Sensitivity_valid)),
          "Specificity_valid:", ('{:.3f}'.format(Specificity_valid)),
          "MCC_valid:", ('{:.3f}'.format(MCC_valid)),
          "RE_valid:", ('{:.3f}'.format(RE_valid)))
    '''
    AUC_fold = '{:.3f}'.format(auc_fold)
    Accuracy_fold = '{:.3f}'.format(Accuracy_fold)
    Sensitivity_fold = '{:.3f}'.format(Sensitivity_fold)
    Specificity_fold = '{:.3f}'.format(Specificity_fold)
    MCC_fold = '{:.3f}'.format(MCC_fold)
    RE_fold = '{:.3f}'.format(RE_fold)
    '''
    AUC_valid = '{:.3f}'.format(auc_valid)
    Accuracy_valid = '{:.3f}'.format(Accuracy_valid)
    Sensitivity_valid = '{:.3f}'.format(Sensitivity_valid)
    Specificity_valid = '{:.3f}'.format(Specificity_valid)
    MCC_valid = '{:.3f}'.format(MCC_valid)
    RE_valid = '{:.3f}'.format(RE_valid)

    return model, auc_fold, Accuracy_fold, Sensitivity_fold, Specificity_fold, MCC_fold, RE_fold, AUC_valid, Accuracy_valid, Sensitivity_valid, Specificity_valid, MCC_valid, RE_valid


def shap_test(model, X, y):
    explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(X)  

    descs = [desc_name[0] for desc_name in Descriptors._descList[:]]  
    del descs[42]  
    del descs[13]  
    del descs[11]  
    '''

    shap.force_plot(explainer.expected_value, shap_values[0, :], X[15, :], matplotlib=True,feature_names=descs)
    shap.force_plot(explainer.expected_value, shap_values[0, :], X[115, :], matplotlib=True, feature_names=descs)
    shap.force_plot(explainer.expected_value, shap_values[0, :], X[145, :], matplotlib=True, feature_names=descs)
    shap.force_plot(explainer.expected_value, shap_values[0, :], X[235, :], matplotlib=True, feature_names=descs)
    shap.force_plot(explainer.expected_value, shap_values[0, :], X[1345, :], matplotlib=True,feature_names=descs)
    shap.force_plot(explainer.expected_value, shap_values[0, :], X[1325, :], matplotlib=True, feature_names=descs)
    shap.force_plot(explainer.expected_value, shap_values[0, :], X[1827, :], matplotlib=True, feature_names=descs)
    shap.force_plot(explainer.expected_value, shap_values[0, :], X[375, :], matplotlib=True, feature_names=descs)
    '''

    shap.summary_plot(shap_values, X, feature_names=descs, max_display=10)

    shap.summary_plot(shap_values, X, plot_type="bar", feature_names=descs, max_display=10)


def main(csv, kwargs):
    smiles_list, label = data_load(csv)
    x_list = list2rep(smiles_list, kwargs["repr"])
    (model, auc_fold, Accuracy_fold, Sensitivity_fold, Specificity_fold, MCC_fold, RE_fold, auc_valid, Accuracy_valid,
     Sensitivity_valid, Specificity_valid, MCC_valid, RE_valid) = model_train(x_list, label, kwargs)
    shap_test(model,x_list,label)
    print("↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑", "dataset:", csv, "represetation:", kwargs["repr"], "model:",
          kwargs["model_type"])


def WebUI_RF(csv, repr="rdkit_FP", criterion="gini", max_features="auto",
             n_estimators=100, bootstrap=True, oob_score=False,
             warm_start=False, min_samples_leaf=1, path="./"):
    smiles_list, label = data_load(csv)
    x_list = list2rep(smiles_list, repr)
    (_, auc_fold, Accuracy_fold, Sensitivity_fold, Specificity_fold, MCC_fold, RE_fold, auc_valid, Accuracy_valid,
     Sensitivity_valid, Specificity_valid, MCC_valid, RE_valid) = model_train(x_list, label, model_type="RF", save=True, repr=repr,
                                    criterion=criterion, max_features=max_features,
                                    n_estimators=n_estimators, bootstrap=bootstrap, oob_score=oob_score,
                                    warm_start=warm_start, min_samples_leaf=min_samples_leaf, path=path)
    return auc_fold, Accuracy_fold, Sensitivity_fold, Specificity_fold, MCC_fold, RE_fold, auc_valid, Accuracy_valid, Sensitivity_valid, Specificity_valid, MCC_valid, RE_valid


if __name__ == "__main__":
    # ml:RF SVM LGB XGB
    def run(ml):
        main("./dataset/eye.csv", 'rdkit_descriptor', ml)
        main("./dataset/eye.csv", 'ECFP', ml)
        main("./dataset/eye.csv", 'rdkit_FP', ml)
        main("./dataset/eye.csv", 'MACCS_FP', ml)
        main("./dataset/eye.csv", 'PubChem', ml)
        main("./dataset/eye.csv", 'CDKextended', ml)
        main("./dataset/eye.csv", 'KlekotaRoth', ml)


    run("RF")
