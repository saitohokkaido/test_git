import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# import mglearn
from IPython.display import display
# import os
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from scipy import signal
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import time
import datetime

# 実行時間の計測
start_time = datetime.datetime.today()

# FIVの生データの読み込み
FIV_dframe = pd.read_excel("raw_true_force_with_runnumbers_labels.xlsx", sheet_name="Sheet1")

# データとラベルに分解
FIV_df_data = FIV_dframe[0:60000]
FIV_df_labels = pd.DataFrame(FIV_dframe.loc[60000])
FIV_df_labels = FIV_df_labels.rename(columns={60000:"FlowRegime"})

# データをオーバーラップなしの3秒ごとに分割して、ラベルを付加
FIV_dframe_split = [""]*20
FIV_dframe_split_unknown = [""]*20
for i in range(20):
    FIV_dframe_split[i] = FIV_df_data[i*3000:((i+1)*3000)].reset_index(drop=True)
    FIV_dframe_split[i] = pd.concat([FIV_dframe_split[i],pd.DataFrame(FIV_df_labels).T])
    FIV_dframe_split_unknown[i] = FIV_dframe_split[i].drop(FIV_dframe_split[i].columns[list(FIV_dframe_split[i].loc["FlowRegime"] != "Unknown")],axis=1).T
    FIV_dframe_split[i] = FIV_dframe_split[i].drop(FIV_dframe_split[i].columns[list(FIV_dframe_split[i].loc["FlowRegime"] == "Unknown")],axis=1).T
    FIV_dframe_split[i]["TargetLabel"] = FIV_dframe_split[i]["FlowRegime"].map({"Bubbly":0,"Slug":1,"Churn":2,"Annular":3})

# (33*20)行(3000+2)列に連結させて、excelに出力。その後、データとラベルに分ける
splited_FIV_list = []
splited_FIV_list_unknown = []
for i in range(20):
    splited_FIV_list.append(FIV_dframe_split[i])
    splited_FIV_list_unknown.append(FIV_dframe_split_unknown[i])
splited_FIV_df = pd.concat(splited_FIV_list)
splited_FIV_df_unknown = pd.concat(splited_FIV_list_unknown)
# splited_FIV_df.to_excel("FIV_with_labels_20bunkatu.xlsx")
splited_FIV_df_data = splited_FIV_df.drop(["FlowRegime","TargetLabel"],axis=1)
splited_FIV_df_target = pd.DataFrame(splited_FIV_df["TargetLabel"])
splited_FIV_df_data_unknown = splited_FIV_df_unknown.drop(["FlowRegime"],axis=1)

# 振動の生データを標準化(平均を0,分散を1に変換)して、その後に特徴量を抽出する
mean_on_raw_data = pd.DataFrame(splited_FIV_df_data.mean(axis=1))
std_on_raw_data = pd.DataFrame(splited_FIV_df_data.std(axis=1))
for i in range(3000):
    mean_on_raw_data[i] = mean_on_raw_data[0]
    std_on_raw_data[i] = std_on_raw_data[0]
splited_FIV_df_data_standardizationed = (splited_FIV_df_data - mean_on_raw_data) / std_on_raw_data

# welch's methodを使って、PSDを計算。その後、特徴量を抽出
fs = 1000
freq = [""]*1320
P = [""]*1320
sum_psd_selected_range = []
max_psd_selected_range = []
var_psd_selected_range = []
std_psd_selected_range = []
skew_psd_selected_range = []
kurt_psd_selected_range = []
peak5th_psd_selected_range = []
for i in range(1320):
    freq[i], P[i] = signal.welch(np.ravel(splited_FIV_df_data_standardizationed.iloc[i]), fs, nperseg=3000/6)
    maxID = signal.argrelmax(10*np.log10(list(P[i])[0:100]).real)
    sum_psd_selected_range.append(np.sum((10*np.log10(list(P[i])[0:100]).real)))
    max_psd_selected_range.append(np.amax(10*np.log10(list(P[i])[0:100]).real))
    var_psd_selected_range.append(np.var(10*np.log10(list(P[i])[0:100]).real))
    std_psd_selected_range.append(np.std(10*np.log10(list(P[i])[0:100]).real))
    skew_psd_selected_range.append(pd.DataFrame(10*np.log10(list(P[i])[0:100]).real).skew(axis=0))
    kurt_psd_selected_range.append(pd.DataFrame(10*np.log10(list(P[i])[0:100]).real).kurt(axis=0))
    peak5th_psd_selected_range.append(np.mean(sorted(list((10*np.log10(list(P[i])).real[np.ravel(maxID)])), reverse=True)[0:3]))

# 標準化されたデータから時間領域と周波数領域の特徴量を抽出。その後、excelに出力
splited_FIV_df_features = pd.DataFrame()
# splited_FIV_df_8features["mean"] = np.mean(splited_FIV_df_data_standardizationed, axis=1)
splited_FIV_df_features["max"] = np.amax(splited_FIV_df_data_standardizationed, axis=1)
splited_FIV_df_features["min"] = np.amin(splited_FIV_df_data_standardizationed, axis=1)
splited_FIV_df_features["median"] = splited_FIV_df_data_standardizationed.median(axis=1)
# splited_FIV_df_8features["variance"] = np.var(splited_FIV_df_data_standardizationed, axis=1)
# splited_FIV_df_8features["standard deviation"] = np.std(splited_FIV_df_data_standardizationed, axis=1)
splited_FIV_df_features["skewness"] = splited_FIV_df_data_standardizationed.skew(axis=1)
splited_FIV_df_features["kurtosis"] = splited_FIV_df_data_standardizationed.kurt(axis=1)
splited_FIV_df_features["sum of PSD"] = np.array(sum_psd_selected_range)
splited_FIV_df_features["max of PSD"] = np.array(max_psd_selected_range)
splited_FIV_df_features["var of PSD"] = np.array(var_psd_selected_range)
splited_FIV_df_features["std of PSD"] = np.array(std_psd_selected_range)
splited_FIV_df_features["skew of PSD"] = np.array(skew_psd_selected_range)
splited_FIV_df_features["kurt of PSD"] = np.array(kurt_psd_selected_range)
splited_FIV_df_features["5 peaks of PSD"] = np.array(peak5th_psd_selected_range)
splited_FIV_df_features_with_labels = pd.concat([splited_FIV_df_features,splited_FIV_df["FlowRegime"]],axis=1)
splited_FIV_df_features_with_labels.to_excel("FIV_with_labels_20bunkatu_features_standardizationed.xlsx")

# ランダムフォレストにて、特徴量データを識別してみる
X_train, X_test, y_train, y_test = train_test_split(splited_FIV_df_features, splited_FIV_df_target, stratify=splited_FIV_df_target, random_state=1)

# ランダムフォレストにおけるグリッドサーチと交差検定を用いた最適化パラメータ推定
rf_pipe = Pipeline([("pca",PCA()),("rf",RandomForestClassifier(n_estimators=100))])
parameters = {"pca__n_components":range(10,13),"rf__max_depth":[7,8,9,10,11]}
rf_grid = GridSearchCV(rf_pipe, parameters, cv=10)
rf_grid.fit(X_train,np.ravel(y_train))
predicted_rf = rf_grid.predict(X_test)

print("**************\n")
print("RandomForest (selected features)\n")
print("estimated best parameters:\n{}\n".format(rf_grid.best_params_))
print("accuracy on train:\n{:.3f}\n".format(rf_grid.score(X_train,y_train)))
print("accuracy on test:\n{:.3f}\n".format(rf_grid.score(X_test,y_test)))
print("classification report\n{}\n".format(classification_report(y_test,predicted_rf)))
print("confusion matrix\n{}\n".format(confusion_matrix(y_test,predicted_rf)))
print("**************\n")

# サポートベクターマシンにおけるグリッドサーチと交差検定を用いた最適化パラメータ推定
svm_pipe = Pipeline([("pca",PCA()), ("scaler",MinMaxScaler()),("svm",SVC())])
parameters = {"pca__n_components":range(10,13),
             "svm__gamma":[0.001,0.01,0.1,1,10],
             "svm__C":[0.001,0.01,0.1,1,10,100,1000,10000]}
svm_grid = GridSearchCV(svm_pipe, parameters, cv=10)
svm_grid.fit(X_train,np.ravel(y_train))
predicted_svm = svm_grid.predict(X_test)

print("**************\n")
print("SupportVectorMachine (selected features)\n")
print("estimated best parameters:\n{}\n".format(svm_grid.best_params_))
print("accuracy on train:\n{:.3f}\n".format(svm_grid.score(X_train,y_train)))
print("accuracy on test:\n{:.3f}\n".format(svm_grid.score(X_test,y_test)))
print("classification report\n{}\n".format(classification_report(y_test,predicted_svm)))
print("confusion matrix\n{}\n".format(confusion_matrix(y_test,predicted_svm)))
print("**************\n")

# マルチレイヤーパーセプトロンにおけるグリッドサーチと交差検定を用いた最適化パラメータ推定
mlp_pipe = Pipeline([("pca",PCA()), ("scaler",StandardScaler()),("mlp",MLPClassifier(solver="lbfgs"))])
parameters = {"pca__n_components":range(10,13),
             "mlp__alpha":[0.1,1,10],
             "mlp__hidden_layer_sizes":[[10],[15],[5,5],[10,10],[50,50]]}
mlp_grid = GridSearchCV(mlp_pipe, parameters, cv=10)
mlp_grid.fit(X_train,np.ravel(y_train))
predicted_mlp = mlp_grid.predict(X_test)

print("**************\n")
print("NeuralNetwork (selected features)\n")
print("estimated best parameters:\n{}\n".format(mlp_grid.best_params_))
print("accuracy on train:\n{:.3f}\n".format(mlp_grid.score(X_train,y_train)))
print("accuracy on test:\n{:.3f}\n".format(mlp_grid.score(X_test,y_test)))
print("classification report\n{}\n".format(classification_report(y_test,predicted_mlp)))
print("confusion matrix\n{}\n".format(confusion_matrix(y_test,predicted_mlp)))
print("**************\n")

# 実行時間の計測
end_time = datetime.datetime.today()
time_delta = end_time - start_time
print("Calculation time: {}".format(time_delta))
