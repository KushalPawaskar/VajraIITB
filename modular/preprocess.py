# import os
import argparse

# import xarray as xr
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
# from sklearn.linear_model import LinearRegression, Ridge, OrthogonalMatchingPursuit
# from sklearn.ensemble import ExtraTreesRegressor
# from sklearn.metrics import r2_score

# import xgboost as xgb

# mport torch
# rom torch import nn
# rom torch.utils.data import TensorDataset
# rom torch.utils.data import DataLoader


parser = argparse.ArgumentParser()
parser.add_argument("--forecast_horizon", help="Enter the forecast horizon in minutes (multiples of 15)")
parser.add_argument("--history", help="Enter the observed history to be considered in minutes (multiples of 15)")
parser.add_argument("--pca", help="Enter either the fraction of variance (float: [0.0, 1.0]) or number (int >= 1) of principal components for PCA on GFS features")
args = parser.parse_args()

# aws_df = pd.read_excel("/Users/kpmac/Documents/IITB/DS593/data/AWS/rainfall-2015_2023.xlsx")
# aws_df.fillna(0, inplace=True)
# aws_df.rename(columns={"Index2": "Time"}, inplace=True)
# 
# # Some data is missing in forecasts for forecasts at 00Z of PR in 2015. Thus dropping all the data corresponding to 2015.
# # Also PR analysis data is only from 2019, so ignoring PR analysis data since PW is anyways available.
# aws_df = aws_df[aws_df["Time"].dt.year.isin([year for year in range(2016, 2024)])].reset_index(drop=True)
# 
# # 14 July 2022 shows a lot of missing data based on filename analysis that was done at the end of the notebook, so remove the corresponding data points from aws_normalized (also remove 15 July since these points will require GFS forecasts from 14 July based on current model)
# aws_df = aws_df[~aws_df["Time"].dt.date.isin([pd.to_datetime("2022-07-14").date(), pd.to_datetime("2022-07-15").date()])].reset_index(drop=True)

regions = ['Andheri', 'B ward', 'Bandra', 'Byculla', 'C ward', 'Chembur',
           'Chincholi', 'Colaba', 'D Ward', 'Dahisar', 'Dindoshi', 'F North',
           'F South', 'G South', 'Gowanpada', 'H West ward', 'K East ward',
           'K West ward', 'Kandivali', 'Kurla', 'L ward', 'M West ward', 'MCGM 1',
           'Malvani', 'Marol', 'Memonwada', 'Mulund', 'N ward', 'Nariman Fire',
           'Rawali camp', 'S ward', 'SWD Workshop dadar', 'Thakare natya',
           'Vikhroli', 'Worli', 'vileparle W']

# aws_array = aws_df.values[:, 1:]
# aws_array_path = "/Users/kpmac/Documents/IITB/DS593/data/preprocessed/aws_array.npy"
# np.save(aws_array_path, aws_array)

aws_array_path = "/Users/kpmac/Documents/IITB/DS593/data/preprocessed/aws_array.npy"
aws_array = np.load(aws_array_path, allow_pickle=True)

pred_times = [pd.to_datetime("09:30:00").time(), pd.to_datetime("15:30:00").time(), pd.to_datetime("21:30:00").time(), pd.to_datetime("03:30:00").time()]

gfs_features_path = "/Users/kpmac/Documents/IITB/DS593/data/preprocessed/gfs_features_array.npy"
gfs_features = np.load(gfs_features_path)
gfs_features = gfs_features[1:]

gfs_features = gfs_features[:, :1296]

encoding_array_path = "/Users/kpmac/Documents/IITB/DS593/data/preprocessed/time_day_runtime_gfstime_encoding_array.npy"
encoding_array = np.load(encoding_array_path)

print("AWS array shape:", aws_array.shape)
print("Encoding array shape:", encoding_array.shape)
print("GFS features shape:", gfs_features.shape)
print()

forecast_horizon = int(args.forecast_horizon)    # int(input("Enter the forecast horizon in minutes (multiples of 15): "))
history = int(args.history)                      # int(input("Enter the observed history to be considered in minutes (multiples of 15): "))

forecast_horizon = forecast_horizon // 15
history = history // 15

aws_array_train = aws_array[:46848]

# Standardizing the aws data based on training part
aws_scaler = MinMaxScaler()
aws_scaler.fit(aws_array_train)

aws_standardized = aws_scaler.transform(aws_array)

aws_features = []
encoding_features = []
y = []

day_after_missing_indices = [11712, 23424, 35136, 46848, 58560, 70272, 74400, 81792]
indices_to_be_adjusted = []
for i in day_after_missing_indices:
    for j in range(i - forecast_horizon + 1, i + 1):
        indices_to_be_adjusted.append(j)
    for j in range(i + 1, i + history):
        indices_to_be_adjusted.append(j)

for idx in range(1 + history, aws_standardized.shape[0]):        # 1 + history because we are going to remove aws_array[0] and considered history*15 minutes of observed history
    if idx not in indices_to_be_adjusted:
        aws_features.append(aws_standardized[idx - history:idx])
        encoding_features.append(encoding_array[idx - history:idx])
        y.append(aws_standardized[idx])

aws_features = np.array(aws_features)
encoding_features = np.array(encoding_features)
y = np.array(y)

for i in range(len(indices_to_be_adjusted)):
    gfs_features = np.delete(gfs_features, obj=indices_to_be_adjusted[i] - i, axis=0)

before_46848 = 0
before_70272 = 0
for i in range(len(indices_to_be_adjusted)):
    if indices_to_be_adjusted[i] < 46848:
        before_46848 = i + 1
    if indices_to_be_adjusted[i] < 70272:
        before_70272 = i + 1

l1 = 46848 - history - 1 - before_46848    # 46848-history-1 because we are going to remove aws_array[0] and consider history*15 minutes of observed history
l2 = 70272 - history - 1 - before_70272

aws_train = aws_features[:l1]
aws_val = aws_features[l1:l2]
aws_test = aws_features[l2:]

encoding_train = encoding_features[:l1]
encoding_val = encoding_features[l1:l2]
encoding_test = encoding_features[l2:]

gfs_train = gfs_features[max(history - 4, 0):l1 + max(history - 4, 0)]      # Adjustment factor required because for loop started from (history=4) + 1 while making the gfs_features list
gfs_val = gfs_features[l1 + max(history - 4, 0):l2 + max(history - 4, 0)]
# gfs_test = gfs_features[l2 + max(history - 4, 0):]
if forecast_horizon == 1:
    gfs_test = gfs_features[l2 + max(history - 4, 0):]
else:
    gfs_test = gfs_features[l2 + max(history - 4, 0):-(forecast_horizon - 1)]   # -(forecast_horizon - 1) to match aws_test size

y_train = y[:l1]
y_val = y[l1:l2]
y_test = y[l2:]

# Standardizing the gfs features based on training part
gfs_scaler = MinMaxScaler()
gfs_scaler.fit(gfs_train)

gfs_train = gfs_scaler.transform(gfs_train)
gfs_val = gfs_scaler.transform(gfs_val)
gfs_test = gfs_scaler.transform(gfs_test)

print("AWS train shape:", aws_train.shape)
print("Encoding train shape:", encoding_train.shape)
print("GFS train shape:", gfs_train.shape)
print("Target train shape", y_train.shape)
print()

print("AWS val shape:", aws_val.shape)
print("Encoding val shape:", encoding_val.shape)
print("GFS val shape:", gfs_val.shape)
print("Target val shape", y_val.shape)
print()

print("AWS test shape:", aws_test.shape)
print("Encoding test shape:", encoding_test.shape)
print("GFS test shape:", gfs_test.shape)
print("Target test shape", y_test.shape)
print()

var_comp = args.pca     # input("Enter either the fraction of variance (float: [0.0, 1.0]) or number (int >= 1) of principal components for PCA on GFS features: ")
# print()
if "." in var_comp:
    var_comp = float(var_comp)
else:
    var_comp = int(var_comp)

pca = PCA(var_comp)
pca.fit(gfs_train)
print("Number of principal components:", pca.n_components_)
print("Fraction of variance captured by principal components:", pca.explained_variance_ratio_.sum())
print()

gfs_train_reduced = pca.transform(gfs_train)
gfs_val_reduced = pca.transform(gfs_val)
gfs_test_reduced = pca.transform(gfs_test)

print("GFS train reduced shape", gfs_train_reduced.shape)
print("GFS val reduced shape", gfs_val_reduced.shape)
print("GFS test reduced shape", gfs_test_reduced.shape)
