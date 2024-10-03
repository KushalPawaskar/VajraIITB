# import os
import argparse

# import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, Ridge, OrthogonalMatchingPursuit
# from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import r2_score

# import xgboost as xgb

import torch
from torch import nn
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader


SEED = 0
np.random.seed(SEED)
torch.manual_seed(SEED)

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

forecast_horizon = int(args.forecast_horizon)    # int(input("Enter the forecast horizon in minutes (multiples of 15): "))
history = int(args.history)                      # int(input("Enter the observed history to be considered in minutes (multiples of 15): "))
# print()

forecast_horizon = forecast_horizon // 15
history = history // 15

aws_array_train = aws_array[:46848]

# Standardizing the aws data based on training part
aws_scaler = MinMaxScaler()
aws_scaler.fit(aws_array_train)

aws_standardized = aws_scaler.transform(aws_array)

aws_features = []
y = []

day_after_missing_indices = [11712, 23424, 35136, 46848, 58560, 70272, 74400, 81792]
indices_to_be_adjusted = []
for i in day_after_missing_indices:
    for j in range(i - forecast_horizon + 1, i + 1):
        indices_to_be_adjusted.append(j)
    for j in range(i + 1, i + history):
        indices_to_be_adjusted.append(j)

print(len(indices_to_be_adjusted))
print(indices_to_be_adjusted)

for idx in range(1 + history, aws_standardized.shape[0]):        # 1 + history because we are going to remove aws_array[0] and considered history*15 minutes of observed history
    if idx not in indices_to_be_adjusted:
        aws_features.append(aws_standardized[idx - history:idx])
        y.append(aws_standardized[idx])

aws_features = np.array(aws_features)
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

print(before_46848)
print(before_70272)

l1 = 46848 - history - 1 - before_46848    # 46848-history-1 because we are going to remove aws_array[0] and consider history*15 minutes of observed history
l2 = 70272 - history - 1 - before_70272

aws_train = aws_features[:l1]
aws_val = aws_features[l1:l2]
aws_test = aws_features[l2:]

gfs_train = gfs_features[max(history - 4, 0):l1 + max(history - 4, 0)]      # Adjustment factor required because for loop started from (history=4) + 1 while making the gfs_features list
gfs_val = gfs_features[l1 + max(history - 4, 0):l2 + max(history - 4, 0)]
# gfs_test = gfs_features[l2 + max(history - 4, 0):]
if forecast_horizon == 1:
    gfs_test = gfs_features[l2 + max(history - 4, 0):]
else:
    gfs_test = gfs_features[l2 + max(history - 4, 0):-(forecast_horizon - 1)]   # -(forecast_horizon - 1) to match aws_test size
print(gfs_test)

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
print("GFS train shape:", gfs_train.shape)
print("Target train shape", y_train.shape)
print()

print("AWS val shape:", aws_val.shape)
print("GFS val shape:", gfs_val.shape)
print("Target val shape", y_val.shape)
print()

print("AWS test shape:", aws_test.shape)
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

# lin_reg_list = []
# for i in range(y.shape[1]):
#     X_train = np.concatenate((aws_train[:, :, i], gfs_train_reduced), axis=1)
#     X_val = np.concatenate((aws_val[:, :, i], gfs_val_reduced), axis=1)
#     reg = LinearRegression().fit(X_train, y_train[:, i])
#     print(regions[i], "training score:", reg.score(X_train, y_train[:, i]), "validation score:", reg.score(X_val, y_val[:, i]))
#     lin_reg_list.append(reg)
# print()

# ridge_reg_list = []
# for i in range(y.shape[1]):
#     X_train = np.concatenate((aws_train[:, :, i], gfs_train_reduced), axis=1)
#     X_val = np.concatenate((aws_val[:, :, i], gfs_val_reduced), axis=1)
#     reg = Ridge(alpha=0.5).fit(X_train, y_train[:, i])        # increasing alpha makes score worse
#     print(regions[i], "training score:", reg.score(X_train, y_train[:, i]), "validation score:", reg.score(X_val, y_val[:, i]))
#     ridge_reg_list.append(reg)
# print()

# orthogonal_matching_pursuit_reg_list = []
# for i in range(y.shape[1]):
#     X_train = np.concatenate((aws_train[:, :, i], gfs_train_reduced), axis=1)
#     X_val = np.concatenate((aws_val[:, :, i], gfs_val_reduced), axis=1)
#     reg = OrthogonalMatchingPursuit(n_nonzero_coefs=X_train.shape[1]).fit(X_train, y_train[:, i])        # reducing n_nonzer_coefs makes scores worse
#     print(regions[i], "training score:", reg.score(X_train, y_train[:, i]), "validation score:", reg.score(X_val, y_val[:, i]))
#     orthogonal_matching_pursuit_reg_list.append(reg)
# print()

# extra_trees_reg_list = []
# for i in range(y.shape[1]):
#     X_train = np.concatenate((aws_train[:, :, i], gfs_train_reduced), axis=1)
#     X_val = np.concatenate((aws_val[:, :, i], gfs_val_reduced), axis=1)
#     reg = ExtraTreesRegressor(n_estimators=100).fit(X_train, y_train[:, i])
#     print(regions[i], "training score:", reg.score(X_train, y_train[:, i]), "validation score:", reg.score(X_val, y_val[:, i]))
#     extra_trees_reg_list.append(reg)
#     break
# print()

# xgb_reg_list = []
# for i in range(y.shape[1]):
#     X_train = np.concatenate((aws_train[:, :, i], gfs_train_reduced), axis=1)
#     X_val = np.concatenate((aws_val[:, :, i], gfs_val_reduced), axis=1)
#     dtrain_reg = xgb.DMatrix(X_train, y_train[:, i])
#     dval_reg = xgb.DMatrix(X_val, y_val[:, i])
#     params = {
#         "objective": "reg:squarederror",
#         "tree_method": "hist",
#     }
#     n = 100
#     evals = [(dtrain_reg, "train"), (dval_reg, "validation")]
#     model = xgb.train(
#         params=params,
#         dtrain=dtrain_reg,
#         num_boost_round=n,
#         evals=evals,
#         early_stopping_rounds=3,
#     )
#     xgb_train_preds = model.predict(dtrain_reg)
#     xgb_val_preds = model.predict(dval_reg)
#     print(regions[i], "Training score:", r2_score(y_train[:, i], xgb_train_preds), "Validation score:", r2_score(y_val[:, i], xgb_val_preds))
#     print()
#     xgb_reg_list.append(model)
# print()


class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_dim_1, hidden_dim_2, hidden_dim_3, output_dim):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_dim_1)
        self.fc2 = nn.Linear(hidden_dim_1, hidden_dim_2)
        self.fc3 = nn.Linear(hidden_dim_2, hidden_dim_3)
        self.fc4 = nn.Linear(hidden_dim_3, output_dim)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.relu(out)
        out = self.fc4(out)
        # out = self.relu(out)
        return out


class ModelCheckpoint:
    def __init__(self, filepath):
        self.filepath = filepath
        self.min_val_loss = float('inf')
    
    def __call__(self, model, val_loss):
        if val_loss < self.min_val_loss:
            self.min_val_loss = val_loss
            torch.save(model.state_dict(), self.filepath)
            print(f'Saved model with validation loss: {self.min_val_loss}')


# Customized loss function
class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()

    def forward(self, y_true, y_pred):
        weights = torch.tensor([
            1 if i < 0.25 else
            10 if i < 0.5 else
            50 if i < 0.75 else
            100 for i in y_true
        ])
        return torch.mean(weights * ((y_true - y_pred) ** 2))


# Set number of decimal places to 4
np.set_printoptions(precision=4)
nn_list = []
plot_list = []
for i in range(y.shape[1]):
    X_train = np.concatenate((aws_train[:, :, i], gfs_train_reduced), axis=1)
    X_train = np.asarray(X_train, dtype=np.float32)
    X_train = torch.tensor(X_train, dtype=torch.float32)

    y_train2 = np.asarray(y_train[:, i], dtype=np.float32)
    y_train2 = np.reshape(y_train2, (-1, 1))
    y_train2 = torch.tensor(y_train2, dtype=torch.float32)
    
    X_val = np.concatenate((aws_val[:, :, i], gfs_val_reduced), axis=1)
    X_val = np.asarray(X_val, dtype=np.float32)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    
    y_val2 = np.asarray(y_val[:, i], dtype=np.float32)
    y_val2 = np.reshape(y_val2, (-1, 1))
    y_val2 = torch.tensor(y_val2, dtype=torch.float32)
    
    model = NeuralNet(
        input_size=X_train.shape[1],
        hidden_dim_1=16,
        hidden_dim_2=8,
        hidden_dim_3=4,
        output_dim=1,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Use batched training
    batch_size = 128
    train_dataset = TensorDataset(X_train, y_train2)
    val_dataset = TensorDataset(X_val, y_val2)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)

    epochs = 50
    learning_rate = 0.003
    l1_lambda = 0.000000
    l2_lambda = 0.000000
    step_size = 40
    gamma = 0.3
    
    nn_list.append(model)
    criterion = CustomLoss()
#     criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)  # Learning rate scheduler
    checkpoint_callback = ModelCheckpoint("/Users/kpmac/Documents/IITB/DS593/model/checkpoints/nn1_" + str(i) + ".pth")
    loss_list = []
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for j, (inputs, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            l1_norm = sum(p.abs().sum() for p in model.parameters())
            l2_norm = sum(p.pow(2.0).sum() for p in model.parameters())
            loss = loss + l1_lambda * l1_norm + l2_lambda * l2_norm
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        # Calculate validation loss
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
    
        print(f'Epoch {epoch+1}, Training Loss : {epoch_loss/len(train_loader)}, Validation Loss: {val_loss/len(val_loader)}')
        loss_list.append(epoch_loss)

        checkpoint_callback(model, avg_val_loss)

        # scheduler.step()
        
    # # Plot the training and validation loss
    # plt.plot(loss_list)
    # plt.xlabel("Iteration")
    # plt.ylabel("Loss")
    # plt.title(f"{regions[i]} Training Loss")
    # plt.show()
    plot_list.append(loss_list)
    
    # Re-read the best model
    model.load_state_dict(torch.load("/Users/kpmac/Documents/IITB/DS593/model/checkpoints/nn1_" + str(i) + ".pth"))
    model.eval()
    with torch.no_grad():
        y_pred_train = model(X_train).cpu().numpy()
        y_pred_val = model(X_val).cpu().numpy()
        print("Training score:%.4f" % r2_score(y_train2, y_pred_train), "Validation score:%.4f " % r2_score(y_val2, y_pred_val), regions[i])
    break
