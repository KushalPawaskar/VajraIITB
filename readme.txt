GFS model initialized at 0530, 1130, 1730, 2330 IST.
Latest model outputs available about 3 hours 30 mintues after initialization.
Outputs include:
    Analysis data at moment of initialization
    Forecasts ranging from 0 hour to 384 hours (0, 1, 2, ..., 119, 120, 123, 126, 129, ..., 381, 384) (0 to 120 in 1 hour gaps, 120 to 384 in 3 hour gaps)
    This is the real time data availability (only past 10 days available).
    Rest of the data (historical) available only in 3 hour gaps (minimum)
    Forecast data with me (historical):
        PW: 3 hour gaps in 3 hour to 162 hour forecast
        PR: 3 hour gaps in 3 hour to 24 hour forecasts
        RH: 3 hour gaps in 3 hour to 24 hour forecasts
        TCC: 3 hour gaps in 3 hour to 24 hour forecasts
    Analysis data with me (historical):
        PW
        PR
        RH

AWS data available from 1 June to 30 September in 15 min gaps from 2015 to 2023

Now, let's say I want to run my model at 2 pm for prediction of rainfall between 2 pm and 2:15 pm.
So, I will have the AWS data till 2 pm. Let us choose the AWS data for the past 24 hours as the input to our model.
I also have GFS data: (considering the latest model forecasts as inputs)
    M-2024-09-20-0530.f012
    M-2024-09-20-0530.f009
    M-2024-09-20-0530.f006
    M-2024-09-20-0530.f003
    M-2024-09-20-0530.f000

My model run times and GFS data availability (run model 1-2 min before multiple of 15 min):
0915-1500   M-2024-09-20-0530 (00Z)
1515-2100   M-2024-09-20-1130 (06Z)
2115-0300   M-2024-09-20-1730 (12Z)
0315-0900   M-2024-09-20-2330 (18Z)

To forecast for 2 June 2015 00:15:00, we will use datapoints from 1 June 2015 00:15:00 to 2 June 2015 00:00:00

81*4*4 + 81*2 + 4 (feature size)

Linear Regression
SVM (Regression)
xgBoost
CNN + LSTM

9x9x4 (3x3)
CNN GFS
LSTM AWS

Steps:
•LinearRegression, Ridge, ElasticNet, Lars, Lasso, LassoLars, OrthogonalMatchingPursuit, ARDRegression, BayesianRidge
•LinearSVR, NuSVR, SVR
•xgBoost
•Add one hot vector and try
•Neural Network
--Linear ReLU Linear ReLU Linear ReLU
•Change loss to weighted MSE

•Make modular code to take different arguments of forecast horizon, history, etc. which will prepare the data
