import xarray as xr
import pygrib as pg
import numpy as np

pr_forecast_path = "/Users/kpmac/Documents/IITB/DS593/data/GFS/PR/Forecast/gfs.0p25.2023051806.f162-25.2023100406.f006.grib2.nc/gfs.0p25.2023100206.f003.grib2.nc"
pw_forecast_path = "/Users/kpmac/Documents/IITB/DS593/data/GFS/PW/Forecast/gfs.0p25.2023052400.f024-25.2023100206.f006.grib2.nc/gfs.0p25.2023100206.f003.grib2.nc"
rh_forecast_path = "/Users/kpmac/Documents/IITB/DS593/data/GFS/RH/Forecast/gfs.0p25.2023052400.f024-25.2023101206.f006.grib2.nc/gfs.0p25.2023101206.f003.grib2.nc"
tcc_forecast_path = "/Users/kpmac/Documents/IITB/DS593/data/GFS/TCC/Forecast/gfs.0p25.2023052400.f024-25.2023101012.f006.grib2.nc/gfs.0p25.2023100206.f003.grib2.nc"

pr_analysis_path = "/Users/kpmac/Documents/IITB/DS593/data/GFS/PR/Analysis/gfs.0p25.2019061212-25.2024091812.f000.grib2.nc/gfs.0p25.2023100206.f000.grib2.nc"
pw_analysis_path = "/Users/kpmac/Documents/IITB/DS593/data/GFS/PW/Analysis/gfs.0p25.2015011500-25.2024091812.f000.grib2.nc/gfs.0p25.2023100206.f000.grib2.nc"
rh_analysis_path = "/Users/kpmac/Documents/IITB/DS593/data/GFS/RH/Analysis/gfs.0p25.2015011500-25.2024091312.f000.grib2.nc/gfs.0p25.2023100206.f000.grib2.nc"

# grib_file_path = "/Users/kpmac/Downloads/gfs.t12z.pgrb2.0p25.f000"
# grib_file = pg.open(grib_file_path)
# grib_object = grib_file.read(1)[0]
# print(grib_object)
# print(grib_object.values)      # note that here, the values are in reverse order of longitude

# dataset = nc.Dataset(datapath)
# 
# print(dataset)
# print()
# print()
# 
# dimkeys = dataset.dimensions.keys()         # <class 'dict_keys'>
# dimvals = dataset.dimensions.values()    # <class 'dict_values'>
# for k, v in zip(dimkeys, dimvals):
#     print(k)
#     print(v)
#     print()
# print()
# print()
# 
# varkeys = dataset.variables.keys()          # <class 'dict_keys'>
# varvals = dataset.variables.values()      # <class 'dict_values'>
# for k, v in zip(varkeys, varvals):
#     print(k)
#     print(v)
#     print()
# print()
# print()
# 
# dataset.close()

# data_vars = dataset.data_vars

# for k, v in data_vars.items():
#     print(k)
#     print(v)
#     print()

# print(data_vars["R_H_L103"])
# df = dataset["PRATE_L1_Avg_1"].to_dataframe()
# print(df.reset_index())
pr_forecast_path = "/Users/kpmac/Documents/IITB/DS593/data/GFS/PR/Forecast/full/gfs.0p25.2016053112.f003.grib2.nc"
pw_forecast_path = "/Users/kpmac/Documents/IITB/DS593/data/GFS/PW/Forecast/full/gfs.0p25.2016053112.f003.grib2.nc"

pr_forecast = xr.open_dataset(pr_forecast_path)
pw_forecast = xr.open_dataset(pw_forecast_path)
rh_forecast = xr.open_dataset(rh_forecast_path)
tcc_forecast = xr.open_dataset(tcc_forecast_path)

pr_analysis = xr.open_dataset(pr_analysis_path)
pw_analysis = xr.open_dataset(pw_analysis_path)
rh_analysis = xr.open_dataset(rh_analysis_path)

# print(pr_forecast)
# print()
# 
# print(pr_analysis)
# print()
# 
# print(pw_forecast)
# print()
# 
# print(pw_analysis)
# print()
# 
# print(rh_forecast)
# print()
# 
# print(rh_analysis)
# print()
# 
# print(tcc_forecast)
# print()


pr_forecast_df = pr_forecast["PRATE_L1_Avg_1"].to_dataframe().reset_index()
print(pr_forecast_df["PRATE_L1_Avg_1"].values)
print()
pw_forecast_df = pw_forecast["P_WAT_L200"].to_dataframe().reset_index()
print(pw_forecast_df["P_WAT_L200"].values)
print()
#temp = np.array([])
#temp = np.concatenate((temp, pr_forecast_df["PRATE_L1_Avg_1"].values), axis=None)
#print(temp)
# rh_forecast_df = rh_forecast["R_H_L103"].to_dataframe()
# rh_analysis_df = rh_analysis["R_H_L103"].to_dataframe()

# print(rh_forecast_df.reset_index())
# print(rh_analysis_df.reset_index())
