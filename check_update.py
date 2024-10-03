import requests
import time

tick = time.time()
response = requests.get("https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25_1hr.pl?dir=%2Fgfs.20240920%2F06%2Fatmos&file=gfs.t06z.pgrb2.0p25.anl&all_var=on&all_lev=on&subregion=&toplat=20&leftlon=72&rightlon=74&bottomlat=18")
print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), "--", response.status_code)
while response.status_code != 200:
    time.sleep(300)
    if time.time() - tick >= 300:
        tick = time.time()
        response = requests.get("https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25_1hr.pl?dir=%2Fgfs.20240920%2F06%2Fatmos&file=gfs.t06z.pgrb2.0p25.anl&all_var=on&all_lev=on&subregion=&toplat=20&leftlon=72&rightlon=74&bottomlat=18")
        print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), "--", response.status_code)
