"""
此文件用于测试运行
Wait on:  1. 各个文件的每个函数, 每个类的参数的用途都解释一下, 方便之后运行
          2. 思考优化一下函数, 方便之后遇到类似的问题能直接调用
"""

import numpy as np
from generalTools import AddressData
from getTwoPointsList import  RecognizeDisturbace
from GUI import DrawImg
import pandas as pd

init_data_pth = r'C:\Users\2892706668\Desktop\20230725_0726_850hpa.nc'
init_obj = AddressData(init_data_pth)
u,v, vo, lons, lats , timeOfSpecific =  (np.squeeze(x) for x in init_obj.get_init_data())
# 对时间截取到小时
timeOfSpecific = np.datetime_as_string(timeOfSpecific, unit='h')
print(u.shape)
print(v.shape)
print(vo.shape)
print(lons.shape)
print(lats.shape)

u_smth = np.stack([init_obj.smth9_2d(u[t]) for t in range(u.shape[0])], axis=0)
v_smth = np.stack([init_obj.smth9_2d(v[t]) for t in range(v.shape[0])], axis=0)
vo_smth = np.stack([init_obj.smth9_2d(vo[t]) for t in range(vo.shape[0])], axis=0)

AddressData.calculate_overlap(center_point1=(0,0), center_point2=(0,0), wOfhalf=1, hOfhalf=1, lons=lons, lats=lats)

# 下面进行追踪
t_length = len(timeOfSpecific)
getDistCenterObj = RecognizeDisturbace( t_length, u_smth, v_smth,vo_smth, lons, lats  )
get_dist_centerlist , test_list = getDistCenterObj.getDstCenter()
tracked_list , real_lonlat_list = getDistCenterObj.trackDstCenter(test_list)


# 下面进行可视化
img_obj_smthtest = DrawImg(lons,lats,timeOfSpecific, (u_smth,v_smth, vo_smth) , save_img_pth= '此处未命名,因为现在没打算保存'    )
img_obj_smthtest.draw_img(get_dist_centerlist, test_list, '所有扰动点示意图')

# 绘制满足连续时刻的点图,  默认为7个连续时刻,  一个时刻是6小时, 那就是持续存在  (7-1) * 6 = 36  h ,
DrawImg.draw_track_points(tracked_list, lons, lats, timeOfSpecific, '符合连续时刻的扰动点')

#-----------------------------
# 下面判断发展扰动和不发展扰动
#-----------------------------
file_pth = r"C:\Users\2892706668\Desktop\ibtracs.WP.list.v04r01.csv"
tc_data = pd.read_csv(file_pth)
pd_time_range = pd.date_range(start='2023-07-25 00:00:00', end='2023-07-26 18:00:00', freq='6h')
tc_data['ISO_TIME'] = pd.to_datetime(tc_data['ISO_TIME'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
mask = tc_data.ISO_TIME.isin(pd_time_range)
tc_dataOfCleaning = tc_data[mask]

# 储存不同时刻台风位置和 i_sid
all_SIDs = tc_dataOfCleaning.SID.unique()
# 使用 groupby 一次性分组
all_t_TCPoints = []
for i_isotime, group in tc_dataOfCleaning.groupby('ISO_TIME'):
    i_t_TcPoints =  []
    i_t_lons= group['USA_LON'].values  # 某个时刻的所有台风的经度
    i_t_lats = group['USA_LAT'].values
    i_t_sids = group['SID']
    for i_lon, i_lat, i_sid in zip(i_t_lons, i_t_lats, i_t_sids):
        i_t_TcPoints.append((np.float64(i_lon), np.float64(i_lat), i_sid))
    all_t_TCPoints.append(i_t_TcPoints)

final_nondev_centers, final_dev_centers = getDistCenterObj.judegeDevOrNonDev(real_lonlat_list, all_t_TCPoints)  # 两个列表均为真实维度

# 可视化分组,  dev 用红色,  nondev 用蓝色
# DrawImg.draw_two_points_list(lons, lats,
#                              dev_pts_list=all_t_TCPoints,
#                              nondev_pts_list=real_lonlat_list,
#                              save_img_name='dev和nondev的不同颜色分组图')


DrawImg.draw_two_points_list(lons, lats,
                             dev_pts_list=final_dev_centers,
                             nondev_pts_list=final_nondev_centers,
                             save_img_name='dev和nondev的不同颜色分组图')

