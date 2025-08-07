"""
此文件用于对getTwoPointsList.py文件内容的可视化, 以此来验证算法的正确性.
即找到正确的 Dev 和 Non-Dev 点.

"""

import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
from matplotlib import colormaps
from typing import Tuple ,  List , Set, Any
from numpy.typing import NDArray
from matplotlib.patches import Rectangle
import os


class DrawImg( object):
    def __init__(self,
                 lons: np.ndarray,
                 lats: np.ndarray,
                 specific_times: np.ndarray,
                 variable: tuple,    # (u, v, vo , .....)
                 save_img_pth: str,
                 ):

        self.lons = lons
        self.lats = lats
        self.specific_times = specific_times
        self.u = variable[0]
        self.v = variable[1]
        self.vo = variable[2]
        self.save_img_pth = save_img_pth

    def draw_img(self, dist_center_list: List[List[Tuple[Any, Any,]]  ] ,
                 test_list: List[List[Tuple[Any, Any,]]  ],
                 save_img_name : str ,
                 ):

        wOfBox, hOfBox = 16, 16  # 画出四度的格点
        # === 第一步：统一 colorbar 的范围 ===
        # 找出所有 vo 数据的全局最小最大值（或你也可以手动设置）
        vmin = np.min(self.vo)  # 或者用 np.percentile(self.vo, 1)
        vmax = np.max(self.vo)  # 或者用 np.percentile(self.vo, 99)
        # 如果数据有异常值，建议用：
        # vmin, vmax = np.percentile(self.vo, [2, 98])

        # === 第二步：创建 4x2 子图 ===
        fig, axes = plt.subplots( 4, 2, figsize=(12, 10),
                                subplot_kw={'projection': ccrs.PlateCarree()})
        axes = axes.flatten()

        # 用于存储每个子图的 contourf 对象（用于 colorbar）
        contourf_plots = []
        for i in  range(len(self.u)):  # 对对每个时刻进行处理
            ax = axes[i]
            # 地理特征
            ax.add_feature(cfeature.LAND, facecolor='lightgray')
            ax.add_feature(cfeature.OCEAN, facecolor='lightblue')
            ax.add_feature(cfeature.COASTLINE)
            ax.add_feature(cfeature.BORDERS, linewidth=0.5)

            # 网格线
            gl = ax.gridlines(draw_labels=True,
                            x_inline=False, y_inline=False,
                            linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
            gl.top_labels = False
            gl.right_labels = False

            # 绘制风矢量（在上层）
            # # 降采样
            skip = (slice(None, None, 6), slice(None, None, 6))
            ax.quiver(self.lons[::6], self.lats[::6], self.u[i][skip], self.v[i][skip],
                    scale=350,  # 控制箭头长度的缩放比例,值越大箭头越短
                    color='k',
                    pivot='middle',  # 箭头以(lon,lat)点为中心绘制
                    width=0.003,  # 箭头宽度
                    headwidth=3,   # 箭头头部宽度
                    headlength= 3,  # 箭头头部长度
                    zorder=1,
                    transform=ccrs.PlateCarree())
            #
            # # 画出流线图
            # ax.streamplot(self.lons[::6], self.lats[::6], self.u[i][skip], self.v[i][skip],
            #         density= 4,   # 对应流线图  控制流线的数量和紧密程度
            #         # scale=350,  # 控制箭头长度的缩放比例,值越大箭头越短
            #         # color='k',
            #         # pivot='middle',  # 箭头以(lon,lat)点为中心绘制
            #         # width=0.003,  # 箭头宽度
            #         # headwidth=3,   # 箭头头部宽度
            #         # headlength= 3,  # 箭头头部长度
            #         color='k',
            #         zorder=1,
            #         transform=ccrs.PlateCarree())


            # 绘制 涡度的填色图（在底层），
            cf = ax.contourf(self.lons, self.lats , self.vo[i],
                            cmap=colormaps['coolwarm'],
                            vmin=vmin, vmax=vmax,  # 关键：统一颜色范围
                            transform=ccrs.PlateCarree(),
                            zorder=0
                             )
            contourf_plots.append(cf)
            ax.set_title(f'{self.specific_times[i]}', fontsize=12)

            # # 画框
            if dist_center_list is not None:
                # for i_point in dist_center_list[i]:
                #         (x, y, w, h) = i_point
                #         # 转换索引为地理坐标
                #         lon_start = self.lons[x-wOfBox]
                #         lat_start = self.lats[y+hOfBox]
                #         lon_width = self.lons[x + wOfBox - 1] - lon_start
                #         lat_height = self.lats[y - hOfBox - 1] - lat_start

                        # # 绘制框 ,  传入左下角的点, 宽 , 高
                        # rect = Rectangle((lon_start, lat_start ),
                        #                  lon_width, lat_height,
                        #                  linewidth=2,
                        #                  edgecolor='red', facecolor='none',
                        #                  linestyle='-',
                        #                  zorder=5,
                        #                  transform=ccrs.PlateCarree())
                        # ax.add_patch(rect)
                        # ax.scatter(lon_start, lat_start,
                        #            c='yellow',
                        #            s= 10,  # s 是面积，相当于 markersize^2
                        #            transform=ccrs.PlateCarree(),
                        #            zorder=6)

                # 下方为测试散点
                for i_point_test in test_list[i]:
                    x_test , y_test =  i_point_test
                    test_lon_start = self.lons[x_test - wOfBox].item()
                    test_lat_start = self.lats[y_test + hOfBox].item()
                    lon_width = self.lons[x_test + wOfBox - 1 ] - test_lon_start
                    lat_height = self.lats[y_test - hOfBox - 1] - test_lat_start
                    # ax.scatter(test_lon_start, test_lat_start,  左下角
                    ax.scatter(self.lons[x_test], self.lats[y_test],
                                   c ='yellow',
                                   s= 10,  # s 是面积，相当于 markersize^2
                                   transform=ccrs.PlateCarree(),
                                   zorder=6)
                    rect = Rectangle((test_lon_start, test_lat_start ),
                                         lon_width, lat_height,
                                         linewidth=2,
                                         edgecolor='red', facecolor='none',
                                         linestyle='-',
                                         zorder=5,
                                         transform=ccrs.PlateCarree())
                    ax.add_patch(rect)

        # === 第三步：添加共享 colorbar ===
        # 获取最后一个 contourf 对象的颜色映射（所有都一样）
        # 使用 fig.colorbar 并指定所有 axes
        fig.subplots_adjust(right=0.85)  # 为 colorbar 留出空间
        cbar_ax = fig.add_axes([0.87, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
        cbar = fig.colorbar(contourf_plots[0], cax=cbar_ax)
        cbar.set_label('Vorticity (vo)', fontsize=12)
        # 在当前路径下新建名字为img的文件夹
        if not os.path.exists('./img'):
            os.makedirs('./img')
        # 保存图片
        plt.savefig(f'./img/{save_img_name}.png', dpi=300)

    @staticmethod
    def draw_track_points(
        tracked_list : list,
        lons: NDArray,
        lats:NDArray,
        specific_times,
        save_img_name: str,
        wOfBox: int = 16,
        hOfBox: int = 16,
                          ):
        """  将所有的时刻的点都画在同一张图上 """
        fig, axes = plt.subplots(1, 1, figsize=(12, 10),
                                 subplot_kw={'projection': ccrs.PlateCarree()})
        ax = axes
        # 用于存储每个子图的 contourf 对象（用于 colorbar）
        for i in range(len(specific_times)):
            ax.add_feature(cfeature.LAND, facecolor='lightgray')
            ax.add_feature(cfeature.OCEAN, facecolor='white')
            ax.add_feature(cfeature.COASTLINE)
            ax.add_feature(cfeature.BORDERS, linewidth=0.5)

            # 网格线
            gl = ax.gridlines(draw_labels=True,
                              x_inline=False, y_inline=False,
                              linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
            gl.top_labels = False
            gl.right_labels = False

            # # 画框
            # colorlist = ['red', 'green', 'blue', 'yellow', 'purple', 'orange', 'gray', 'pink']
            if tracked_list is not None:
                # 下方为测试散点
                for i_point_test in tracked_list[i]:
                    x_test, y_test, id = i_point_test
                    test_lon_start = lons[x_test - wOfBox].item()
                    test_lat_start = lats[y_test + hOfBox].item()
                    lon_width = lons[x_test + wOfBox - 1] - test_lon_start
                    lat_height = lats[y_test - hOfBox - 1] - test_lat_start
                    # ax.scatter(test_lon_start, test_lat_start,  左下角

                    ax.scatter(lons[x_test], lats[y_test],
                               # c = colorlist[i],
                               c='k',
                               s=10,  # s 是面积，相当于 markersize^2
                               transform=ccrs.PlateCarree(),
                               zorder=6)
                    rect = Rectangle((test_lon_start, test_lat_start),
                                     lon_width, lat_height,
                                     linewidth=2,
                                     edgecolor='red', facecolor='none',
                                     linestyle='-',
                                     zorder=5,
                                     transform=ccrs.PlateCarree())
                    ax.add_patch(rect)

                    # 标记点对应的时刻的索引
                    ax.text(
                        lons[x_test],  # x 坐标（经度）
                        lats[y_test] + 0.1,  # y 坐标（纬度）稍微上移 0.1 度
                        str(i),  # 标注内容（这里用 `i` 作为示例）
                        # color=colorlist[i],
                        color='k',
                        fontsize=15,
                        transform=ccrs.PlateCarree(),
                    )

        if not os.path.exists('./img'):
            os.makedirs('./img')
        # 保存图片
        plt.savefig(f'./img/{save_img_name}.png', dpi=300)


#-------------------------------------
# 绘制两组不同的点,  即 dev 和  non-dev
#-------------------------------------
    @staticmethod
    def draw_two_points_list(
        lons:  NDArray,
        lats:  NDArray,
        dev_pts_list :  List[List[Tuple[float, float, Any]]],
        nondev_pts_list: List[List[Tuple[float, float, Any]]],
        save_img_name: str,
        ):
        """
        画出两组点的分布图, 其中 dev_pts_list 和 nondev_pts_list 都是二维列表, 第一维是时间维度, 第二维是点的坐标
        Args:
            lons: 经度
            lats: 纬度
            dev_pts_list:  dev 点的坐标列表
            nondev_pts_list: non-dev 点的坐标列表
            save_img_name: 保存图片的名字， 最终保存图片的路径为: ./img/save_img_name.png
        """

        pd_time_range = len(dev_pts_list)
        fig, axes = plt.subplots(1, 1, figsize=(12, 10),
                                 subplot_kw={'projection': ccrs.PlateCarree()})
        ax = axes
        # 用于存储每个子图的 contourf 对象（用于 colorbar）
        ax.set_extent([lons.min(), lons.max(), lats.min(), lats.max()], crs=ccrs.PlateCarree())
        for i in range(pd_time_range):
            ax.add_feature(cfeature.LAND, facecolor='lightgray')
            ax.add_feature(cfeature.OCEAN, facecolor='white')
            ax.add_feature(cfeature.COASTLINE, )
            ax.add_feature(cfeature.BORDERS, linewidth=0.1)

            # 网格线
            gl = ax.gridlines(draw_labels=True,
                              x_inline=False, y_inline=False,
                              linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
            gl.top_labels = False
            gl.right_labels = False

            # # 画框
            colorlist = ['red', 'green', 'blue', 'yellow', 'purple', 'orange', 'gray', 'pink']
            if (dev_pts_list is not None) or (nondev_pts_list is not None):
                # 下方为TC中心
                for i_point_test in dev_pts_list[i]:
                    # print(i_point_test)
                    x_test, y_test, sid = i_point_test
                    ax.scatter(x_test, y_test,  # lons[x_test], lats[y_test],
                               # c = colorlist[i],
                               c='r',
                               s=10,  # s 是面积，相当于 markersize^2
                               zorder=1,
                               transform=ccrs.PlateCarree(),
                               )
                    # ax.text(
                    #     x_test,  # x 坐标（经度）
                    #     y_test + 0.1,  # y 坐标（纬度）稍微上移 0.1 度
                    #     str(i),  # 标注内容（这里用 `i` 作为示例）
                    #     # color=colorlist[i],
                    #     color = 'r',
                    #     fontsize= 8,
                    #     transform=ccrs.PlateCarree(),
                    #     zorder=7  # 确保文本在散点和矩形之上
                    #     )

                # 下方为找到的 non-dev 扰动中心
                for i_point_test in nondev_pts_list[i]:
                    # print(i_point_test)
                    x_test, y_test, id = i_point_test
                    test_lon_start = x_test
                    test_lat_start = y_test
                    ax.scatter(x_test, y_test,
                               # c = colorlist[i],
                               c='blue',
                               s=10,  # s 是面积，相当于 markersize^2
                               zorder=0,
                               transform=ccrs.PlateCarree()
                               )

                    # if i in [6, 7] :
                    #     rect = Rectangle((test_lon_start, test_lat_start ),
                    #                          lon_width, lat_height,
                    #                          linewidth=2,
                    #                          edgecolor='red', facecolor='none',
                    #                          linestyle='-',
                    #                          zorder=5,
                    #                          transform=ccrs.PlateCarree())
                    #     ax.add_patch(rect)

                    #  # 标记点对应的时刻的索引

                #  ax.text(
                #      lons[x_test],  # x 坐标（经度）
                #      lats[y_test] + 0.1,  # y 坐标（纬度）稍微上移 0.1 度
                #      str(i),  # 标注内容（这里用 `i` 作为示例）
                #      # color=colorlist[i],
                #      color = 'blue',
                #      fontsize= 8,
                #      transform=ccrs.PlateCarree(),
                #      zorder=7  # 确保文本在散点和矩形之上
                # )

        plt.savefig(f'./img/{save_img_name}.png', dpi=300)

