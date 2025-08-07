"""
本文件用于存放一些常用的工具类和函数, 希望随着经验的积累, 越来越多的工具类和函数被加入到这个文件中.

"""
import xarray  as xr
import numpy as np
from typing import Tuple ,  List , Set, Any
from scipy import ndimage

# 关于数据处理的类
class AddressData(object):
    """
    Attributes:
        init_data_pth: str  : 初始数据的路径, 也就是读取的nc文件的路径
    """
    def __init__(self, init_data_pth: str ):
        self.init_data_pth = init_data_pth

    # 获取原始数据
    def get_init_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        init_data = xr.open_dataset(self.init_data_pth)
        # 获取每隔6小时的数据 (原始数据间隔为1小时)，  经纬度在下载的时候已经确定
        init_data = init_data.sel(valid_time=init_data.valid_time.values[::6])
        u = init_data.u.values                # u风场  , 2D array
        v = init_data.v.values                # v风场  , 2D array
        vo = init_data.vo.values              # 涡度场    , 2D array
        lons = init_data.longitude.values      # 经度    , 1 D array
        lats = init_data.latitude.values       # 纬度    , 1 D  array
        timeOfSpecific = init_data.valid_time.values  # 时间    , 1D array , 内部存放具体时刻
        return u, v, vo, lons, lats, timeOfSpecific

    # 二维九点平滑函数  2D smth9 for (lat, lon)
    def smth9_2d(self, data: np.ndarray,    p=0.5, q=0.25) -> np.ndarray:
        """
        二维九点平滑函数
        Args:
            data: 2D array
            p:  用于调节平滑程度的参数， 默认为0.5
            q:  用于调节平滑程度的参数， 默认为0.25
        Returns:
            2D array 平滑后的数组
        """
        kernel = np.array([
            [p / 4, q / 4, p / 4],
            [q / 4, 1 - p - q, q / 4],
            [p / 4, q / 4, p / 4]
        ])
        smoothed = ndimage.convolve(data, kernel, mode='constant', cval=0.0)
        return smoothed

    # 覆盖面积计算
    @staticmethod
    def calculate_overlap(center_point1: tuple,
                          center_point2: tuple,
                          wOfhalf: int,
                          hOfhalf: int,
                          lons: np.ndarray,
                          lats: np.ndarray,
                          ration_threshold: float = 0.5
                          ) -> Tuple[bool, float]:
        """
        对两个中心点，框定同样大小的box, 计算两个box的重叠面积占参考区域的比例。
        Args:
            center_point1: (x1, y1) 第一个中心点的经纬度坐标。
            center_point2: (x2, y2) 第二个中心点的经纬度坐标。
            wOfhalf: int  半径的宽度, 也就是两个框的宽度的一半的长度 。
            hOfhalf: int  半径的高度, 也就是两个框的高度的一半的长度 。
            lons: 1D array  经度
            lats: 1D array  纬度
            ration_threshold: float  覆盖面积的阈值, 默认为0.5
        Returns:
            Tuple[bool, float]: 返回一个元组，包含两个元素：
                - **onoff** (`bool`): 表示两个区域是否存在重叠覆盖（True 表示有覆盖）。
                - **overlap_ratio** (`float`): 表示实际的重叠面积占参考区域（如第一个区域）面积的比例，范围 [0, 1]。
        """

        mask1 = np.zeros((len(lons), len(lats)))
        mask2 = np.zeros((len(lons), len(lats)))

        x1 = center_point1[0]
        x2 = center_point2[0]
        y1 = center_point1[1]
        y2 = center_point2[1]
        mask1[x1 - wOfhalf:x1 + wOfhalf + 1, y1 - hOfhalf:y1 + hOfhalf + 1] = 1
        mask2[x2 - wOfhalf:x2 + wOfhalf + 1, y2 - hOfhalf:y2 + hOfhalf + 1] = 1
        intersection = np.logical_and(mask1, mask2).sum()

        box_area = 2 * wOfhalf * 2 * hOfhalf
        overlap_ratio = intersection / box_area
        onoff = overlap_ratio > ration_threshold

        return onoff, overlap_ratio


# ----------------------------------------
# 关于风的处理
# ---------------------------------------
class WindRelated(object):
    def __init__(self, u: np.ndarray, v: np.ndarray, lons: np.ndarray, lats: np.ndarray):
        self.u = u
        self.v = v
        self.lons = lons
        self.lats = lats

    # 气旋性环流判断
    # 后期可以补充一下这个函数

    # 风速计算
    @staticmethod
    def wind_speed(u: np.ndarray, v: np.ndarray):
        """
        计算风速
        Args:
            u: 2D array  u风场
            v: 2D array  v风场
        Returns:
            2D array 风速
        """
        return np.sqrt(u ** 2 + v ** 2)
