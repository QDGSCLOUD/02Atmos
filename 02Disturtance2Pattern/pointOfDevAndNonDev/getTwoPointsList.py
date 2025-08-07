import numpy as np
import cv2
from typing import Tuple ,  List , Set, Any
from collections import defaultdict
from generalTools import AddressData

# 下方 对于  count_id 超过7 需要改一下


# 拿到环流点
"""  此处对拿到的框的内部点， 每个点都遍历一遍，从内向外找  """
class RecognizeDisturbace(object):
    """
    识别扰动中心的类
    Attributes:
        t_length: int, 时间长度
        u_smth: np.ndarray, 3D array (time, lat, lon), 平滑后的u
        v_smth: np.ndarray, 3D array (time, lat, lon), 平滑后的v
        vo_smth: np.ndarray, 3D array (time, lat, lon), 平滑后的vo
        lons: np.ndarray, 1D array, 经度
        lats: np.ndarray, 1D array, 纬度
    """
    def __init__(self,
                 t_length: int,
                 u_smth: np.ndarray,
                 v_smth: np.ndarray,
                 vo_smth: np.ndarray,
                 lons: np.ndarray,
                 lats: np.ndarray,
                 ):
        self.t_length: int = t_length
        self.u_smth: np.ndarray = u_smth
        self.v_smth: np.ndarray = v_smth
        self.vo_smth: np.ndarray = vo_smth
        self.lons: np.ndarray = lons
        self.lats: np.ndarray = lats

    def getDstCenter(self, radius_threshold: int = ( 4 /0.25 ),
                           vo_threshold: float = 3* (10 ** -5),
                     ) -> Tuple[List, List]:
        """
        识别出最大涡度位置 和扰动中心
        Args:
            radius_threshold: int, 环流点的半径阈值，默认16个格点。 默认要用4度， 分辨率为0.25度.
            vo_threshold: float, 环流点的最大涡度阈值，默认3*10^-5。 默认要用3*10^-5， 单位/s
            
        Returns: 
            Tuple[List, List]:
            - maxVo_center_list: List, 满足涡度阈值，且以lenOfHalfBox的环流box中，最大涡度的位置的列表
            - all_disturbance_list: List, (x, y) 得到单个场的扰动中心列表，每个点均为索引点， 若要使用真实的经纬度， 需要用lons, lats转换x,y
        """

        H, W = self.u_smth.shape[1], self.u_smth.shape[2]
        lenOfHalfBox = int(radius_threshold /2)
        maxVo_center_list = []
        all_disturbance_list = []
        for i_t in range(self.t_length):
            i_t_center_list = []
            test_list = []

            # condition1 :  先找出涡度 大于 3*10^-5
            conditionOfVo = np.where(self.vo_smth[i_t] > vo_threshold, 255, 0).astype(np.uint8)
            contours, _ = cv2.findContours(conditionOfVo, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # # 画出conditionOfVo对应的黑白图
            # plt.imshow(conditionOfVo, cmap='gray')

            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)        # 如果用 w = h = 16 就找不到了

                # condition2: 对涡度找到的box计算其面积
                area_threshold = radius_threshold ** 2
                box_area = w * h
                if box_area > area_threshold:
                    # condition3:
                    mask = np.zeros(conditionOfVo.shape, dtype=np.uint8)
                    cv2.drawContours(mask, [cnt], -1, 255, cv2.FILLED)

                    max_y, max_x = np.unravel_index(
                        np.argmax(np.where(mask == 255, self.vo_smth[i_t], -np.inf)),
                        self.vo_smth[i_t].shape
                    )
                    i_t_center_list.append((max_x, max_y, w, h))

                    # 记录候选中心
                    valid_candidates = []

                    # 获取轮廓内的所有点（或可改为稀疏采样：步长>1）
                    coords = np.where(mask == 255)
                    ys, xs = coords[0], coords[1]

                    # 遍历轮廓内每个点
                    for x, y in zip(xs, ys):  # 提高计算速度， 可以稀疏采样   对xs, ys 取[::2]
                        if not (
                                y >= lenOfHalfBox and y < H - lenOfHalfBox and x >= lenOfHalfBox and x < W - lenOfHalfBox):
                            continue  # 跳出靠近边界的点

                        # 检查以 (x, y) 为中心，在 lenOfHalfBox 范围内是否满足环流结构
                        has_north_u = has_south_u = has_west_v = has_east_v = False
                        radius = 1  # 从内圈开始向外
                        max_radius = lenOfHalfBox

                        while radius <= max_radius:
                            # 北边：u[y - radius, x] < 0 （北风向南吹）
                            if radius <= y:
                                u_north = self.u_smth[i_t][y - radius, x]
                                if u_north < 0:
                                    has_north_u = True

                            # 南边：u[y + radius, x] > 0 （南风向北吹）
                            if radius < H - y:
                                u_south = self.u_smth[i_t][y + radius, x]
                                if u_south > 0:
                                    has_south_u = True

                            # 西边：v[y, x - radius] < 0 （西风向东吹）
                            if radius <= x:
                                v_west = self.v_smth[i_t][y, x - radius]
                                if v_west < 0:
                                    has_west_v = True

                            # 东边：v[y, x + radius] > 0 （东风向西吹）
                            if radius < W - x:
                                v_east = self.v_smth[i_t][y, x + radius]
                                if v_east > 0:
                                    has_east_v = True

                            # 提前退出：如果四个方向都已经满足过，无需继续向外
                            if has_north_u and has_south_u and has_west_v and has_east_v:
                                break

                            radius += 1

                        # 判断是否构成完整环流
                        if np.all([has_north_u, has_south_u, has_west_v, has_east_v]):  # 至少3个方向满足
                            valid_candidates.append((x, y))

                    # 对每个点在进行筛选
                    refined_candidates = []

                    # 定义检查的两层（最外和次外）
                    outer_radius = 16
                    inner_radius = 15
                    # radii_to_check = [outer_radius, inner_radius]  # 可改为 lenOfHalfBox, lenOfHalfBox-1
                    radii_to_check = [outer_radius]

                    for (x, y) in valid_candidates:
                        # 边界检查：确保能取到 r=16 的外围点
                        if not (
                                y >= outer_radius and y < H - outer_radius and x >= outer_radius and x < W - outer_radius):
                            continue
                        # 四个方向的状态：是否至少有一层满足
                        north_ok = False
                        south_ok = False
                        west_ok = False
                        east_ok = False

                        # 记录每个方向是否“至少有一层满足”
                        directions_ok = 0  # 满足条件的方向数

                        for r in radii_to_check:
                            # 北：u[y - r, x] < 0
                            if not north_ok:
                                u_north = self.u_smth[i_t][y - r, x]
                                if u_north < 0:
                                    north_ok = True
                                    directions_ok += 1

                            # 南：u[y + r, x] > 0
                            if not south_ok:
                                u_south = self.u_smth[i_t][y + r, x]
                                if u_south > 0:
                                    south_ok = True
                                    directions_ok += 1

                            # 西：v[y, x - r] < 0
                            if not west_ok:
                                v_west = self.v_smth[i_t][y, x - r]
                                if v_west < 0:
                                    west_ok = True
                                    directions_ok += 1

                            # 东：v[y, x + r] > 0
                            if not east_ok:
                                v_east = self.v_smth[i_t][y, x + r]
                                if v_east > 0:
                                    east_ok = True
                                    directions_ok += 1

                        # 判断：至少两个方向满足
                        if directions_ok >= 4:
                            refined_candidates.append((x, y))

                    # === 从 refined_candidates 中选择最终中心 ===
                    if refined_candidates:
                        candidates_array = np.array(refined_candidates)
                        distances = np.linalg.norm(candidates_array - np.array([max_x, max_y]), axis=1)
                        best_idx = np.argmin(distances)
                        final_center_x, final_center_y = refined_candidates[best_idx]

                        test_list.append((final_center_x, final_center_y))
                    else:
                        pass

            all_disturbance_list.append(test_list)  # 更加精确的中心
            maxVo_center_list.append(i_t_center_list)  # 粗滤的中心点

        return maxVo_center_list, all_disturbance_list

    # 通过id 对存在的时间进行标记, 从而选出符合连续时刻数量的id
    def trackDstCenter(self, all_disturbance_list: list,
                       wOfhalf: int = 16,
                       hOfhalf: int = 16,
                       overlap_thresh: float = 0.7,
                       count_threshold: int = 7
                       ) -> Tuple[List[List[Tuple[float, float, Any]]], List[List[Tuple[float, float, Any]]]]:
        """
        Args:
            all_disturbance_list: List, (x, y) 得到单个场的扰动中心列表，每个点均为索引点
            wOfhalf: int, 环流点的半径阈值，默认16个格点。 默认要用4度， 分辨率为0.25度.
            hOfhalf: int, 环流点的半径阈值，默认16个格点。 默认要用4度， 分辨率为0.25度.
            overlap_thresh: float,  重叠率阈值，默认0.7
            count_threshold: int,  连续出现的次数阈值，默认7 ， 即连续7帧出现的扰动中心才算有效， 具体需要根据实际情况调整

        Returns:
            Tuple[List, List]:
            - tracked_list: List, 经过追踪的 (x, y, id) 列表，每个时刻的点都有id , 每个(x,y)都代表点的索引值
            - reaLonLat_list: List, 经过追踪的 (lon, lat, id) 列表，每个时刻的点都有id， 每个(lon,lat)都代表点的真实经纬度
        """

        tracked_list = []  # 存储每个时刻的 (x, y, id)
        next_id = 0  # 全局 ID 计数器
        for i, current_points in enumerate(all_disturbance_list):
            current_tracked = []  # 当前时刻的 (x, y, id)

            if i == 0:
                # 第一帧：每个点分配新 ID
                for point in current_points:
                    current_tracked.append((point[0], point[1], next_id))
                    next_id += 1
            else:
                prev_tracked = tracked_list[i - 1]  # 上一帧的 (x, y, id)
                used_ids: Set[int] = set()  # 记录已被匹配的 ID，防止重复使用

                for point in current_points:
                    best_match_id = None
                    best_overlap_ratio = 0.0

                    # 尝试匹配上一帧的所有点
                    for px, py, pid in prev_tracked:
                        onoff, overlap_ratio = AddressData.calculate_overlap(
                            center_point1=(px, py),
                            center_point2=point,
                            wOfhalf=wOfhalf,
                            hOfhalf=hOfhalf,
                            lons = self.lons,
                            lats = self.lats,
                            ration_threshold=overlap_thresh  # 注意：这里传入的是判断阈值
                        )

                        if onoff and overlap_ratio > best_overlap_ratio:
                            best_overlap_ratio = overlap_ratio
                            best_match_id = pid

                    if best_match_id is not None and best_match_id not in used_ids:
                        # 成功匹配，复用 ID
                        current_tracked.append((point[0], point[1], best_match_id))
                        used_ids.add(best_match_id)
                    else:
                        # 无匹配，分配新 ID
                        current_tracked.append((point[0], point[1], next_id))
                        next_id += 1

            tracked_list.append(current_tracked)

        # 对时间进行筛选， 也就是每个id呈现的次数
        id_count = defaultdict(int)  # id -> 出现次数
        for frame in tracked_list:
            for x, y, id_ in frame:
                id_count[id_] += 1

        valid_ids = {id_ for id_, count in id_count.items() if count >= count_threshold}

        print(f"原始 ID 数量: {len(id_count)}")
        print(f"保留的长寿命 ID 数量 (>=7帧): {len(valid_ids)}")
        print(f"保留的 ID: {sorted(valid_ids)}")

        # 获取最终的追踪列表
        filtered_tracked_list = []
        reaLonLat_list = []
        for frame in tracked_list:
            filtered_frame = []
            i_frame_real_lonlatlist = []
            for x, y, id_ in frame:
                if id_ in valid_ids:
                    filtered_frame.append((x, y, id_))
                    i_frame_real_lonlatlist.append((self.lons[x].item(), self.lats[y].item(), id_))
            filtered_tracked_list.append(filtered_frame)
            reaLonLat_list.append(i_frame_real_lonlatlist)
        tracked_list = filtered_tracked_list
        return tracked_list, reaLonLat_list


    # 区分发展扰动(TC)  和 不发展扰动点
    @staticmethod
    def judegeDevOrNonDev(realLonLat_list: List[List[Tuple[float, float, Any]]],
                          all_t_TCPoints:List[List[Tuple[float, float, Any]]],
                          wOfBox_degree: int = 5,  # 单位: 度 , 左右伸展也就是 10 度了
                          hOfBox_degree: int = 5):
        """
        将发展扰动和不发展扰动点分开， 并返回两个列表， 一个是不发展扰动点的中心列表， 一个是发展扰动点的中心列表
        Argus:
            realLonLat_list: List, 经过追踪的 (lon, lat, id) 列表，每个时刻的点都有id， 每个(lon,lat)都代表点的真实经纬度
            all_t_TCPoints: List, 所有时刻的发展扰动点的中心列表， 每个时刻的点都有id， 每个(x,y)都代表点的索引值
            wOfBox_degree: int,  扰动点的半径阈值，默认10度 , 经度跨度为10度
            hOfBox_degree: int,  扰动点的半径阈值，默认10度 ， 纬度跨度为10度

        Returns:
            Tuple[List[List[Tuple[float, float, Any]]], List[List[Tuple[float, float, Any]]]:
            - final_nondev_centers: List, [[(lon, lat, id)], ...] 经过筛选的不发展扰动点的中心列表
            - final_dev_centers: List,  [[(lon, lat, id)], ...]  经过筛选的发展扰动点的中心列表
        """

        t_length = len(realLonLat_list)  # 时刻数量
        # realLonLat_list 的所有元素, 以tuple中的Any也就是id 为key , 其他两个float作为value
        track_PtsById = defaultdict(list)
        tc_PtsById = defaultdict(list)
        for frame in realLonLat_list:
            for x, y, id_ in frame:
                track_PtsById[id_].append((x, y))
        for frame in all_t_TCPoints:
            for x, y, id_ in frame:
                tc_PtsById[id_].append((x, y))
        print("最初的track_PtsByIds : ")
        print(track_PtsById.keys())
        # 对每个时刻的所有的点进行判断，判断是否是发展扰动还是不发展扰动
        for i_t in range(t_length):
            # 某个时刻具体的点
            i_t_tracked_list = realLonLat_list[i_t]
            i_t_all_t_TCPoints = all_t_TCPoints[i_t]

            for i_pt in i_t_tracked_list:
                x_pt, y_pt, id_pt = i_pt
                # 以该点 为中心， 锁定 w, h 范围
                """ 注意: 此处假定两个列表的点都代表经纬度, 而不是索引值 """
                box = {
                    "lon_min": x_pt - wOfBox_degree,
                    "lon_max": x_pt + wOfBox_degree,
                    "lat_min": y_pt - hOfBox_degree,
                    "lat_max": y_pt + hOfBox_degree,
                }

                # 查看范围内是否有 tc 点
                for i_tc_point in i_t_all_t_TCPoints:
                    x_tc_pt, y_tc_pt, sid = i_tc_point
                    if sid in tc_PtsById.keys():
                        if (x_tc_pt >= box["lon_min"]) and \
                                (x_tc_pt <= box["lon_max"]) and \
                                (y_tc_pt >= box["lat_min"]) and \
                                (y_tc_pt <= box["lat_max"]):
                            del tc_PtsById[sid]  # 删除键值对
                            del track_PtsById[id_pt]

        print("最终的track_PtsById.keys():")
        print(track_PtsById.keys())
        # track_PtsById剩下的 key 就是不发展扰动的点, 删除的就是发展扰动的点

        final_nondev_centers = []
        final_dev_centers = []
        for i_t in range(t_length):
            i_t_nondev_centers = []
            i_t_dev_centers = []

            i_t_tracked_list = realLonLat_list[i_t]
            # i_t_all_t_TCPoints = all_t_TCPoints[i_t]
            for i_track_point in i_t_tracked_list:
                x, y, id = i_track_point
                if id in track_PtsById.keys():
                    # 扰动点
                    i_t_nondev_centers.append((x, y, id))
                else:
                    # 非扰动点
                    i_t_dev_centers.append((x, y, id))  # 后面得想办法把 sid 加进来, 可能得修改前面的代码
            final_nondev_centers.append(i_t_nondev_centers)
            final_dev_centers.append(i_t_dev_centers)
        return final_nondev_centers, final_dev_centers


