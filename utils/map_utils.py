import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

from shapely.geometry import LineString, Point, Polygon
from shapely.affinity import affine_transform, rotate
# from utils.cubic_spline_planner import Spline2D

def wrap_to_pi(theta):
    return (theta+np.pi) % (2*np.pi) - np.pi
    
def get_polylines(lines):
    polylines = {}

    for line in lines.keys():
        polyline = np.array([(map_point.x, map_point.y) for map_point in lines[line].polyline])
        if len(polyline) > 1:
            direction = wrap_to_pi(np.arctan2(polyline[1:, 1]-polyline[:-1, 1], polyline[1:, 0]-polyline[:-1, 0]))
            direction = np.insert(direction, -1, direction[-1])[:, np.newaxis]
        else:
            direction = np.array([0])[:, np.newaxis]
        polylines[line] = np.concatenate([polyline, direction], axis=-1)

    return polylines

from shapely.geometry import LineString, Point
# from shapely.ops import interpolate

def resample_centerline_Shapely(path, num_waypoints):
    # Create a LineString object from the path
    line = LineString(path)

    # Calculate the total length of the path and the expected distance between waypoints
    total_length = line.length
    expected_length = total_length / (num_waypoints - 1)

    # Calculate the expected positions of the waypoints along the path
    expected_points = [line.interpolate(i * expected_length) for i in range(num_waypoints)]

    # Convert the expected points to a list of coordinates
    new_path = [[p.x, p.y] for p in expected_points]
    return np.array(new_path)

def resample_centerline_nonShapely(centerline, num_waypoints):
    # 选择出 centerline 中没有 0 的所有行
    # print(np.abs(centerline)>0.0)
    # # print(centerline)
    # centerline = centerline[np.all(np.abs(centerline)>0.0, axis=1)]
    # 计算原始路径中每个相邻 waypoint 之间的距离
    distances = np.cumsum(np.sqrt(np.sum(np.diff(centerline, axis=0)**2, axis=1)))
    distances = np.insert(distances, 0, 0)  # 添加起点距离为0

    # 计算整个路径的总长度，并计算出每个相邻 waypoint 之间的期望距离
    total_distance = distances[-1]
    # expected_distance = total_distance / (num_waypoints - 1)

    # 计算每个 waypoint 在新路径中的期望位置
    expected_distances = np.linspace(0, total_distance, num_waypoints)
    expected_positions = np.zeros((num_waypoints, 2))
    for i in range(num_waypoints):
        if i == 0:
            expected_positions[i] = centerline[0]
        elif i == num_waypoints-1:
            expected_positions[i] = centerline[-1]
        else:
            d = expected_distances[i]
            idx = np.searchsorted(distances, d)
            t = (d - distances[idx-1]) / (distances[idx] - distances[idx-1])
            expected_positions[i] = centerline[idx-1] + t * (centerline[idx] - centerline[idx-1])

    # 使用线性插值方法将期望位置之间的间隔填充，并返回新路径的 waypoint 列表
    interp_func = interp1d(expected_distances, expected_positions, axis=0)
    new_distances = np.linspace(0, total_distance, num_waypoints)
    new_positions = interp_func(new_distances)
    # print(np.all(centerline), np.all(new_positions))
    # if np.all(np.abs(centerline)<100):
    #     print('centerline', centerline)
    # if np.all(np.abs(new_positions)<600):
    #     print('new_position', new_positions)
    #     print('centerline', centerline)
    return new_positions