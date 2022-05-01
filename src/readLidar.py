import json
import numpy as np
from absl import app
from pylot.drivers.sensor_setup import CameraSetup, LidarSetup
from pylot.perception.depth_frame import DepthFrame
from pylot.perception.point_cloud import PointCloud
from pylot.utils import Location, Rotation, Transform, Vector2D


def readLidar(pointsFile, sensorConfig):
    f = open (pointsFile, "r")
    data = json.loads(f.read())
    f.close()
    points = np.array([[d[0],d[1],d[2]] for d in data[0]])
    f = open (sensorConfig, "r")
    data = json.loads(f.read())
    name = 'lidar'
    lidar_type = 'sensor.lidar.ray_cast'
    transform =  Transform(Location(), Rotation())
    range_ = 5000.0
    rotation_frequency = 20.0
    channels = 32
    upper_fov = 15.0
    lower_fov = -30.0
    points_per_second = 500000
    legacy = True
    for d in data:
        if(d['name'].lower() == 'lidar'):
            name = d['name']
            lidar_type = d['type'] if d['type'].lower() == 'sensor.lidar.ray_cast' or d['type'].lower() == 'velodyne' else 'sensor.lidar.ray_cast'
            if('transform'in  d):
                transform = Transform(location=Location(d['transform'][0], d['transform'][1], d['transform'][2]), rotation=Rotation(0, 0, 0)) if not isinstance(d['transform'][0], list) \
                else Transform(location=Location(d['transform'][0][0], d['transform'][0][1], d['transform'][0][2]), rotation=Rotation(d['transform'][1][0], d['transform'][1][1], d['transform'][1][2]))
            if('RANGE' in  d['settings']):
                range_ = d['settings']['RANGE']
            if('ROTATION_FREQUENCY' in  d['settings']):
                rotation_frequency = d['settings']['ROTATION_FREQUENCY']
            if('CHANNELS' in  d['settings']):
                channels = d['settings']['CHANNELS']
            if('UPPER_FOV' in  d['settings']):
                upper_fov = d['settings']['UPPER_FOV']
            if('LOWER_FOV' in  d['settings']):
                lower_fov = d['settings']['LOWER_FOV']
            if('PPS' in  d['settings']):
                points_per_second = d['settings']['PPS']
            if('LEGACY' in  d['settings']):
                legacy = d['settings']['LEGACY']
    f.close()
    lidar_setup = LidarSetup(name,
                            lidar_type,
                            transform,
                            range_,
                            rotation_frequency,
                            channels,
                            upper_fov,
                            lower_fov,
                            points_per_second,
                            legacy)
    #print(points)
    return PointCloud(points, lidar_setup)


def main(args):
    print(readLidar('test.json', 'sensor_config.json'))


if __name__ == '__main__':
    app.run(main)
