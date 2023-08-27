import rospy
from time import sleep
from math import degrees
import math
import numpy as np

from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from sensor_msgs.msg import PointCloud
from tf.transformations import euler_from_quaternion
from laser_pc import laser_to_points2d
import tf


class Navigation():
    def __init__(self):
        self.r = rospy.Rate(50)
        self.points2d=laser_to_points2d()
        self.listener = tf.TransformListener()
        self.cmd_vel = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.laserSub = rospy.Subscriber('/scan', LaserScan, self.laserCallback)
        rospy.Subscriber('/odom', Odometry, self.odometryCallback)
        sleep(0.5)

    def odometryCallback(self, msg):
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        rot_q = msg.pose.pose.orientation

        (_, _, self.theta) = euler_from_quaternion([rot_q.x, rot_q.y, rot_q.z, rot_q.w])

    def returnOdometry(self):

        return self.x, self.y, self.theta

    def laserCallback(self, msg):
        self.ranges = msg.ranges

        self.laser = self.ranges[270:359:18] + self.ranges[0:90:18]

        self.points2d.py,self.points2d.px = [],[]

        angle_min=msg.angle_min
        angle_max=msg.angle_max
        angle_increment=msg.angle_increment
        range_min=msg.range_min
        range_max=msg.range_max

        self.points2d.update(self.ranges,angle_min,angle_max,angle_increment)
        self.listener.waitForTransform("/base_footprint", "/odom", rospy.Time(0),rospy.Duration(1.0))
        self.p = self.listener.transformPointCloud("odom", self.points2d.base_link_point2d)

        #self.pointCloud.publish([self.p.header, self.p.points, self.p.channels], PointCloud)

        self.r.sleep()

    def publish(self, v, r):
        speed = Twist()
        speed.linear.x = v
        speed.angular.z = r

        self.cmd_vel.publish(speed)
        self.r.sleep()

    def checkForObstacle(self, x, y, threshold=0.1):
        x= float(x)
        y= float(y)
        points = self.p.points
        for point in points:
            if math.isnan(point.x) or math.isnan(point.y):
                continue
            _x = point.x
            _y = point.y

            if _x - threshold <= x and _x + threshold >= x:
                if _y - threshold <= y and _y + threshold >= y:
                    #print(point)
                    return True
        return False

    # def checkForObstacle(self):
    #     th = round(degrees(self.theta)) if self.theta >=0 else round(359 + degrees(self.theta))
    #     newRanges = list()
    #     newIdx = th
    #     for _ in range(0,359):
    #         if newIdx > 359:
    #             newIdx =0
    #         newRanges.append(self.ranges[newIdx])
    #         newIdx= newIdx+1
    #
    #     front    = newRanges[90]
    #     fr_right = newRanges[270]
    #     fr_left  = newRanges[45]
    #     right    = newRanges[180]
    #     left     = newRanges[0]
    #
    #     print(th)
    #     return front, fr_right, fr_left, right, left

    # def checkAll(self):
    #     front, fr_right, fr_left, right, left = self.checkForObstacle()
    #     print('Front: '+str(front))
    #     print('Front left: '+str(fr_right))
    #     print('Front right: '+str(fr_left))
    #     print('Right: '+str(right))
    #     print('Left: '+str(left))
    #
    # def printpc(self):
    #     sleep(1)
    #     print(self.p)
    #     rospy.spin()
