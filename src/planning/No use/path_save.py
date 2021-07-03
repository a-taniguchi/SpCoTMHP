#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import rospy
from geometry_msgs.msg import Point, Pose, Quaternion, Twist, Vector3
from nav_msgs.msg import Odometry

global_i = 0

class Odom_saver():

    def __init__(self):
        rospy.init_node("odomsaver")
        self.r = rospy.Rate(50)  # 50hz
        self.save_path_as_csv = True
        self.sim_odom = Odometry()

        #self.sub= rospy.Subscriber("/hsrb/odom", Odometry, self.callback)

        
        self.path_dict = {}


    def callback(self, message):
        global global_i
        print(type(message))
        # print(type(self.sim_odom))
        rospy.loginfo("odom : x %lf  : y %lf\n", message.pose.pose.position.x, message.pose.pose.position.y)

        addRow = [0, message.pose.pose.position.x, message.pose.pose.position.y, 0, 
                message.pose.pose.orientation.x, message.pose.pose.orientation.y, message.pose.pose.orientation.z, message.pose.pose.orientation.w,
                message.twist.twist.linear.x, message.twist.twist.linear.y, message.twist.twist.linear.z, 0, 0, message.twist.twist.angular.z]                     
        self.path_dict[len(self.path_dict)] = addRow
        print(len(self.path_dict))
        global_i = global_i + 1

        if global_i == 1000:
            cols = ["time", "x", "y", "z", "w0", "w1", "w2", "w3", "vx", "vy", "vz", "roll", "pitch", "yaw"]
            df = pd.DataFrame.from_dict(self.path_dict, orient='index',columns=cols)
            df.to_csv("path_data.csv", index=False)


    def save_csv(self):
        # Save CSV path file
        cols = ["time", "x", "y", "z", "w0", "w1", "w2", "w3", "vx", "vy", "vz", "roll", "pitch", "yaw"]
        df = pd.DataFrame.from_dict(self.path_dict, orient='index',columns=cols)
        df.to_csv("path_data.csv", index=False)


if __name__ == '__main__':
    print('Odom Saver is Started...')
    test = Odom_saver()
    sub = rospy.Subscriber("/odom", Odometry, test.callback)
    #sub = rospy.Subscriber("/hsrb/odom", Odometry, test.callback)
    rospy.spin()
    #i=0
    #while i < 10:
    #    sub= rospy.Subscriber("/hsrb/odom", Odometry, test.callback)
    #    print(sub)
    #    i = i+1
    #test.save_csv()

     

    
    
    
