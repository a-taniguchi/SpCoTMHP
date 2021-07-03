#!/usr/bin/env python
#coding:utf-8
import numpy as np
import rospy
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Header
from nav_msgs.msg import Path
filename = "/root/HSR/catkin_ws/src/spconavi_ros/src/data/3LDK_01/navi/path_data.csv"
#filename = "/root/HSR/catkin_ws/src/spconavi_ros/src/data/3LDK_01/navi/T200N6A1S0G7_Path_ROS200_2.csv"

class Simple_path_simulator():

    def __init__(self):
        rospy.init_node('Simple_Path_Publisher')
        pub = rospy.Publisher("/omni_path_follower/path", Path, queue_size=50)
        #pub = rospy.Publisher("/move_base/DWAPlannerROS/global_plan", Path, queue_size=50)
        self.r = rospy.Rate(50)  # 50hz
        #Initialize odometry header
        self.path_header = Header()
        self.path_header.seq = 0
        self.path_header.stamp = rospy.Time.now()
        self.path_header.frame_id = "map"

        self.path = Path()
        self.path.header = self.path_header

        #get pose data from csv
        #self.csv_path_data = pd.read_csv(filename)
        self.csv_path_data = np.loadtxt(filename, delimiter=",")
        pose_list = self.get_poses_from_csvdata()
        self.path.poses =pose_list
        #initialize publisher
        self.path_pub = rospy.Publisher("/omni_path_follower/path", Path, queue_size=50)
        #self.path_pub = rospy.Publisher("/move_base/DWAPlannerROS/global_plan", Path, queue_size=50)

    def get_poses_from_csvdata(self):
        #Get poses from csv data
        poses_list = []
        print(self.csv_path_data)
        for indx in range(len(self.csv_path_data)):
            #print(indx)

            temp_pose = PoseStamped()
            temp_pose.pose.position.x = self.csv_path_data[indx][1]
            temp_pose.pose.position.y = self.csv_path_data[indx][2]

            """
            temp_pose.pose.position.x = self.csv_path_data[indx][1]
            temp_pose.pose.position.y = self.csv_path_data[indx][2]
            temp_pose.pose.position.z = self.csv_path_data[indx][3]
            temp_pose.pose.orientation.x = self.csv_path_data[indx][4]
            temp_pose.pose.orientation.y = self.csv_path_data[indx][5]
            temp_pose.pose.orientation.z = self.csv_path_data[indx][6]
            temp_pose.pose.orientation.w = self.csv_path_data[indx][7]
            temp_pose.header = self.path_header
            temp_pose.header.seq = indx
            """
            poses_list.append(temp_pose)
        return poses_list

    
    def publish_path_topic(self):
        self.path_pub.publish(self.path)
        self.r.sleep()



if __name__ == '__main__':
    print('Path Publisher is Started...')
    test = Simple_path_simulator()
    try:
        while not rospy.is_shutdown():
            test.publish_path_topic()
            print('Success!!!!')
    except KeyboardInterrupt:
        print("finished!")

