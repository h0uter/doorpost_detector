from geometry_msgs.msg import Pose2D
from sensor_msgs.msg import PointCloud2
from sensor_msgs.msg import PointCloud

import uncertainty_pose_estimator.ros2_numpy as rnp

import numpy as np

def pc_msg2npy(pc_msg):
    XYZ_array = []
    for i in range(len(pc_msg.points)):
        x = pc_msg.points[i].x
        y = pc_msg.points[i].y
        z = pc_msg.points[i].z
        XYZ_array.append([x,y,z])

    return XYZ_array

def npy2pc_msg(msg, npy):

    pc_msg = rnp.msgify(PointCloud2, npy)

    # pc_msg = msg
    # pc_msg.points = pc_msg.points[0:len(npy)]

    # print(len(npy))

    # for i in range(len(npy)):
        # pc_msg.points[i].x = npy[i][0]
        # pc_msg.points[i].y = npy[i][1]
        # pc_msg.points[i].z = npy[i][2]
    return pc_msg

def crop_pointcloud(pc_msg, crop_target, crop_margin):
    if pc_msg == None:
        pc_msg = PointCloud2()
    
    # pc = pc_msg2npy(pc_msg)
    pc = rnp.numpify(pc_msg)

    pc_array = np.array([np.array(list(t)) for t in pc ])

    cropped_pc = []

    for point in pc_array:
        if crop_target.x - crop_margin <= point[0] <= crop_target.x + crop_margin and  crop_target.y - crop_margin <= point[1] <= crop_target.y + crop_margin:
            cropped_pc.append(point)

  # unComment to return as ROS msg
  # cropped_pc_msg = npy2pc_msg(pc_msg, cropped_pc)
  # return cropped_pc_msg 
    return cropped_pc


def door_info2door_pose(door_info):
    door_pose = Pose2D()
    # print(door_info.first_post_lin_x)
    #   print(door_info.first_post_lin_y)

    # HACK: calc door pose from two points instead of taking first post
    door_pose.x = door_info.first_post_lin_x
    door_pose.y = door_info.first_post_lin_y 
    # TODO: calc the door pose angle from the above.
    door_pose.theta = 0.0

    return door_pose