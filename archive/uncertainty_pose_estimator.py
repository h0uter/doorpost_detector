from typing import SupportsAbs
from .uncertainty_pose_estimator_base import UncertaintyPoseEstimatorBase
import rclpy
from rclpy.executors import MultiThreadedExecutor

from geometry_msgs.msg import Pose2D

from uncertainty_pose_estimator.utils.utils import crop_pointcloud, door_info2door_pose
from uncertainty_pose_estimator.utils.debug_functions import debug_estimate_drift, debug_get_door_pose_kb
from uncertainty_pose_estimator.door_detector.script_tara import door_post_pose_detector
from uncertainty_pose_estimator.detect_door import process_pointcloud_to_door_post_poses
from uncertainty_pose_estimator.utils.o3d_arrow import *

import numpy as np
import open3d as o3d


class UncertaintyPoseEstimator(UncertaintyPoseEstimatorBase):
    def __init__(self):
        super().__init__()
        self.pointcloud = None
        self.crop_margin = 0.8

    def _point_cloud_callback(self, msg):
        '''store the most recent pointcloud'''
        self.pointcloud = msg

    async def _estimate_drift_callback(self, request, response):
        ''' 
        When a request to estimate the drift is received,
        run the estimation pipeline and return the estimated drift
        '''
        self.get_logger().info('Incoming request\ndoorname: %s ' % (request.door_name))

        pc = self.pointcloud # the most recent pointcloud is saved in self.pointcloud.

        spot_pos_info = await self.call_service_async(service_name="/get_spot_position")
        door_info = await self.call_service_async(
            service_name="/get_doorway", 
            building="Villa", 
            door=request.door_name, 
            room=spot_pos_info.room[0]
            )
        
        expected_door_pose = door_info2door_pose(door_info)
        self.get_logger().info(f"expected_door_pose: {expected_door_pose}")

        # TODO: convert the frames of the doorpost from KB to global frame
        expected_door_pose.y = 0.1 # HACK: where door post is is relative to hardcoded start
        expected_door_pose.x = 0.9 # HACK: where door post is is relative to hardcoded start
        self.get_logger().info(f" HACKED expected_door_pose: {expected_door_pose}")

        # crop the pointcloud around the expected door pose pose retrieved from the knowledgebase
        cropped_pc_npy = crop_pointcloud(
            pc, 
            expected_door_pose, 
            self.crop_margin
            )

        def npy2pcd(points):
            point_cloud = o3d.geometry.PointCloud() # instantiate point cloud
            point_cloud.points = o3d.utility.Vector3dVector(points) # fill pointcloud with numpy points
            # TODO: make this respect the vis setting
            # o3d.visualization.draw_geometries([point_cloud])
            return point_cloud

        FOR = get_o3d_FOR()
        o3d_pc = npy2pcd(cropped_pc_npy)
        o3d.visualization.draw_geometries([FOR, o3d_pc])

        pipeline_response = process_pointcloud_to_door_post_poses(cropped_pc_npy)
        self.get_logger().info(f"pipeline response: {pipeline_response}")

        # TODO: get the door pose instead of the first door post pose
        if pipeline_response['success']:
            detected_door_info = {
                'first_post_lin_x': pipeline_response.poses[0],
                'first_post_lin_y': pipeline_response.poses[1],
            }

            detected_door_pose = door_info2door_pose(detected_door_info)
            # use the detected door pose to estimate the drift.
            # calculate the actual drift estimation
            response = self.estimate_drift(detected_door_pose, expected_door_pose)        
            self.get_logger().info(f'response.succes: {response["success"]}')

        else:  
            self.get_logger().info(f'FAILURE')

        self.get_logger().info(f'response: {response["success"]}')
        return response

    def estimate_drift(self, detected_door_pose, expected_door_pose):
        '''
        estimate the actual drift based on the discrepancy between
        the expected door pose and the detected door pose.
        '''
        drift_estimate = Pose2D()

        drift_estimate.x = expected_door_pose.x - detected_door_pose.x
        drift_estimate.y = expected_door_pose.y - detected_door_pose.y
        drift_estimate.theta = expected_door_pose.theta - detected_door_pose.theta

        print("the door has been detected at [x,y,theta]: [", detected_door_pose.x, ", ",
              detected_door_pose.y, ", ", detected_door_pose.theta, "]")

        print("the door is in the knowledge base at [x,y,theta]: [", expected_door_pose.x, ", ",
              expected_door_pose.y, ", ", expected_door_pose.theta, "]")

        print("the estimated drift is [x,y,theta]: [", drift_estimate.x, ", ",
              drift_estimate.y, ", ", drift_estimate.theta, "]")

        return drift_estimate

    def _identify_door_pose_callback(self, request, response):
        ''' 
        When a request to identify a new door is recieved,
        the pose of the new door is identified and returned.
        '''
        response.door_post_poses = door_post_pose_detector()
        self.get_logger().info('Incoming identify_door request')

        return response

def main(args=None):
    rclpy.init(args=args)
    executor = MultiThreadedExecutor()
    uncertainty_pose_estimator = UncertaintyPoseEstimator()
    rclpy.spin(uncertainty_pose_estimator, executor=executor)
    uncertainty_pose_estimator.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
