# This file is created using the 'package_generator',
# based on a configuration file.
# Do not manually edit this file!
import rclpy
from rclpy.qos import QoSPresetProfiles
from rclpy.executors import MultiThreadedExecutor

from geometry_msgs.msg import Twist
from sensor_msgs.msg import PointCloud2
from knowledge_base_interfaces.srv import GetDoorWay
from knowledge_base_interfaces.srv import GetSpotPosition
from uncertainty_pose_estimator_interfaces.srv import EstimateDrift
from uncertainty_pose_estimator_interfaces.srv import IdentifyDoorPose

from uncertainty_pose_estimator.utils.base_node import BaseNode


class UncertaintyPoseEstimatorBase(BaseNode):
    default_qos = QoSPresetProfiles.get_from_short_key('system_default')

    def __init__(self,
                 node_name='uncertainty_pose_estimator',
                 wait_for_servers=False,
                 **kwargs):
        super().__init__(node_name=node_name, **kwargs)

        publishers_dict = {}
        subscriptions_dict = {
            '/point_cloud': PointCloud2,
            '/floorplan_pose': Twist,
        }
        services_dict = {
            '/estimate_drift': EstimateDrift,
            '/identify_door_pose': IdentifyDoorPose,
        }
        service_clients_dict = {
            '/get_spot_position': GetSpotPosition,
            '/get_doorway': GetDoorWay,
        }
        actions_dict = {}
        action_clients_dict = {}

        self._publishers = self._create_publishers(
            publishers_dict)
        self._subscriptions = self._create_subscriptions(
            subscriptions_dict)
        self._services = self._create_services(
            services_dict)
        self._service_clients = self._create_service_clients(
            service_clients_dict, wait_for_servers)
        self._actions = self._create_actions(
            actions_dict)
        self._action_clients = self._create_action_clients(
            action_clients_dict, wait_for_servers)

        # Keep track of the goal handles
        self.goal_handles = {
            action_name: None for action_name in action_clients_dict.keys()}

    def _estimate_drift_callback(self, request, response):
        self.get_logger().info('Incoming estimate_drift request')
        return response

    def _identify_door_pose_callback(self, request, response):
        self.get_logger().info('Incoming identify_door_pose request')
        return response

    def _point_cloud_callback(self, msg):
        self.get_logger().info('Incoming _point_cloud msg')

    def _floorplan_pose_callback(self, msg):
        self.get_logger().info('Incoming _floorplan_pose msg')


def main(args=None):
    rclpy.init(args=args)

    executor = MultiThreadedExecutor()
    uncertaintyposeestimator = UncertaintyPoseEstimatorBase()
    rclpy.spin(uncertaintyposeestimator, executor=executor)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    uncertaintyposeestimator.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
