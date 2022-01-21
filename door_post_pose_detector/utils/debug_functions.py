from geometry_msgs.msg import Pose2D

class DebugKnowledgeBaseResponse():
    def __init__(self) -> None:
        self.lin_x = 10
        self.lin_y = 3
        self.rot_z = 0
        self.success = True

#debug
def debug_get_door_pose_kb(door_name):
    # TODO: replace the debug response with the code read the contents of the cb message
    debug_kb_response = DebugKnowledgeBaseResponse()

    # convert your response to a Pose2D
    response = Pose2D()
    response.x = debug_kb_response.lin_x
    response.y = debug_kb_response.lin_y
    response.theta = debug_kb_response.rot_z
    
    return response


def debug_estimate_drift(detected_door, expected_door):
    '''
    spoof drift estimation based on 2 poses
    '''
    drift_estimate = Pose2D()
    drift_estimate.x = 10.0
    drift_estimate.y = 3.0
    drift_estimate.theta = 0.0

    return drift_estimate

