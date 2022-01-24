from door_post_pose_detector.usecases.detect_door_pipeline import cropped_pointcloud_to_door_post_poses_pipeline
import numpy as np
import os


def test_run():
    # points = np.load('data/robot_cropped_pointcloud.npy')
    path = os.path.abspath(".")

    for i in range(0, 7):
        points = None
        if i == 0:
            points = np.load(f'{path}/data/door0_cropped.npy')
        else:
            # crop margin 0.8m works decent
            points = np.load(f'{path}/data/door{i}_cropped_m0_8.npy')
            # crop margin 1.5m has lots of problems
            # points = np.load(f'data/door{i}_cropped_m1_5.npy')
        
        response = cropped_pointcloud_to_door_post_poses_pipeline(points, vis=1)
        print(f">>> dataset {i}: success: {response['success']}, poses: {response['poses']}")



def test_pc1_success():
    path = os.path.abspath(".")
    full_path = os.path.join( path, 'data', 'door1_cropped_m0_8.npy')
    print(full_path)
    points = np.load(full_path)
    # points = np.load(f'{path}/data/door1_cropped_m0_8.npy')
    # points = np.load(f'../../data/door1_cropped_m0_8.npy')
    response = cropped_pointcloud_to_door_post_poses_pipeline(points, vis=0)
    assert response['success'] == True


if __name__ == "__main__":
    # test_pc1_success()
    test_run()
