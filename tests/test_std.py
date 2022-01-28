import numpy as np
import os
import pytest

from door_post_pose_detector.usecases.detect_door_usecase import cropped_pointcloud_to_door_post_poses_usecase

# content of test_sample.py
def func2(x):
    return x + 1

def test_answer2():
    assert func2(3) == 4

def test_pointcloud1_response_poses():
    '''test 10 times if the accuracy is consistent'''

    acc_x = 0.2
    acc_y = 0.2
    responses = []
    poses = []

    for i in range(10):
        two_up =  os.path.abspath(os.path.join(__file__ ,"../.."))
        full_path = os.path.join(two_up, 'data', 'door1_cropped_m0_8.npy')
        points = np.load(full_path)
        response = cropped_pointcloud_to_door_post_poses_usecase(points, vis=0)
        print(f">>> trial {i}: success: {response['success']}, poses: {response['poses']}")
        responses.append(response)
        poses.append(response['poses'])
        
    print(f"variance in pose estimates {np.var(poses, axis=0)}")
    print(f"mean in pose estimates {np.mean(poses, axis=0)}")
    
    for response in responses:
        print(f"poses: {response['poses']}")
        assert response['poses'][0] == pytest.approx(1.92464301, abs=acc_x)
        assert response['poses'][1] == pytest.approx(-0.86583511, abs=acc_y)
        assert response['poses'][2] == pytest.approx(1.83260872, abs=acc_x)
        assert response['poses'][3] == pytest.approx(0.15936917, abs=acc_y)

def pointcloudN_response_poses(dataset_num, means, acc_x, acc_y):
    '''test 10 times if the accuracy is consistent'''

    responses = []
    poses = []

    for i in range(5):
        two_up =  os.path.abspath(os.path.join(__file__ ,"../.."))
        full_path = os.path.join(two_up, 'data', f'door{dataset_num}_cropped_m0_8.npy')
        points = np.load(full_path)
        response = cropped_pointcloud_to_door_post_poses_usecase(points, vis=0)
        print(f">>> trial {i}: success: {response['success']}, poses: {response['poses']}")
        responses.append(response)
        poses.append(response['poses'])
        
    print(f"variance in pose estimates {np.var(poses, axis=0)}")
    print(f"mean in pose estimates {np.mean(poses, axis=0)}")
    
    for response in responses:
        print(f"poses: {response['poses']}")
        assert response['poses'][0] == pytest.approx(means[0], abs=acc_x)
        assert response['poses'][1] == pytest.approx(means[1], abs=acc_y)
        assert response['poses'][2] == pytest.approx(means[2], abs=acc_x)
        assert response['poses'][3] == pytest.approx(means[3], abs=acc_y)

def test_pointcloud2_response_poses():
    pointcloudN_response_poses(2, [ 1.41358081, -0.09048086,  1.32636594,  0.85316349], 0.2, 0.2)

def test_pointcloud3_response_poses():
    pointcloudN_response_poses(3, [ 1.63835107, -0.76956589,  1.79067968,  0.15064277], 0.2, 0.2)

def test_pointcloud4_response_poses():
    pointcloudN_response_poses(4, [ 1.21417205, -0.70171573,  1.36259072,  0.20507376], 0.2, 0.2)

def test_pointcloud5_response_poses():
    pointcloudN_response_poses(5, [ 0.76884589,  1.17149341, -0.07714339,  1.39480131], 0.2, 0.2)

def test_pointcloud6_response_poses():
    pointcloudN_response_poses(6, [ 1.21417205, -0.70171573,  1.36259072,  0.20507376], 0.2, 0.2)

def test_pointcloud7_response_poses():
    pointcloudN_response_poses(7, [ 1.21417205, -0.70171573,  1.36259072,  0.20507376], 0.2, 0.2)

def test_pointcloud8_response_poses():
    pointcloudN_response_poses(8, [ 1.21417205, -0.70171573,  1.36259072,  0.20507376], 0.2, 0.2)


def test_pointcloud1_response_success():
    # two_up =  os.path.abspath(os.path.join(__file__ ,"../.."))
    two_up =  os.path.abspath(os.path.join(__file__ ,"../.."))
    full_path = os.path.join(two_up, 'data', 'door1_cropped_m0_8.npy')
    points = np.load(full_path)
    response = cropped_pointcloud_to_door_post_poses_usecase(points, vis=0)
    assert response['success'] == True

# def 

if __name__ == "__main__":
    # test_pointcloud1_response_success()
    test_pointcloud5_response_poses()