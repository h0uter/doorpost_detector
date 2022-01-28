import numpy as np
import os
import pytest

from door_post_pose_detector.usecases.detect_door_pipeline import cropped_pointcloud_to_door_post_poses_pipeline

# content of test_sample.py
def func2(x):
    return x + 1

def test_answer2():
    assert func2(3) == 4

def test_pointcloud1_response_poses():
    '''test 10 times if the accuracy is consistent'''
    acc = 1.0

    for i in range(10):
        two_up =  os.path.abspath(os.path.join(__file__ ,"../.."))
        full_path = os.path.join(two_up, 'data', 'door1_cropped_m0_8.npy')
        points = np.load(full_path)
        response = cropped_pointcloud_to_door_post_poses_pipeline(points, vis=0)
        assert response['poses'][0] == pytest.approx(0.9697678685188293, acc) or response['poses'][2] == pytest.approx(0.9697678685188293, acc)
        # assert response['poses'] == pytest.approx([0.9697678685188293, -0.3988088965415959, 0.923923909664154, 0.5727753043174739], [1., 1., 1.,1.])

def test_pointcloud1_response_success():
    # two_up =  os.path.abspath(os.path.join(__file__ ,"../.."))
    two_up =  os.path.abspath(os.path.join(__file__ ,"../.."))
    full_path = os.path.join(two_up, 'data', 'door1_cropped_m0_8.npy')
    points = np.load(full_path)
    response = cropped_pointcloud_to_door_post_poses_pipeline(points, vis=0)
    assert response['success'] == True

# def 