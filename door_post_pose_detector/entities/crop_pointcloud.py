import numpy.typing as npt


def crop_pointcloud(pc_array: npt.NDArray, crop_target: tuple, crop_margin: tuple):

    cropped_pc = [
        point
        for point in pc_array
        if crop_target[0] - crop_margin <= point[0] <= crop_target[0] + crop_margin
        and crop_target[1] - crop_margin <= point[1] <= crop_target[1] + crop_margin
    ]

    return cropped_pc
