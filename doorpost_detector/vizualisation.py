# ENTRYPOINT
import open3d as o3d
import matplotlib.pyplot as plt
import numpy as np

from doorpost_detector.utils.o3d_arrow import (
    draw_geometries,
    get_o3d_FOR,
    get_arrow,
)


### VISUALIZATION ###
def set_axes_equal(ax: plt.Axes):
    """Set 3D plot axes to equal scale.

    Make axes of 3D plot have equal scale so that spheres appear as
    spheres and cubes as cubes.  Required since `ax.axis('equal')`
    and `ax.set_aspect('equal')` don't work on 3D.
    """

    def _set_axes_radius(ax: plt.Axes, origin: tuple, radius: float) -> None:
        x, y, z = origin
        ax.set_xlim3d([x - radius, x + radius])
        ax.set_ylim3d([y - radius, y + radius])
        ax.set_zlim3d([z - radius, z + radius])

    limits = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d(),])
    origin = np.mean(limits, axis=1)
    radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
    _set_axes_radius(ax, origin, radius)


def plot_points(points, best_inliers: list[int], outliers: list) -> None:
    # TODO: update to not take the points argument
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        points[best_inliers][:, 0],
        points[best_inliers][:, 1],
        points[best_inliers][:, 2],
        c="b",
        marker="o",
        label="Inlier data",
    )
    ax.scatter(
        points[outliers][:, 0],
        points[outliers][:, 1],
        points[outliers][:, 2],
        c="r",
        marker="o",
        label="Outlier data",
    )
    ax.legend(loc="lower left")
    ax.set_box_aspect([1, 1, 1])
    set_axes_equal(ax)
    plt.show()


def display_inlier_outlier(cloud: o3d.geometry.PointCloud, ind: list) -> None:
    """displays the result of the o3d outlier removal"""
    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)

    print("Showing outliers (red) and inliers (gray): ")
    outlier_cloud.paint_uniform_color([1, 0, 0])
    inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    # TODO: make this respect the vis setting
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])


def display_end_result(
    best_fit_doorpost_a, best_fit_doorpost_b, post_vectors, pointcloud_orig
):
    # cleanup this plotting mess
    FOR = get_o3d_FOR()
    # prevent nonetype from fucking up
    if best_fit_doorpost_a:
        xa, ya = best_fit_doorpost_a
        arrow_a = get_arrow([xa, ya, 0.01], vec=post_vectors[0])

        if best_fit_doorpost_b:
            xb, yb = best_fit_doorpost_b
            arrow_b = get_arrow([xb, yb, 0.01], vec=post_vectors[1])
            draw_geometries([FOR, pointcloud_orig, arrow_a, arrow_b])
        else:
            draw_geometries([FOR, pointcloud_orig, arrow_a])
