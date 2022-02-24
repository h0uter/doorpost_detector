# doorpost_detector
This package contains a pointcloud processing pipeline to estimate the poses of doorposts.


## Quick start
`pip install -r requirements.txt`

`pip install -e .`


## To obtain doorpost poses from cropped pointcloud
```python
import doorpost_detector.api as dpd

response = dpd.doorpost_pose_from_cropped_pointcloud_usecase(points)

```


## To obtain doorpost poses from pointcloud
```python
import doorpost_detector.api as dpd

response = dpd.doorpost_pose_from_pointcloud_and_door_location_estimate_usecase(points, door_location)

```


## Response format
```python
@dataclass
class Response:
    success: bool
    poses: tuple[float, float, float, float]
    certainty: tuple[float, float]
```
