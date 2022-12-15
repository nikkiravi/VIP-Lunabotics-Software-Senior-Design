import cv2
import matplotlib.pyplot as plt
import numpy as np
import json
import os
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

# Global Variables
TARGET_H, TARGET_W = 500, 500
bridge = CvBridge()

def get_rotation_matrix():
    R = np.array([[0.999996, 0.000563499, 0.0025959, 1], 
                    [-0.000570318, 0.999996, 0.00262663, 1],
                    [-0.00259441, -0.00262811, 0.999993, 1],
                    [1,1,1,1]])

    return R

def get_translational_matrix():
    T = np.identity(4)
    t = np.array([-0.0589796006679535, 3.27704656228889e-05, 0.00028060661861673])
    T[:3, 3] = t[:3]
    return T


def load_parameters(file):
    """
    You can obtain the extrinsics and intrinsics of a camera using the command below in a terminal (such as the Ubuntu terminal or the Windows command prompt).

    rs-enumerate-devices -c

    This command will generate a long list of extrinsic and intrinsic data.  You can then scroll up through the listing in the terminal until you reach the sections titled Extrinsic from "Color" to "Depth" and Extrinsic from "Depth" to "Color"
    
    url: https://support.intelrealsense.com/hc/en-us/community/posts/1500000652922-Use-millimeter-as-unit-for-extrinsic-parameter-between-depth-camera-and-color-camera
    
    """

    with open(file, "rt") as fptr:
        contents = json.load(fptr)

    fx, fy = contents["fx"], contents["fy"]
    u0, v0 = contents["u0"], contents["v0"]

    K = np.array([[fx, 0, u0, 0] , [0, fy, v0, 0], 
    [0, 0, 1, 0], [0, 0, 0, 1]]) # Intrinsic Parameters

    # Extrinsic Parameters
    R = get_rotation_matrix()
    t = get_translational_matrix()

    Rt = np.array([[0., -1., 0., 0.],
                  [0., 0., -1., 0.],
                  [1., 0., 0., 0.],
                  [0., 0., 0., 1.]]) # Rotate to camera coordinates

    Rt = np.matmul(np.matmul(Rt, R), t)
    return Rt, K


def callback(message):
    image = bridge.imgmsg_to_cv2(message, desired_encoding='rgb8') # shape: 480 x 640 x 3
    Rt, K = load_parameters(r"/home/parallels/catkin_ws/src/cv/scripts/RGBd/camera.json")
    



if __name__ == "__main__":
    # Code is inspired by https://github.com/darylclimb/cvml_project/blob/master/projections/ipm/ipm.py
    # Code is inspired by https://github.com/darylclimb/cvml_project/blob/cb06850b9477550b9c9e3c5651d013b347cc9b1b/projections/ipm/utils.py#L36

    rospy.init_node("ipm", anonymous=True)
    rospy.loginfo("Node has been initialized")

    rospy.Subscriber("/camera/color/image_raw", data_class=Image, callback=callback)
    rospy.loginfo("Node started")

    rospy.spin()