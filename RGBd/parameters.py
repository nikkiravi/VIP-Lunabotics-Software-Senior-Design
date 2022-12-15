import cv2
import numpy as np
import os
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

### Global Variables
bridge = CvBridge()
CHECKERBOARD = (6, 9) # Dimensions of the Checkerboard
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001) # Termination Criteria

def display_corners(image, corners, ret):
    # Draw and display the corners
    cv2.drawChessboardCorners(image, CHECKERBOARD, corners, ret)
    cv2.imshow('img', image)
    cv2.waitKey(500)
    cv2.destroyAllWindows()

def display_image(image):
    cv2.imshow("window", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    quit()

def get_corners(image, gray, objp, display=False):
    # Find the chessboard corners
    # If desired number of corners are found in the image then ret = true
    # cv2.CALIB_CB_ADAPTIVE_THRESH = Use adaptive thresholding to convert the image to black and white, rather than a fixed threshold level (computed from the average image brightness)
    # cv2.CALIB_CB_FAST_CHECK = Run a fast check on the image that looks for chessboard corners, and shortcut the call if none is found. This can drastically speed up the call in the degenerate condition when no chessboard is observed.
    # cv2.CALIB_CB_NORMALIZE_IMAGE = Normalize the image gamma with equalizeHist before applying fixed or adaptive thresholding.
    ret, corners = cv2.findChessboardCorners(gray, patternSize=CHECKERBOARD, flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)

    # If desired number of corners are detected, we refine the pixel coordinates and display them on the checkerboard
    if(ret):
        corners2 = cv2.cornerSubPix(gray, corners=corners, winSize=(11, 11), zeroZone=(-1, -1), criteria=criteria) # refining pixel coordinates for given 2d points
        if(display):
            display_corners(image, corners2, ret)
        
        return objp, corners2

    else:
        return None, None

def get_instrinsic_extrinsic_parameters(gray, obj_points, img_points):
    """
    Performing camera calibration by 
    passing the value of known 3D points (objpoints)
    and corresponding pixel coordinates of the 
    detected corners (imgpoints)
    """
    # K = Intrinsic Matrix
    # dist = Lens distortion coefficients. These coefficients will be explained in a future post.
    # R = Rotation Vector
    # t = Translation Vector

    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, gray.shape[::-1], None, None)
    print(rvecs)
    print(tvecs)
    R, jacobian = cv2.Rodrigues(rvecs[0])
    t = tvecs[0]
    return K, R, t

def ipm(images, display=False):
    # Arrays to store object points and image points from all the images.
    obj_points = [] # 3d point in real world space
    img_points = [] # 2d points in image plane

    # Defining the world coordinates for 3D points
    objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), dtype=np.float32)
    objp[0, :, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2) # Create a multi-dimensional meshgrid
    gray = None

    for image in images:
        # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        height, width, channels = image.shape
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        obj_pts, img_pts = get_corners(image, gray, objp=objp, display=display)
        if(obj_points is not None and img_pts is not None):
            obj_points.append(obj_pts)
            img_points.append(img_pts)

    K, R, t = get_instrinsic_extrinsic_parameters(gray, obj_points=obj_points, img_points=img_points)
    return K, R, t


def callback(message):
    image = bridge.imgmsg_to_cv2(message, desired_encoding='bgr8') # shape: 480 x 640 x 3
    cv2.imwrite(r"/home/parallels/catkin_ws/src/cv/scripts/RGBd/dataset/7.png", image)
    rospy.loginfo("Image is loaded")



def image_from_camera():
    rospy.init_node("ipm", anonymous=True)
    rospy.loginfo("Node has been initialized")

    rospy.Subscriber("/camera/color/image_raw", data_class=Image, callback=callback)
    rospy.loginfo("Node started")

    rospy.spin()


if __name__ == "__main__":
    path = r"/home/parallels/catkin_ws/src/cv/scripts/RGBd/dataset"
    mode = "calibrate"

    if(mode == "rospy"):
        image_from_camera()
    else:
        images = [cv2.imread(path + "/" + file) for file in os.listdir(path)]
        K, R, t = ipm(images, display=False)
        np.savez_compressed("/home/parallels/catkin_ws/src/cv/scripts/RGBd/parameters", K=K, R=R, t=t)