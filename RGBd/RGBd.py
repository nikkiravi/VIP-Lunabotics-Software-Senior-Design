import cv2
import numpy as np
import open3d as o3d
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, PointCloud
from geometry_msgs.msg import Point32

### Global Variables
bridge = CvBridge()
publisher = rospy.Publisher("/reconstructed_point_cloud", PointCloud, queue_size=1)


def display_image(image):
	cv2.imshow("window", image)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	quit()

class RGBDepth:
    def __init__(self):
        # Define variables
        self.rgb_image = None
        self.depth_image = None
        self.rgbd_image = None
        self.geometry_depth = None
        self.geometry_rgb = None
        self.K = None
        self.R = None
        self.t = None
        self.Rt = None

    def ros_callback(self):
        rospy.init_node("rdb_d", anonymous=True)
        rospy.loginfo("Node has been initialized")

        # Subscribe to topic
        rospy.Subscriber("/camera/color/image_raw", data_class=Image, callback=rgbd.get_rgb_image)
        rospy.Subscriber("/camera/depth/image_rect_raw", data_class=Image, callback=rgbd.get_depth_image)
        rospy.sleep(2)

        point_cloud = self.display_point_cloud(display=True)

        # Publish to topic
        pc = PointCloud()
        point_cloud_points = []
        coord = Point32()

        for x, y, z in point_cloud.points:
            coord.x, coord.y, coord.z = x, y, z
            point_cloud_points.append(coord)

        pc.points = point_cloud_points
        publisher.publish(pc)
        rospy.loginfo("Published")  

        rospy.loginfo("Node started")
        rospy.spin()

    def get_rgb_image(self, message):
        self.rgb_image = bridge.imgmsg_to_cv2(message, desired_encoding='bgr8') # shape: 480 x 640 x 3
        self.geometry_rgb = o3d.geometry.Image(self.rgb_image)
        cv2.imwrite("/home/parallels/catkin_ws/src/cv/scripts/RGBd/rgb_img.png", self.rgb_image)
        
        rospy.loginfo("RGB Image is loaded")
    
    def get_depth_image(self, message):
        self.depth_image = np.asarray(np.frombuffer(message.data, dtype=np.uint16).reshape(message.height, message.width, -1))  # bridge.imgmsg_to_cv2(message, desired_encoding="passthrough")
        self.geometry_depth = o3d.geometry.Image(self.depth_image)
        cv2.imwrite("/home/parallels/catkin_ws/src/cv/scripts/RGBd/depth_img.png", self.depth_image)
        
        rospy.loginfo("Depth Image is loaded")

    def get_camera_parameters(self):
        parameters = np.load(r"/home/parallels/catkin_ws/src/cv/scripts/RGBd/parameters.npz")
        self.K, self.R, self.t = np.array(parameters["K"]), np.array(parameters["R"]), np.array(parameters["t"])

        self.K = np.column_stack((self.K, np.array([0,0,0])))
        self.K = np.row_stack((self.K, np.array([0,0,0,1])))

        self.Rt = np.column_stack((self.R, self.t))
        self.Rt = np.row_stack((self.Rt, np.array([0,0,0,1])))

    def display_point_cloud(self, display=False):
        self.get_camera_parameters()
        fx, cx, fy, cy = self.K[0,0], self.K[0,2], self.K[1,1], self.K[1,2]

        self.rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(self.geometry_rgb, self.geometry_depth, convert_rgb_to_intensity=False)
        height, width = self.rgb_image.shape[:2]
        point_cloud = o3d.geometry.PointCloud.create_from_rgbd_image(self.rgbd_image, o3d.camera.PinholeCameraIntrinsic(width=width, height=height, fx=fx, fy=fy, cx=cx, cy=cy))

        # Flip it, otherwise the pointcloud will be upside down
        point_cloud.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

        if(display):
            o3d.visualization.draw_geometries([point_cloud])

        return point_cloud      


if __name__ == "__main__":
    rgbd = RGBDepth()
    rgbd.ros_callback()
    