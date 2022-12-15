import rospy
import numpy as np
from sensor_msgs.msg import Image
import cv2
from deblurring.predict import Predictor
from detectron.detectron import prediction
from BeV.predict import predict
from cv_bridge import CvBridge

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Global Variables
bridge = CvBridge()
publisher = rospy.Publisher("/cv_pipeline_result/", data_class=Image, queue_size=10)
result = None

def deblur(image):
    weights_path = r"/home/parallels/catkin_ws/src/cv/scripts/deblurring/best_fpn.h5"
    predictor = Predictor(weights_path=weights_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    deblurred = predictor(image, mask=None)
    deblurred = cv2.cvtColor(deblurred, cv2.COLOR_RGB2BGR)

    return deblurred

def segmentation(image):
    cfg_path = r"/home/parallels/catkin_ws/src/cv/scripts/detectron/cfg_model.pickle"
    meta_data = r"/home/parallels/catkin_ws/src/cv/scripts/detectron/output/category_test_coco_format.json"
    model_weights_path = r"/home/parallels/catkin_ws/src/cv/scripts/detectron/output/model_final.pth"

    segmented = prediction(cfg_path, model_weights_path, image, meta_data, thresh=0.3)

    return segmented

def birds_eye_view(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    config_file_path = r"/home/parallels/catkin_ws/src/cv/scripts/BeV/config.2_F.unetxst.yml"
    model_weight_path = r"/home/parallels/catkin_ws/src/cv/scripts/BeV/best_weights.hdf5"

    BeV_transformed = predict(image, config_file_path, model_weight_path)
    return BeV_transformed

def cv_pipeline(image, blur=False):
    rospy.loginfo("Image is loaded")
    cv2.imwrite("original.png", image)

    if(blur):
        # To purposefully blur an image for testing purposes
        image = cv2.GaussianBlur(image, ksize=(3,3), sigmaX=1.3)
        rospy.loginfo("Image is blurred")
        cv2.imwrite("Blurred.png", image)

    deblurred = deblur(image)
    rospy.loginfo("Image is deblurred")
    cv2.imwrite("deblurred.png", deblurred)
    
    segmented = segmentation(deblurred)
    rospy.loginfo("Image is segmented")
    cv2.imwrite("segmented.png", segmented)

    BeV_transformed = birds_eye_view(segmented)
    rospy.loginfo("Image transformed to BeV")
    cv2.imwrite("BeV.png", BeV_transformed)

    return BeV_transformed


def callback(message):
    image = bridge.imgmsg_to_cv2(message, desired_encoding='bgr8') # shape: 480 x 640 x 3
    image = cv2.imread("/home/parallels/catkin_ws/src/cv/scripts/render0909.png")
    result = cv_pipeline(image)
    result = bridge.cv2_to_imgmsg(result, encoding="passthrough")

    publisher.publish(result) 
    rospy.loginfo("Image Published")
    

if __name__ == "__main__":
    rospy.init_node("ros_wrapper", anonymous=True)
    rospy.loginfo("Node has been initialized")

    rospy.Subscriber("/camera/color/image_raw", data_class=Image, callback=callback)
    rospy.loginfo("Node started") # simply keeps python from exiting until this node is stopped

    rospy.spin()
