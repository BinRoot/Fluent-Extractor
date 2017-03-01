#!/usr/bin/env python
import rospy, rospkg
from std_msgs.msg import String
from std_msgs.msg import Header
import os
import subprocess

import cv2
import cv_bridge

import sys
ros_path = rospkg.RosPack().get_path('fluent_extractor')
svm_rank_dir = os.path.join(ros_path, 'svm_rank_linux64')
sys.path.insert(0, svm_rank_dir)
import infer_total_utility

from sensor_msgs.msg import (
    Image,
)

# prev_val = None
# alpha = 0.8

def send_image(img):
    msg = cv_bridge.CvBridge().cv2_to_imgmsg(img, encoding="bgr8")
    pub = rospy.Publisher('/robot/xdisplay', Image, latch=True, queue_size=1)
    pub.publish(msg)
    # Sleep to allow for image to be published.
    rospy.sleep(1)


def callback(data):
    fluent_str_arr = data.data.split(' ')
    fluent_vector = []
    for fluent_str in fluent_str_arr:
        if fluent_str == '':
            continue
        print(fluent_str.split(':')[1])
        fluent_vector.append(float(fluent_str.split(':')[1]))
    print(fluent_vector)

    val = infer_total_utility.infer_total_utility(fluent_vector)
    # if prev_val is not None:
    #     val = alpha * val + (1 - alpha) * pre_val
    # prev_val = val

    val_print = 'Fluent value is {}'.format(val)
    print(val_print)
    robot_face_imgfile = os.path.join(ros_path, 'robot.png')
    img = cv2.imread(robot_face_imgfile)
    cloth_mask_imgfile = os.path.join(ros_path, 'cloth.png')
    cloth_img = cv2.imread(cloth_mask_imgfile)
    font = cv2.FONT_HERSHEY_SIMPLEX
    redness = min(max(0, 255 - 3*(val + 30)), 255)
    #cv2.rectangle(img, (0, 350), (1024, 600), (255, max(150, 255-redness), redness), -1)
    cv2.rectangle(img, (0, 600 - 256), (1024, 600), (0, 0, 0), -1)
    cloth_img = cv2.resize(cloth_img, None, fx=0.5, fy=0.5)
    x_offset = 1024 - cloth_img.shape[1]
    y_offset = 600 - cloth_img.shape[0]
    img[y_offset:y_offset+cloth_img.shape[0], x_offset:x_offset+cloth_img.shape[1]] = cloth_img
    cv2.putText(img, val_print, (30, 600 - 256/2), font, 1, (16, 32, 255), 3)
    cv2.imshow('face', img)
    cv2.waitKey(40)
    send_image(img)

if __name__ == '__main__':
    rospy.init_node('value_inferer')
    rospy.Subscriber("/vcla/cloth_folding/fluent_vector", String, callback, queue_size=1)
    rospy.spin()


