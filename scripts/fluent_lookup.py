#!/usr/bin/env python
import rospy, rospkg
from std_msgs.msg import String
from std_msgs.msg import Header
import os
import subprocess
import csv
import yaml

import cv2
import cv_bridge

import numpy as np

from sensor_msgs.msg import (
    Image,
)


def send_image(img):
    msg = cv_bridge.CvBridge().cv2_to_imgmsg(img, encoding="bgr8")
    pub.publish(msg)
    # Sleep to allow for image to be published.
#    rospy.sleep(1)


def callback(data):
    ros_path = rospkg.RosPack().get_path('fluent_extractor')
    
    dim_vals = data.data.split(" ")
    fluent_vec = []
    for dim_val in dim_vals:
        if dim_val == '':
            continue
        fluent_vec.append(float(dim_val.split(":")[1]))

    fluent_vec = np.asarray(fluent_vec)

    print(fluent_vec)
    # look up the fluent_vec
    shortest_dist = float('inf')
    best_hoi = None
    for hoi in hoi_data:
        dist = np.linalg.norm(hoi['fluent_a'] - fluent_vec)
        if dist < shortest_dist:
            shortest_dist = dist
            best_hoi = hoi

    cloth_mask_imgfile = os.path.join(ros_path, 'cloth.png')

    cloth_img = cv2.imread(cloth_mask_imgfile)
    img_fa = best_hoi['fluent_a_img']
    img_fb = best_hoi['fluent_b_img']

    grip_x, grip_y = tuple(best_hoi['grip'])
    release_x, release_y = tuple(best_hoi['release'])

    pub_hoi.publish("{},{},{},{}".format(grip_x, grip_y, release_x, release_y))

    grip_x *= img_fa.shape[0]
    grip_y *= img_fa.shape[0]
    release_x *= img_fa.shape[0]
    release_y *= img_fa.shape[0]

    grip = (int(grip_x), int(grip_y))
    release = (int(release_x), int(release_y))

    cv2.circle(img_fa, grip, 10, (0, 0, 255), -1)
    cv2.circle(img_fa, release, 10, (255, 0, 0), -1)

    robot_face_imgfile = os.path.join(ros_path, 'robot.png')
    img = cv2.imread(robot_face_imgfile)

    x_offset = 0
    y_offset = 600 - img_fa.shape[0]
    img[y_offset:y_offset+img_fa.shape[0], x_offset:x_offset+img_fa.shape[1]] = img_fa

    x_offset = 1024 - img_fb.shape[1]
    y_offset = 600 - img_fb.shape[0]
    img[y_offset:y_offset+img_fb.shape[0], x_offset:x_offset+img_fb.shape[1]] = img_fb

    arrow_imgfile = os.path.join(ros_path, 'arrow.png')
    img_arrow = cv2.imread(arrow_imgfile)

    x_offset = 1024/2 - img_arrow.shape[1]/2
    y_offset = 600 - img_arrow.shape[0] - 100
    img[y_offset:y_offset+img_arrow.shape[0], x_offset:x_offset+img_arrow.shape[1]] = img_arrow

    send_image(img)
    

if __name__ == '__main__':
    rospy.init_node('fluent_lookup')

    pub = rospy.Publisher('/robot/xdisplay', Image, latch=True, queue_size=1)
    pub_hoi = rospy.Publisher('/vcla/cloth_folding/hoi_action', String, queue_size=1)

    ros_path = rospkg.RosPack().get_path('fluent_extractor')
    hoi_path = os.path.join(ros_path, 'fluents', 'hoi.csv')
    
    print('hoi path', hoi_path)
    hoi_data = []
    with open(hoi_path, 'rb') as csv_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            fluent_a_filename = row[0]
            fluent_b_filename = row[1]
            grip_point = [float(row[2]), float(row[3])]
            release_point = [float(row[4]), float(row[5])]
            fluent_a_filepath = os.path.join(ros_path, 'fluents', fluent_a_filename)
            fluent_b_filepath = os.path.join(ros_path, 'fluents', fluent_b_filename)
            fluent_a_data = np.asarray(cv2.cv.Load(fluent_a_filepath)).flatten()
            fluent_b_data = np.asarray(cv2.cv.Load(fluent_b_filepath)).flatten()
            state_a_filename = "{}_state.png".format(fluent_a_filename.split('_')[0])
            state_b_filename = "{}_state.png".format(fluent_b_filename.split('_')[0])
            fluent_a_img_filepath = os.path.join(ros_path, 'fluents', state_a_filename)
            fluent_a_img = cv2.imread(fluent_a_img_filepath)
            fluent_b_img_filepath = os.path.join(ros_path, 'fluents', state_b_filename)
            fluent_b_img = cv2.imread(fluent_b_img_filepath)
            hoi_data.append({
                'fluent_a': fluent_a_data,
                'fluent_b': fluent_b_data,
                'grip': grip_point,
                'release': release_point,
                'fluent_b_img': fluent_b_img,
                'fluent_b_filename': state_b_filename,
                'fluent_a_img': fluent_a_img,
                'fluent_a_filename': state_a_filename
            })

    rospy.Subscriber("/vcla/cloth_folding/fluent_vector", String, callback)
    rospy.spin()


