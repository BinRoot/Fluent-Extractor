#!/usr/bin/env python
import rospy

from std_msgs.msg import String
from baxter_core_msgs.msg import EndpointState


def show_endpoint(endpoint):
    pos_str = '[{}, {}, {}]'.format(endpoint.pose.position.x, endpoint.pose.position.y, endpoint.pose.position.z)
    quat_str = 'Quaternion(x={}, y={}, z={}, w={})'.format(endpoint.pose.orientation.x, endpoint.pose.orientation.y,
                                                           endpoint.pose.orientation.z, endpoint.pose.orientation.w)
    return '{}\n{}'.format(pos_str, quat_str)


def callback_left(data):
    str = '< Left: {}\n'.format(show_endpoint(data))
    print(str)


def callback_right(data):
    str = '> Right:\n{}\n'.format(show_endpoint(data))
    print(str)

if __name__ == '__main__':
    rospy.init_node('cloth_folder_arm_status')
    rospy.Subscriber("/robot/limb/left/endpoint_state", EndpointState, callback_left)
    rospy.Subscriber("/robot/limb/right/endpoint_state", EndpointState, callback_right)
    rospy.spin()

