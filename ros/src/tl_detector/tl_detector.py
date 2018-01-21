#!/usr/bin/env python
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, Pose, Point
from styx_msgs.msg import TrafficLightArray, TrafficLight
from styx_msgs.msg import Lane
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from light_classification.tl_classifier import TLClassifier
import math
import rospy
import tf
import cv2
import yaml
import numpy as np
import os

STATE_COUNT_THRESHOLD = 3


class TLDetector(object):
    def __init__(self):
        rospy.init_node('tl_detector')

        self.pose = None
        self.waypoints = None
        self.camera_image = None
        self.lights = []
        self.stop_line_indices = []
        self.car_current_waypoint = None
        self.unknown_count = 0

        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string)
        tl_classifier_class = rospy.get_param('~inference_class')
        self.bridge = CvBridge()
        self.light_classifier = TLClassifier()

        self.state = TrafficLight.UNKNOWN
        self.last_state = TrafficLight.UNKNOWN
        self.last_wp = -1
        self.state_count = 0

        # ROS publishers
        self.upcoming_red_light_pub = rospy.Publisher(
            '/traffic_waypoint', Int32, queue_size=1)

        # ROS subscribers
        sub1 = rospy.Subscriber(
            '/current_pose', PoseStamped, self.pose_cb, queue_size=1)
        sub2 = rospy.Subscriber(
            '/base_waypoints', Lane, self.waypoints_cb, queue_size=1)

        # /vehicle/traffic_lights provides you with the location of the traffic
        # light in 3D map space and helps you acquire an accurate ground truth
        # data source for the traffic light classifier by sending the current
        # color state of all traffic lights in the simulator.
        # When testing on the vehicle, the color state will not be available.
        # You'll need to rely on the position of the light and the camera image
        # to predict it.
        sub3 = rospy.Subscriber(
            '/vehicle/traffic_lights',
            TrafficLightArray,
            self.traffic_cb,
            queue_size=1)
        sub4 = rospy.Subscriber(
            '/image_color', Image, self.image_cb, queue_size=1)

        rospy.spin()

    def pose_cb(self, msg):
        self.pose = msg

    def waypoints_cb(self, waypoints):
        self.waypoints = waypoints
        stop_line_positions = self.config['stop_line_positions']

        for stop_line_position in stop_line_positions:
            line_pose = Pose()
            line_pose.position.x = stop_line_position[0]
            line_pose.position.y = stop_line_position[1]
            # Find the nearest waypoint
            # closest_wp = 1
            closest_wp = self.get_closest_waypoint(line_pose)
            print "wp = ", closest_wp
            # Add the stop line waypoint index to the list
            self.stop_line_indices.append(closest_wp)

    def traffic_cb(self, msg):
        self.lights = msg.lights

    def current_waypoint_cb(self, waypoint_idx):
        self.car_current_waypoint = waypoint_idx.data

    def image_cb(self, msg):
        """Identifies red lights in the incoming camera image and publishes
            the index of the waypoint closest to the red light's stop line to
            /traffic_waypoint
        Args:
            msg (Image): image from car-mounted camera
        """
        self.has_image = True
        self.camera_image = msg
        light_wp, state = self.process_traffic_lights()
        print light_wp, state
        if state == TrafficLight.UNKNOWN:
            self.unknown_count = self.unknown_count + 1
            if (self.unknown_count <= STATE_COUNT_THRESHOLD):
                return
        else:
            self.unknown_count = 0

        # treat yellow as red for stopping purposes
        if state == TrafficLight.YELLOW:
            state = TrafficLight.RED

        if self.state != state:
            self.state_count = 0
            self.state = state
        elif self.state_count >= STATE_COUNT_THRESHOLD:
            self.last_state = self.state
            if state not in [TrafficLight.RED, TrafficLight.YELLOW]:
                light_wp = -1
            self.last_wp = light_wp
            self.upcoming_red_light_pub.publish(Int32(light_wp))
        else:
            self.upcoming_red_light_pub.publish(Int32(self.last_wp))
        self.state_count += 1

    def get_light_state(self, light):
        """Determines the current color of the traffic light
        Args:
            light (TrafficLight): light to classify
        Returns:
            int: ID of traffic light color
                 (specified in styx_msgs/TrafficLight)
        """
        if (not self.has_image):
            self.prev_light_loc = None
            return TrafficLight.UNKNOWN
        camera_img = self.bridge.imgmsg_to_cv2(self.camera_image, "rgb8")
        return self.light_classifier.get_classification(camera_img)

    def get_closest_waypoint(self, pose):
        """Identifies the closest path waypoint to the given position
            https://en.wikipedia.org/wiki/Closest_pair_of_points_problem
        Args:
            pose (Pose): position to match a waypoint to
        Returns:
            int: index of the closest waypoint in self.waypoints
        """
        # Initialize comparison values
        closest_wp_dist = 9999
        closest_wp_ind = -1

        # Comparison Loop
        for i in range(0, len(self.waypoints.waypoints)):
            wp_dist = self.distance(
                self.waypoints.waypoints[i].pose.pose.position, pose.position)
            if wp_dist < closest_wp_dist:
                closest_wp_dist = wp_dist
                closest_wp_ind = i
        return closest_wp_ind

    def index_diff(self, idx1, idx2):
        N_waypts = len(self.waypoints.waypoints)
        if idx2 >= idx1:
            diff = idx2 - idx1
        else:
            diff = N_waypts - idx1 - 1 + idx2
        return diff

    def process_traffic_lights(self):
        """Finds closest visible traffic light, if one exists,
           and determines its location and color
        Returns:
            int: index of waypoint closest to the upcoming stop line for a
                 traffic light (-1 if none exists)
            int: ID of traffic light color
                 (specified in styx_msgs/TrafficLight)
        """
        max_dist_tl = 200 / 0.63 # length of track by distance between waypoints
        light_wp_idx = -1
        state = TrafficLight.UNKNOWN
        light_idx = None

        min_dist = 9999

        car_current_waypoint = self.get_closest_waypoint(
            self.pose.pose)
        # print "ccp",car_current_waypoint

        if self.pose and (car_current_waypoint is not None):
            for i, stop_line_index in enumerate(self.stop_line_indices):
                idx_dist = self.index_diff(stop_line_index,
                                           car_current_waypoint)
                # print "idxdist",idx_dist
                if idx_dist < min_dist:
                    light_idx = i
                    min_dist = idx_dist
                    light_wp_idx = stop_line_index

        # print "minimum_dist", min_dist

        # If waypoint has been found get traffic light state
        if min_dist < max_dist_tl:
            state = self.get_light_state(light_idx)

            return light_wp_idx, state

        return -1, TrafficLight.UNKNOWN

    def distance(self, obj1, obj2):
        x, y, z = obj1.x - obj2.x, obj1.y - obj2.y, obj1.z - obj2.z
        return math.sqrt(x * x + y * y + z * z)


if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
