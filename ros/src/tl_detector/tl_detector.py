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

        tl_classifier_class = rospy.get_param('~inference_class')

        self.pose = None
        self.waypoints = None
        self.camera_image = None
        self.lights = []
        self.tl_waypoints_ind = []

        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string)

        self.bridge = CvBridge()
        self.light_classifier = globals()[tl_classifier_class]()
        self.listener = tf.TransformListener()

        self.state = TrafficLight.UNKNOWN
        self.last_wp = -1
        self.state_count = 0

        # ROS publishers
        self.upcoming_red_light_pub = rospy.Publisher('/traffic_waypoint', Int32, queue_size=1)
        self.log_pub = rospy.Publisher('/vehicle/visible_light_idx', Int32, queue_size=1)

        # /vehicle/traffic_lights provides you with the location of the traffic
        # light in 3D map space and helps you acquire an accurate ground truth
        # data source for the traffic light classifier by sending the current
        # color state of all traffic lights in the simulator.
        # When testing on the vehicle, the color state will not be available.
        # You'll need to rely on the position of the light and the camera image
        # to predict it.

        # ROS subscribers
        sub1 = rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb, queue_size=1)
        sub2 = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb, queue_size=1)
        sub3 = rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.traffic_cb, queue_size=1)
        sub4 = rospy.Subscriber('/image_color', Image, self.image_cb, queue_size=1)

        rospy.spin()

    def pose_cb(self, msg):
        self.pose = msg

    def waypoints_cb(self, waypoints):
        self.waypoints = waypoints

    def traffic_cb(self, msg):
        self.lights = msg.lights

    def image_cb(self, msg):
        """Identifies red lights in the incoming camera image and publishes
            the index of the waypoint closest to the red light's stop line to
            /traffic_waypoint
        Args:
            msg (Image): image from car-mounted camera
        """
        self.has_image = True
        self.camera_image = msg
        light_wp_ind, state = self.process_traffic_lights()

        # Publish upcoming red lights at camera frequency.
        # Each predicted state has to occur `STATE_COUNT_THRESHOLD` number
        # of times till we start using it.
        # Otherwise the previous stable state is used.
        if self.state != state:
            self.state_count = 0
            self.state = state
        elif self.state_count >= STATE_COUNT_THRESHOLD:
            light_wp_ind = light_wp_ind if state == TrafficLight.RED else -1
            self.last_wp = light_wp_ind
            self.upcoming_red_light_pub.publish(Int32(light_wp_ind))
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
        if(not self.has_image):
            return TrafficLight.UNKNOWN
        cam_image = self.bridge.imgmsg_to_cv2(self.camera_image, "rgb8")
        return self.light_classifier.get_classification(cam_image)

    def get_closest_wp(self, pose):
        closest_waypoint_dist = 9999
        closest_waypoint = -1

        # Looping through base waypoints to find the one closest to the car.
        for i in range(0, len(self.waypoints.waypoints)):
            waypoint_distance = self.distance_calc(self.waypoints.waypoints[i].pose.pose.position, pose.position)
            if waypoint_distance < closest_waypoint_dist:
                closest_waypoint_dist = waypoint_distance
                closest_waypoint = i
        return closest_waypoint

    def get_tl_wp(self):
        stop_line_positions = self.config['stop_line_positions']

        for stop_line_position in stop_line_positions:
            stop_line_position_pose = Pose()
            stop_line_position_pose.position.x = stop_line_position[0]
            stop_line_position_pose.position.y = stop_line_position[1]
            stop_line_position_pose.position.z = 0
            self.tl_waypoints_ind.append(self.get_closest_wp(stop_line_position_pose))

    def get_veh_location(self, car, light, track_length):
        loc = light - car
        if (loc < -0.5 * track_length):
            loc += track_length
        elif (loc > 0.5 * track_length):
            loc -= track_length
        return loc

    def get_closest_tl(self, car_wp, fov):
        light_wp_ind = -1
        light_number = -1

        if(self.tl_waypoints_ind == []):
            self.get_tl_wp()
        track_dist = len(self.waypoints.waypoints)
        smallest_tl_distance = 9999

        for i, tl_ind in enumerate(self.tl_waypoints_ind):
            distance_between_wp = self.get_veh_location(car_wp, tl_ind, track_dist)
            if(distance_between_wp < fov and distance_between_wp < smallest_tl_distance and distance_between_wp > 0):
                light_wp_ind = tl_ind
                light_number = i
                smallest_tl_distance = distance_between_wp

        return light_number, light_wp_ind

    def process_traffic_lights(self):
        """Finds closest visible traffic light, if one exists,
           and determines its location and color
        Returns:
            int: index of waypoint closest to the upcoming stop line for a
                 traffic light (-1 if none exists)
            int: ID of traffic light color
                 (specified in styx_msgs/TrafficLight)
        """
        fov = 180
        light_ind = -1
        light_num = -1
        light = None

        if(self.pose and self.waypoints):
            car_wp = self.get_closest_wp(self.pose.pose)
            light_num, light_ind = self.get_closest_tl(car_wp, fov)

        # If waypoint has been found get traffic light state
        if light_ind >= 0 and light_num >= 0:
            light = self.lights[light_num]
            state = self.get_light_state(light)

            rospy.loginfo("Traffic light detected - State: {}".format(state))
            return light_ind, state

        return -1, TrafficLight.UNKNOWN
    
    def distance_calc(self, WL1, WL2):
        x, y, z = WL1.x - WL2.x, WL1.y - WL2.y, WL1.z - WL2.z
        return math.sqrt(x*x + y*y + z*z)

if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Traffic Node Interrupt')
