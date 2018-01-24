#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped, TwistStamped
from styx_msgs.msg import Lane, Waypoint
from std_msgs.msg import Int32

import math
import tf

'''
This node will publish waypoints from the car's current position to some `x` distance ahead.
As mentioned in the doc, you should ideally first implement a version which does not care
about traffic lights or obstacles.
Once you have created dbw_node, you will update this node to use the status of traffic lights too.
Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.
TODO (for Yousuf and Aaron): Stopline location for each traffic light.
'''

LOOKAHEAD_WPS = 200 # Number of waypoints we will publish. You can change this number
#MAX_SPEED = 10 * 0.44704 # 1 mph = 0.44704 m/s, for site testing
MAX_DEACCELERATION = 0.5
STOP_DISTANCE = 5.0

class WaypointUpdater(object):
    def __init__(self):
        rospy.init_node('waypoint_updater')

        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        # TODO: Add a subscriber for /traffic_waypoint and /obstacle_waypoint below
        rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb)
        #rospy.Subscriber('/obstacle_waypoint', Lane, self.obstacle_cb)
       
        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)

        #Add other member variables you need below
        self.current_pose = None
        self.waypoints = None
        self.traffic_light_wp = None  
         
        #rospy.spin()
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            self.publish()
            rate.sleep()


    def pose_cb(self, msg):
        self.current_pose = msg
        #if self.current_pose is not None:
            #self.publish()

    def waypoints_cb(self, msg):
        #if self.waypoints is None:
        self.waypoints = msg

    def traffic_cb(self, msg):
        self.traffic_light_wp = msg.data
        rospy.loginfo("Traffic light detected: " + str(msg.data))
        #if self.traffic_light_wp > -1:
            #self.publish()

    def obstacle_cb(self, msg):
        # TODO: Callback for /obstacle_waypoint message. We will implement it later
        pass
        
    def current_vel_cb(self, msg):
        current_velocity = msg.twist.linear.x

    def get_waypoint_velocity(self, waypoint):
        return waypoint.twist.twist.linear.x

    def set_waypoint_velocity(self, waypoints, waypoint, velocity):
        waypoints[waypoint].twist.twist.linear.x = velocity

    def distance(self, waypoints, wp1, wp2):
        dist = 0
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
        for i in range(wp1, wp2+1):
            dist += dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
            wp1 = i
        return dist
		
    def distance(self, p1, p2):
        x = p1.x - p2.x
        y = p1.y - p2.y
        z = p1.z - p2.z
        return math.sqrt(x*x + y*y + z*z)	
		
    def decelerate(self, waypoints, tl_wp):
    
        if tl_wp >= len(waypoints) or len(waypoints) < 1:
            return []
            
        last = waypoints[tl_wp]
        last.twist.twist.linear.x = 0.0
        for wp in waypoints[:tl_wp][::-1]:
            dist = self.distance(wp.pose.pose.position, last.pose.pose.position)
            dist = max(0.0, dist - STOP_DISTANCE)
            vel  = math.sqrt(2 * MAX_DEACCELERATION* dist)
            if vel < 1.0:
                vel = 0.0
            wp.twist.twist.linear.x = min(vel, wp.twist.twist.linear.x)
        return waypoints	
		   
    def get_closest_waypoint(self, pose, waypoints):
        closest_dist = float('inf')
        closest_wp = 0
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2)
        for idx, wp in enumerate(waypoints):
            dist = dl(pose.position, wp.pose.pose.position)
            if (dist < closest_dist):
                closest_dist = dist
                closest_wp = idx
                  
        return closest_wp
    
    def get_next_waypoint(self, pose, waypoints):
        closest_wp = self.get_closest_waypoint(pose, waypoints)
        wp_x = waypoints[closest_wp].pose.pose.position.x
        wp_y = waypoints[closest_wp].pose.pose.position.y
        heading = math.atan2((wp_y - pose.position.y), (wp_x - pose.position.x))
        quaternion = (pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w)
        _, _, yaw = tf.transformations.euler_from_quaternion(quaternion)
        angle = math.fabs(yaw - heading)

        if angle > (math.pi / 4.0):        
            closest_wp += 1
        
        return closest_wp
		
    def compute_final_waypoints(self, waypoints, first_wp, last_wp):
        final_wpts = []
        traffic_light = True
		
        if self.traffic_light_wp is None or self.traffic_light_wp < 0:
            traffic_light = False
		
		
        for i in range(first_wp, last_wp):
            index = i % len(waypoints)
            wp = Waypoint()
            wp.pose.pose.position.x  = waypoints[index].pose.pose.position.x
            wp.pose.pose.position.y  = waypoints[index].pose.pose.position.y
            wp.pose.pose.position.z  = waypoints[index].pose.pose.position.z
            wp.pose.pose.orientation = waypoints[index].pose.pose.orientation

            wp.twist.twist.linear.x = waypoints[index].twist.twist.linear.x 

            final_wpts.append(wp)

        if traffic_light:
            trafficlight_wp = len(final_wpts)

            for i in range(last_wp, first_wp + LOOKAHEAD_WPS):
                index = i % len(waypoints)
                wp = Waypoint()
                wp.pose.pose.position.x  = waypoints[index].pose.pose.position.x
                wp.pose.pose.position.y  = waypoints[index].pose.pose.position.y
                wp.pose.pose.position.z  = waypoints[index].pose.pose.position.z
                wp.pose.pose.orientation = waypoints[index].pose.pose.orientation
                wp.twist.twist.linear.x  = 0.0
                final_wpts.append(wp)
				
            final_wpts = self.decelerate(final_wpts, trafficlight_wp)

        return final_wpts
    
    def publish(self):
        if self.current_pose is not None and self.waypoints is not None:
            pose = self.current_pose
            wpts = self.waypoints.waypoints
            traffic_wpts = self.traffic_light_wp
                
            lane = Lane()
            lane.header.frame_id = '/world'
            
            
			
            next_wp = self.get_next_waypoint(pose.pose, wpts)
                        
            #further_wps = self.waypoints.waypoints[next_wp:next_wp+LOOKAHEAD_WPS]
            #lane.waypoints = further_wps

	    #traffic_light_wp_index = max(0, traffic_wpts - next_wp)
				
        # Brake if a red light is detected
            if self.traffic_light_wp is None or self.traffic_light_wp < 0:
                rospy.loginfo("traffic_light_wp < 0 ")
                lane.waypoints = self.compute_final_waypoints(wpts, next_wp, next_wp+LOOKAHEAD_WPS)
            else:
                rospy.loginfo("traffic_light_wp > 0")
                lane.waypoints = self.compute_final_waypoints(wpts, next_wp, traffic_wpts)
                
            #posx = wpts[next_wp].pose.pose.position.x
            #posy = wpts[next_wp].pose.pose.position.y
            #rospy.loginfo("Nearest waypoint: idx=%d posx=%f posy=%f", next_wp, posx, posy)
			
            lane.header.stamp = rospy.Time().now()
			
            self.final_waypoints_pub.publish(lane)

if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
