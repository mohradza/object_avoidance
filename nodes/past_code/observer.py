#!/usr/bin/env python

# Author: Michael Ohradzansky
# Date: February 2017
# Purpose: Compute azimuthal nearness as a function
#          of gamma, provided tangential optic flow 
#          and body velocities.



# ROS imports
import roslib, rospy
import message_filters

# Numpy imports
import numpy as np
import math
import std_msgs.msg

# message imports specific to the observer
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import TwistStamped
from optic_flow_code.msg import OpticFlowMsg
from std_msgs.msg import Bool
    # Possibly use this as the observer call as well
class Nearness_Observer:
    def __init__(self):
        # Inititalize state parameters
        self.vel_x = 0.0
        self.vel_y = 0.0
        self.yaw_rate = 0.0
        self.pos_x = 0.0
        self.pos_y = 0.0
        self.psi = 0.0
        self.gamma_size = 50
        self.mu = [0.0]*self.gamma_size
        self.rho = 20.0
        self.first_run = True
        self.last_time = 0.0 
        self.observer_switch = False
        self.flow_scale = 15
        self.OF_tang_sub = rospy.Subscriber('optic_flow', OpticFlowMsg, self.OF_callback)
#       self.OF_r_sub = message_filters.Subscriber('optic_flow_r', OFlowAves)
        self.vel_sub = rospy.Subscriber('mavros/local_position/velocity', TwistStamped, self.vel_callback)
        self.observer_switch_sub = rospy.Subscriber('MATLAB/record', Bool, self.switch_callback)
       # Publish the nearness:
        self.nearness_pub = rospy.Publisher("nearness",Float32MultiArray, queue_size=10) 

        
    def OF_callback(self,flow):
        if True:
            # Determine dt and update self.last_time
            now_s = flow.header.stamp.secs
            now_ns = flow.header.stamp.nsecs
            current_time = float(now_s) + float(now_ns)*1e-9
            dt = current_time - self.last_time
            self.last_time = current_time
            if self.first_run:
                self.first_run = False
            else:
                #Fill flow vectors with data
                flow_tang = flow.x[:]
        
                # Initialize observer vectors
                Qdotm = [0.0]*self.gamma_size
                Qdote = [0.0]*self.gamma_size
                mudot = [0.0]*self.gamma_size
                nearness = self.mu[:]

                # Initialize gamma
                gamma = np.linspace(0,2*math.pi,self.gamma_size)
        
                for i in range(self.gamma_size):
                    ut = self.vel_x*math.sin(gamma[i])-self.vel_y*math.cos(gamma[i]) 
                    ur = -1*(self.vel_x*math.cos(gamma[i])+self.vel_y*math.sin(gamma[i]))
                    Qdotm[i] = flow_tang[i]/self.flow_scale
                    Qdote[i] = -self.yaw_rate+nearness[i]*ut

                    # Cap the nearness values to reasonable ranges
                    if nearness[i] > 5:
                        nearness[i] = 5
                    elif nearness[i] < -5:
                        nearness[i] = -5
                    
                    # Compute the actual nearness
                    mu_sq = nearness[i]**2
                    mudot[i] = ur*(mu_sq)-self.rho*ut*(Qdote[i] -Qdotm[i])
                    self.mu[i] = nearness[i] + mudot[i]*dt
                    nearness[i] = self.mu[i] 
                
                msg = Float32MultiArray()
                msg.data = nearness[:]
                self.nearness_pub.publish(msg)
    
    # Velocity callback
    # +x is out the front
    # +y is to the left
    # yaw is positive ccw
    def vel_callback(self, vel):
        self.vel_x = vel.twist.linear.x
        self.vel_y = vel.twist.linear.y
        self.yaw_rate = vel.twist.angular.z

    def switch_callback(self, switch):
        self.observer_switch = switch.data    
    
def main():
  observer = Nearness_Observer()
  try:
    rospy.spin()
  except KeyboardInterrupt:
    print "Shutting Down"

if __name__ == '__main__':
    rospy.init_node('nearness_observer', anonymous = True)
    main()


