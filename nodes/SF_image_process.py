#!/usr/bin/env python
from __future__ import division

# ROS imports
import roslib, rospy
# opencv imports
import cv2

# numpy imports - basic math and matrix manipulation
import numpy as np
import math
import std_msgs.msg
import operator

# imports for ROS image handling
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image

# message imports specific to this package
from object_avoidance.msg import OpticFlowMsg
from object_avoidance.msg import FourierCoefsMsg
from object_avoidance.msg import YawRateCmdMsg
from object_avoidance.msg import FOF_and_ResidualMsg
from std_msgs.msg import Float32




def draw_optic_flow_signal(color_image, flow_signal, gamma_size):
    gamma_ring_points = []
    signal_loc_x = []
    signal_loc_y = []
    gamma_x = []
    gamma_y = []

    x_center = int(color_image.shape[0]/2)
    y_center = int(color_image.shape[1]/2)
    color_red = [255,230,0]
    color_green = [255,0,0]
    linewidth = 2
    
    radius = 110
    sig_scale = 10
    gamma = np.linspace(-math.pi, math.pi - .017, gamma_size)

    for i in range(gamma_size):
        gamma_x.append(x_center - int(radius*math.cos(gamma[i])))
        gamma_y.append(y_center + int(radius*math.sin(gamma[i])));
        signal_loc_x.append(gamma_x[i] - int(sig_scale*flow_signal[i]*math.cos(gamma[i])))
        signal_loc_y.append(gamma_y[i] + int(sig_scale*flow_signal[i]*math.sin(gamma[i])))

    for i in range(gamma_size):
        if i == 0:
            cv2.line(color_image, (gamma_y[i], gamma_x[i]), (gamma_y[29], gamma_x[29]), color_green, linewidth)
            cv2.line(color_image, (signal_loc_y[i], signal_loc_x[i]), (signal_loc_y[29], signal_loc_x[29]), color_red, linewidth)
           # cv2.line(color_image, (gamma_y[i], gamma_x[i]), (gamma_y[29], gamma_x[29]), color_green, 1)
        else:
            cv2.line(color_image, (gamma_y[i], gamma_x[i]), (gamma_y[i-1], gamma_x[i-1]), color_green, linewidth)
            cv2.line(color_image, (signal_loc_y[i], signal_loc_x[i]), (signal_loc_y[i-1], signal_loc_x[i-1]), color_red, linewidth)
           # cv2.line(color_image, (gamma_y[i], gamma_x[i]), (gamma_y[i-1], gamma_x[i-1]), color_green, 1)

    cv2.imshow('optic_flow_signal', color_image)
    cv2.waitKey(1)
    return color_image

class Optic_Flow_Visualizer:
    def __init__(self):
        # Define the source of the images, e.g. rostopic name
        self.image_source = "/usb_cam/image_raw"
        self.flow_source = "FOF_data"
        # Initialize image aquisition
        self.bridge = CvBridge()
        self.prev_image = None
        # Raw Image Subscriber
        self.image_sub = rospy.Subscriber(self.image_source,Image,self.image_callback)

        # Flow Subscriber
        self.flow_sub = rospy.Subscriber(self.flow_source, FOF_and_ResidualMsg, self.flow_cb)
         
        # Publish processed image
        self.image_out_pub = rospy.Publisher("image_out", Image, queue_size = 10)

        # Define image size parameters
        self.rows = 0
        self.cols = 0
        self.gamma_size = 30
        self.SF_signal = [0.0]*self.gamma_size
    
    def flow_cb(self,data):
        self.SF_signal = data.FOF_OF_SF


    def image_callback(self,image):
        try: # if there is an image
            # Acquire the image, and convert to single channel gray image
            color_image = self.bridge.imgmsg_to_cv2(image,"rgb8")

            # optional: resize the image
            # curr_image = cv2.resize(curr_image, (0,0), fx=0.5, fy=0.5) 

            # Flip the image (mirror vertically)
            color_image = cv2.flip(color_image, 1)

            # If this is the first loop, initialize image matrices
            if self.prev_image is None:
                self.prev_image = color_image
                self.rows = color_image.shape[0]
                self.cols = color_image.shape[1]
                return # skip the rest of this loop

            # Draw the superimposed flow signal
            image_out = draw_optic_flow_signal(color_image, self.SF_signal, self.gamma_size)

            # Publish the images
           # image_out_msg = self.bridge.cv2_to_imgmsg(image_out, "rgb8")
            self.image_out_pub.publish(self.bridge.cv2_to_imgmsg(image_out,"rgb8"))

        except CvBridgeError, e:
            print e



################################################################################



def main():
  optic_flow_vis = Optic_Flow_Visualizer()
  try:
    rospy.spin()
  except KeyboardInterrupt:
    print "Shutting down"
  cv.DestroyAllWindows()

################################################################################

if __name__ == '__main__':
    rospy.init_node('optic_flow_visualizer', anonymous=True)
    main()

