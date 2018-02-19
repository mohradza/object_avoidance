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
from object_avoidance.msg import FlowRingOutMsg
from std_msgs.msg import Float32
from std_msgs.msg import Bool

def define_rings_at_which_to_track_optic_flow(image, gamma_size, num_rings):
    points_to_track = []
    x_center = int(image.shape[1]/2)
    y_center = int(image.shape[0]/2)
    # i.r. good for 160x120 image size
    inner_radius = 50
    gamma = np.linspace(0, 2*math.pi-.017, gamma_size)
    dg = gamma[2] - gamma[1]
    dr = 4

    for ring in range(num_rings):
       for g in gamma:
          new_point = [x_center + int((inner_radius+ring*dr)*math.sin(g)), y_center - int((inner_radius+ring*dr)*math.cos(g))]
          points_to_track.append(new_point)

    points_to_track = np.array(points_to_track, dtype=np.float32) # note: float32 required for opencv optic flow calculations
    points_to_track = points_to_track.reshape(points_to_track.shape[0], 1, points_to_track.shape[1]) # for some reason this needs to be shape (npoints, 1, 2)
  
    return points_to_track


class Optic_Flow_Calculator:
    def __init__(self):
        # Define the source of the images, e.g. rostopic name
        self.image_source = "/usb_cam/image_raw"

        # Initialize image aquisition
        self.bridge = CvBridge()
        self.prev_image = None
        self.last_time = 0

        # Lucas Kanade Optic Flow parameters
        self.lk_params = dict( winSize  = (12,12),
                               maxLevel = 5,
                               criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        
        # Flow Ring Publisher
        self.Flow_rings_pub = rospy.Publisher('flow_rings', FlowRingOutMsg, queue_size=10)      

        # Raw Image Subscriber
        self.image_sub = rospy.Subscriber(self.image_source,Image,self.image_callback)

        # Define image size parameters
        self.rows = 0
        self.cols = 0
        self.num_rings = 3
        self.gamma_size = 60
        self.pixel_scale = 15

    def image_callback(self,image):
        try: # if there is an image
            # Acquire the image, and convert to single channel gray image
            curr_image = self.bridge.imgmsg_to_cv2(image, "mono8")
            # color_image = self.bridge.imgmsg_to_cv2(image,"rgb8")

            # Get time stamp
            # We want the new timestamp as close to image
            # aquisition as possible
            secs = image.header.stamp.secs
            nsecs = image.header.stamp.nsecs
            curr_time = float(secs) + float(nsecs)*1e-9

            if len(curr_image.shape) > 2:
                if curr_image.shape[2] > 1: # color image, convert it to gray
                    curr_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # shape should now be (rows, columns)
                elif curr_image.shape[2] == 1: # mono image, with wrong formatting
                    curr_image = curr_image[:,:,0] # shape should now be (rows, columns)

            # optional: resize the image
            curr_image = cv2.resize(curr_image, (0,0), fx=0.5, fy=0.5) 
            #  color_image = cv2.resize(color_image, (0,0), fx=0.5, fy=0.5)
            
            # Flip the image (mirror vertically)
            curr_image = cv2.flip(curr_image, 1)
            # color_image = cv2.flip(color_image, 1)

            # optional: apply gaussian blur
            # curr_image = cv2.GaussianBlur(curr_image,(5,5),0)

            # If this is the first loop, initialize image matrices
            if self.prev_image is None:
                self.prev_image = curr_image
                self.rows = curr_image.shape[0]
                self.cols = curr_image.shape[1]
                self.last_time = curr_time
                self.points_to_track = define_rings_at_which_to_track_optic_flow(curr_image, self.gamma_size, self.num_rings)
#                self.gamma_ring_points = define_gamma_ring_points(curr_image, self.gamma_size)
                self.flow_status_msg = True
                return # skip the rest of this loop

            # get time between images
            dt = curr_time - self.last_time
            self.last_time - curr_time

            # calculate optic flow with lucas kanade
            # see: http://docs.opencv.org/modules/video/doc/motion_analysis_and_object_tracking.html
            new_position_of_tracked_points, status, error = cv2.calcOpticalFlowPyrLK(self.prev_image, curr_image, self.points_to_track, None, **self.lk_params)

            # calculate flow field
            flow = ((new_position_of_tracked_points - self.points_to_track)/dt)*self.pixel_scale;

            # Output the flow rings
            msg = FlowRingOutMsg()
            msg.Qdot_u = flow[:,0,0]
            msg.Qdot_v = flow[:,0,1]
            self.Flow_rings_pub.publish(msg)
            # save current image and time for next loop
            self.prev_image = curr_image
#Moved up            self.last_time = curr_time


        except CvBridgeError, e:
            print e


################################################################################



def main():
  optic_flow_calculator = Optic_Flow_Calculator()
  try:
    rospy.spin()
  except KeyboardInterrupt:
    print "Shutting down"
  cv.DestroyAllWindows()

################################################################################

if __name__ == '__main__':
    rospy.init_node('optic_flow_calculator', anonymous=True)
    main()

