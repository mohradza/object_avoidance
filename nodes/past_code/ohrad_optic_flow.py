#!/usr/bin/env python

# ROS imports
import roslib, rospy

# opencv imports
import cv2, cv

# numpy imports - basic math and matrix manipulation
import numpy as np
import math
import std_msgs.msg

# imports for ROS image handling
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image

# message imports specific to this package
from optic_flow_code.msg import OpticFlowMsg
from std_msgs.msg import Float32MultiArray
################################################################################

def draw_optic_flow_field(gray_image, points, flow):
    '''
    gray_image: opencv gray image, e.g. shape = (width, height)
    points: points at which optic flow is tracked, e.g. shape = (npoints, 1, 2)
    flow: optic flow field, should be same shape as points
    '''
    color_img = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
    color_red = [0,0,255] # bgr colorspace
    linewidth = 1
    for i, point in enumerate(points):
        x = point[0,0]
        y = point[0,1]
        vx = flow[i][0,0]
        vy = flow[i][0,1]
        cv2.line(color_img, (x,y), (x+vx, y+vy), color_red, linewidth) # draw a red line from the point with vector = [vx, vy]        
    
    cv2.imshow('optic_flow_field',color_img)
    cv2.waitKey(1)


################################################################################
    
def define_rings_at_which_to_track_optic_flow(image, gamma_size, num_rings):
    points_to_track = []
    x_center = int(image.shape[0]/2)
    y_center = int(image.shape[1]/2)
    inner_radius = 160
    gamma = np.linspace(0, 2*math.pi, gamma_size)
    dr = 5
   
    for ring in range(num_rings):
        for g in gamma:
            new_point = [y_center - int((inner_radius+ring*dr)*math.sin(g)), x_center - int((inner_radius+ring*dr)*math.cos(g))]
            points_to_track.append(new_point)
    
    points_to_track = np.array(points_to_track, dtype=np.float32) # note: float32 required for opencv optic flow calculations
    points_to_track = points_to_track.reshape(points_to_track.shape[0], 1, points_to_track.shape[1]) # for some reason this needs to be shape (npoints, 1, 2)
    return points_to_track


def average_ring_flow(num_rings, gamma_size,flow):
    total_OF_tang = [0]*gamma_size
    total_OF_r = [0]*gamma_size
    gamma = np.linspace(0, 2*math.pi, gamma_size)
    for ring in range(num_rings):
       	for i in range(gamma_size):
          	index = ring*gamma_size + i
            	# According to Jishnu's MATLAB code:
            	#u = flow[index][1,0]
            	# v = flow[index][0,0]
            	total_OF_tang[i] = total_OF_tang[i]+(-1*flow[index][0,0]*math.cos(gamma[i])+flow[index][0,1]*math.sin(gamma[i]))
            	total_OF_r[i] = total_OF_r[i] - (flow[index][0,0]*math.sin(gamma[i])+flow[index][0,1]*math.cos(gamma[i])) 
       
    total_OF_tang[:] = [x / num_rings for x in total_OF_tang]
    total_OF_r[:] = [x / num_rings for x in total_OF_r]

    return total_OF_tang, total_OF_r

################################################################################


class Optic_Flow_Calculator:
    def __init__(self):
        # Define the source of the images, e.g. rostopic name
        self.image_source = "/camera/image_raw"
        
        # Initialize image aquisition
        self.bridge = CvBridge()
        self.prev_image = None
        self.last_time = 0
        
        # Lucas Kanade Optic Flow parameters
        self.lk_params = dict( winSize  = (25,25),
                               maxLevel = 2,
                               criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        
        # Lucas Kanade Publisher
        self.optic_flow_pub = rospy.Publisher("optic_flow", OpticFlowMsg, queue_size=10)
        self.optic_flow_MATLAB_pub = rospy.Publisher("MATLAB/optic_flow", Float32MultiArray, queue_size=10)
        
        # Tangential OF publisher
       # self.tang_OF_pub = rospy.Publisher("tang_OF", TangOpticFlowMsg, queue_size=10 )

        # Raw Image Subscriber
        self.image_sub = rospy.Subscriber(self.image_source,Image,self.image_callback)

        # Define image size parameters
        self.rows = 0
        self.cols = 0
        self.num_rings = 7
        self.gamma_size = 50

    def image_callback(self,image):
        try: # if there is an image
            # Acquire the image, and convert to single channel gray image
            curr_image = self.bridge.imgmsg_to_cv2(image, "mono8")
            if len(curr_image.shape) > 2:
                if curr_image.shape[2] > 1: # color image, convert it to gray
                    curr_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # shape should now be (rows, columns)
                elif curr_image.shape[2] == 1: # mono image, with wrong formatting
                    curr_image = curr_image[:,:,0] # shape should now be (rows, columns)
                
            # optional: resize the image
            curr_image = cv2.resize(curr_image, (0,0), fx=0.5, fy=0.5) 
            
            # optional: add a gaussian blur to the image
            curr_image = cv2.GaussianBlur(curr_image,(5,5),0)

            # Get time stamp
            secs = image.header.stamp.secs
            nsecs = image.header.stamp.nsecs
            curr_time = float(secs) + float(nsecs)*1e-9
            
            # If this is the first loop, initialize image matrices
            if self.prev_image is None:
                self.prev_image = curr_image
                self.rows = curr_image.shape[0]
                self.cols = curr_image.shape[1]
                self.last_time = curr_time
                self.points_to_track = define_rings_at_which_to_track_optic_flow(curr_image, self.gamma_size, self.num_rings)
                return # skip the rest of this loop
                
            # get time between images
            dt = curr_time - self.last_time
            
            # calculate optic flow with lucas kanade
            # see: http://docs.opencv.org/modules/video/doc/motion_analysis_and_object_tracking.html
            new_position_of_tracked_points, status, error = cv2.calcOpticalFlowPyrLK(self.prev_image, curr_image, self.points_to_track, None, **self.lk_params)
              
            # calculate flow field
            flow = new_position_of_tracked_points - self.points_to_track
            
            # draw the flow field
            # draw_optic_flow_field(curr_image, self.points_to_track, flow)
            
            # Compute Tangential OF
            OF_tang, OF_r = average_ring_flow(self.num_rings,self.gamma_size,flow)           
 
            # Publish Tangential OF msg
    	    msg = OpticFlowMsg()
    	    msg.x = OF_tang[:]
            msg.header.stamp = rospy.Time.now()
    	    self.optic_flow_pub.publish(msg)
            
            # Publish a message that is readable by MATLAB
    	    msg2 = Float32MultiArray()
    	    msg2.data = OF_tang[:]
    	    self.optic_flow_MATLAB_pub.publish(msg2)
            
            # save current image and time for next loop
            self.prev_image = curr_image
            self.last_time = curr_time
            
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
