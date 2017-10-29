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


def define_rings_at_which_to_track_optic_flow(image, gamma_size, num_rings):
    points_to_track = []
    x_center = int(image.shape[0]/2)
    y_center = int(image.shape[1]/2)
    # i.r. good for 160x120 image size
    inner_radius = 50
    gamma = np.linspace(0, 2*math.pi-.017, gamma_size)
    dg = gamma[2] - gamma[1]
    dr = 2

    for ring in range(num_rings):
       for g in gamma:
          new_point = [y_center - int((inner_radius+ring*dr)*math.sin(g)), x_center - int((inner_radius+ring*dr)*math.cos(g))]
          points_to_track.append(new_point)

    points_to_track = np.array(points_to_track, dtype=np.float32) # note: float32 required for opencv optic flow calculations
    points_to_track = points_to_track.reshape(points_to_track.shape[0], 1, points_to_track.shape[1]) # for some reason this needs to be shape (npoints, 1, 2)
  
    return points_to_track

def define_gamma_ring_points(image, gamma_size):
    gamma_ring_points = []
    ring_radius = 55
    gamma = np.linspace(0, 2*math.pi-.017, gamma_size)
    x_center = int(image.shape[0]/2)
    y_center = int(image.shape[1]/2)
    for g in gamma:
        new_point = [y_center - int(ring_radius*math.sin(g)), x_center - int(ring_radius*math.cos(g))]
        gamma_ring_points.append(new_point)
    
    gamma_ring_points = np.array(gamma_ring_points, dtype=np.float32)

    return gamma_ring_points
    
def average_ring_flow(self, num_rings, gamma_size,flow):
    total_OF_tang = [0]*gamma_size
    OF_reformat = [0]*gamma_size
    gamma = np.linspace(0, 2*math.pi-.017, gamma_size)
    for ring in range(num_rings):
        for i in range(gamma_size):
                index = ring*gamma_size + i
                # According to Jishnu's MATLAB code:
                #u = flow[index][1,0]
                # v = flow[index][0,0]
                total_OF_tang[i] = total_OF_tang[i]+(-1*flow[index][0,0]*math.cos(gamma[i])+flow[index][0,1]*math.sin(gamma[i]))

    total_OF_tang[:] = [x / num_rings for x in total_OF_tang]

    # Reformat so that optic flow is -pi -> pi
    for i in range(gamma_size):
        if (i < (gamma_size//2)):
            OF_reformat[i] = -total_OF_tang[gamma_size//2 - i]
        if (i >=(gamma_size//2)):
            OF_reformat[i] = -total_OF_tang[(gamma_size + gamma_size//2 - 1)-i]

    return OF_reformat

def FOF_control_calc(gamma_size, OF_tang_prev_filtered, OF_tang_curr):
    
    # Flow of Flow Smoothing Parameters
    dt = .05
    tau = .75
    alpha = dt/(tau+dt)

    # Controller Parameters
    k_0 = .3
    c_psi = .1
    c_d = .25

    # Initialize the gamma vector [-pi -> ~pi, 30]
    gamma = np.linspace(-math.pi, math.pi-.017, gamma_size)
    dg = gamma[2] - gamma[1] # Delta Gamma
    LP_OF = [0.0]*gamma_size
    R_FOF = [0.0]*gamma_size

    # Run the Low Pass filter 
    # In this case, self.OF_tang_prev is the unflitered OF signal
    # from the previous time step.
    for i in range(gamma_size):
        LP_OF[i] = alpha*OF_tang_curr[i] + (1-alpha)*OF_tang_prev_filtered[i]
        
    # Final flow of flow output
    for i in range(gamma_size-1):
        R_FOF[i] = LP_OF[i]*OF_tang_curr[i+1] - OF_tang_curr[i]*LP_OF[i+1]
    
    R_FOF[gamma_size-1] = 0.0
    
    # Dynamic Threshold
    mean = np.mean(R_FOF)
    std_dev = 0.0
    for i in range(gamma_size):
        std_dev += np.square(R_FOF[i] - mean)

    std_dev = np.sqrt(std_dev/(gamma_size))
    min_threshold_FOF = 3*std_dev 
 
    # Extract r_0 and d_0 from processed  signal
    index_max_FOF = np.argmax(R_FOF)
    d_0_FOF = R_FOF[index_max_FOF]
    min_threshold_FOF = .3
    if d_0_FOF > min_threshold_FOF:
        # Find the gamma location
        r_0_FOF = gamma[index_max_FOF]

        # Calculate the control signal
        yaw_rate_cmd_FOF = k_0*np.sign(r_0_FOF)*math.exp(-c_psi*math.fabs(r_0_FOF))*math.exp(-c_d*1/(math.fabs(d_0_FOF)))
    else:
        yaw_rate_cmd_FOF = 0.0

    return yaw_rate_cmd_FOF, R_FOF, LP_OF, min_threshold_FOF

class Optic_Flow_Calculator:
    def __init__(self):
        # Define the source of the images, e.g. rostopic name
        self.image_source = "/usb_cam/image_raw"

        # Initialize image aquisition
        self.bridge = CvBridge()
        self.prev_image = None
        self.last_time = 0

        # Lucas Kanade Optic Flow parameters
        self.lk_params = dict( winSize  = (25,25),
                               maxLevel = 5,
                               criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        
        # FOF and Residual SF publishing
        self.FOF_pub = rospy.Publisher("FOF_data", FOF_and_ResidualMsg, queue_size=10)

        # Raw Image Subscriber
        self.image_sub = rospy.Subscriber(self.image_source,Image,self.image_callback)

        # Publish yaw rate command
        self.yaw_rate_cmd_pub = rospy.Publisher("controller_out/yaw_rate_cmd", YawRateCmdMsg, queue_size=10)

        # Define image size parameters
        self.rows = 0
        self.cols = 0
        self.num_rings = 5
        self.gamma_size = 60
        yaw_rate_cmd = 0
        self.OF_tang_prev = [0.0]*self.gamma_size
        self.OF_tang_prev_filtered = [0.0]*self.gamma_size
        self.OF_tang_curr = [0.0]*self.gamma_size        

    def image_callback(self,image):
        try: # if there is an image
            # Acquire the image, and convert to single channel gray image
            curr_image = self.bridge.imgmsg_to_cv2(image, "mono8")
#            color_image = self.bridge.imgmsg_to_cv2(image,"rgb8")

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
#            color_image = cv2.resize(color_image, (0,0), fx=0.5, fy=0.5)
            # Flip the image (mirror vertically)
            curr_image = cv2.flip(curr_image, 1)
#            color_image = cv2.flip(color_image, 1)

            # optional: apply gaussian blur
#            curr_image = cv2.GaussianBlur(curr_image,(5,5),0)

            # If this is the first loop, initialize image matrices
            if self.prev_image is None:
                self.prev_image = curr_image
                self.rows = curr_image.shape[0]
                self.cols = curr_image.shape[1]
                self.last_time = curr_time
                self.points_to_track = define_rings_at_which_to_track_optic_flow(curr_image, self.gamma_size, self.num_rings)
                self.gamma_ring_points = define_gamma_ring_points(curr_image, self.gamma_size)
                return # skip the rest of this loop

            # get time between images
            dt = curr_time - self.last_time
            self.last_time - curr_time

            # calculate optic flow with lucas kanade
            # see: http://docs.opencv.org/modules/video/doc/motion_analysis_and_object_tracking.html
            new_position_of_tracked_points, status, error = cv2.calcOpticalFlowPyrLK(self.prev_image, curr_image, self.points_to_track, None, **self.lk_params)

            # calculate flow field
            flow = (new_position_of_tracked_points - self.points_to_track)/dt

            # Compute Tangential OF
            self.OF_tang_prev = self.OF_tang_curr
            self.OF_tang_curr = average_ring_flow(self, self.num_rings,self.gamma_size,flow)
                      
            # Compute Flow of Flow datasets and yaw control command
            FOF_yaw_rate_cmd, R_FOF, self.OF_tang_prev_filtered, FOF_thresh = FOF_control_calc(self.gamma_size, self.OF_tang_prev_filtered, self.OF_tang_curr)            
  
            ##### ADD ALL NECESSARY MESSAGES ######
            msg = FOF_and_ResidualMsg()
            msg.Qdot_meas = self.OF_tang_curr
            msg.FOF_OF_SF = R_FOF
            msg.FOF_threshold = FOF_thresh
            msg.FOF_yaw_rate_cmd = FOF_yaw_rate_cmd
#            msg.FR_Qdot_SF = Qdot_SF 
#            msg.FR_yaw_rate_cmd = FR_yaw_rate_cmd
#            msg.FR_threshold = FR_thresh  

            self.FOF_pub.publish(msg)

            # Publish yaw rate command
            msg = YawRateCmdMsg()
            msg.header.stamp = rospy.Time.now()
            msg.yaw_rate_cmd = FOF_yaw_rate_cmd
            self.yaw_rate_cmd_pub.publish(msg)


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

