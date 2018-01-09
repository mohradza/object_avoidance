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

def define_rings_at_which_to_track_optic_flow(image, gamma_size, gamma, num_rings, cos_g, sin_g):
  #  global gamma
    points_to_track = []
    x_center = int(image.shape[0]/2)
    y_center = int(image.shape[1]/2)
    inner_radius = 50
    dg = gamma[2] - gamma[1]
    dr = 2

#    cos_g = [cos(g) for g in gamma]
#    for ring in range(num_rings):
#       for g in gamma:
#          new_point = [y_center - int((inner_radius+ring*dr)*math.sin(g)), x_center - int((inner_radius+ring*dr)*math.cos(g))]
#          points_to_track.append(new_point)

    for ring in range(num_rings):
       for i in range(gamma_size):
          new_point = [y_center - int((inner_radius+ring*dr)*sin_g[i]), x_center - int((inner_radius+ring*dr)*cos_g[i])]
          points_to_track.append(new_point)
    points_to_track = np.array(points_to_track, dtype=np.float32) # note: float32 required for opencv optic flow calculations
    points_to_track = points_to_track.reshape(points_to_track.shape[0], 1, points_to_track.shape[1]) # for some reason this needs to be shape (npoints, 1, 2)
    return points_to_track

def average_ring_flow(self, num_rings, gamma_size, gamma, flow):
 #   global gamma
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
            OF_reformat[i] = -total_OF_tang[(gamma_size + gamma_size//2 -1)-i]

    return OF_reformat


# Fourier Residual Method
def control_calc(num_harmonics, gamma_size, gamma, Qdot_meas):
    a_0 = 0.0

    # Controller Parameters
    k_0 = .5
    c_psi = .1
    c_d = .1
#    global gamma
    dg = gamma[2] - gamma[1]
    Qdot_WF = [0]*gamma_size
    Qdot_SF = [0]*gamma_size

    a = [0.0]*num_harmonics
    b = [0.0]*num_harmonics
    # Initialize the Coefficients
    for n in range(num_harmonics):
        a[n] = 0.0
        b[n] = 0.0



    # Compute a_0
    for i in range(gamma_size):
        a_0 = a_0 + math.cos(0*gamma[i])*Qdot_meas[i]
    a_0 = a_0*dg/math.pi

    # Compute the rest of the coefficients
    for n in range(num_harmonics):
        for i in range(gamma_size):
            a[n] = a[n] + math.cos((n+1)*gamma[i])*Qdot_meas[i]
            b[n] = b[n] + math.sin((n+1)*gamma[i])*Qdot_meas[i]
        a[n] = a[n]*dg/math.pi
        b[n] = b[n]*dg/math.pi

    # Calculate Qdot_WF
    for i in range(gamma_size):
        for n in range(num_harmonics):
            Qdot_WF[i] = Qdot_WF[i] + a[n]*math.cos((n+1)*gamma[i]) + b[n]*math.sin((n+1)*gamma[i])
        Qdot_WF[i] = Qdot_WF[i] + a_0/2.0

    # Calculate Qdot_SF
    Qdot_SF = np.subtract(Qdot_meas, Qdot_WF)

    mean = np.mean(Qdot_SF)
    std_dev = 0.0
    for i in range(gamma_size):
        std_dev += np.square(Qdot_SF[i] - mean)

    std_dev = np.sqrt(std_dev/gamma_size)
    # Dynamic Threshold
#    min_threshold = 3*std_dev
    # Static Threshold
    min_threshold = .3

    # Extract r_0 and d_0 from SF signal
    index_max = np.argmax(Qdot_SF)
    d_0 = Qdot_SF[index_max]


    if d_0 > min_threshold:
        r_0 = gamma[index_max]
        # Calculate the control signal
        yaw_rate_cmd = k_0*np.sign(r_0)*math.exp(-c_psi*math.fabs(r_0))*math.exp(-c_d*1/(math.fabs(d_0)))
    else:
        yaw_rate_cmd = 0.0

    return yaw_rate_cmd, Qdot_SF, min_threshold



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
        self.FR_pub = rospy.Publisher("FR_data", FOF_and_ResidualMsg, queue_size=10)
 
        # Define image size parameters
        self.rows = 0
        self.cols = 0
        self.num_rings = 5
        self.gamma_size =  60
        self.num_harmonics = 4
        yaw_rate_cmd = 0
        self.OF_tang_prev = [0.0]*self.gamma_size
        self.OF_tang_prev_filtered = [0.0]*self.gamma_size
        self.OF_tang_curr = [0.0]*self.gamma_size
        self.gamma_list = np.linspace(-math.pi, math.pi-.017, self.gamma_size)
        self.cos_g = [0.0]*self.gamma_size
        self.sin_g = [0.0]*self.gamma_size
        for i in range(self.gamma_size):
	    self.cos_g[i] = math.cos(self.gamma_list[i])
            self.sin_g[i] = math.sin(self.gamma_list[i])

    def image_callback(self,image):
        rospy.loginfo_throttle(2,"Streaming")
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

            # Flip the image (mirror vertically)
            curr_image = cv2.flip(curr_image, 1)

            # optional: apply gaussian blur
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
                self.points_to_track = define_rings_at_which_to_track_optic_flow(curr_image, self.gamma_size, self.gamma, self.num_rings, self.cos_g, self.sin_g)
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
            self.OF_tang_prev = self.OF_tang_curr
            self.OF_tang_curr = average_ring_flow(self, self.num_rings, self.gamma_size, self.gamma_list, flow)

            # Compute Fourier coefficients and yaw control Command
            FR_yaw_rate_cmd, Qdot_SF, FR_thresh = control_calc(self.num_harmonics, self.gamma_size, self.gamma_list, self.OF_tang_curr)

            ##### ADD ALL NECESSARY MESSAGES ######
            msg = FOF_and_ResidualMsg()
            msg.Qdot_meas = self.OF_tang_curr
#            msg.FOF_OF_SF = R_FOF
#            msg.FOF_threshold = FOF_thresh
#            msg.FOF_yaw_rate_cmd = FOF_yaw_rate_cmd
            msg.FR_Qdot_SF = Qdot_SF
            msg.FR_yaw_rate_cmd = FR_yaw_rate_cmd
            msg.FR_threshold = FR_thresh

            self.FR_pub.publish(msg)

            # Publish yaw rate command
            msg = YawRateCmdMsg()
            msg.header.stamp = rospy.Time.now()
            msg.yaw_rate_cmd = FR_yaw_rate_cmd
            self.yaw_rate_cmd_pub.publish(msg)


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




