import argparse
import time
import signal
import sys
import numpy as np
import torch

import cv2
from PIL import Image, ImageDraw

from pid import PID
from ball_detection import ball_tracking_algo, get_instance_segmentation_model

import pantilthat as pth

from imutils.video import VideoStream

import matplotlib.pyplot as plt

SERVO_LIMIT = 90
img_x = 320
img_y = 240
p_pan, i_pan, d_pan = 0.006, 0.0027, 0.0041
p_tilt, i_tilt, d_tilt = 0.01, 0.0025, 0.001

def signal_handler(sig, frame):
    print("[INFO] Exiting...")

    pth.servo_enable(1, False)
    pth.servo_enable(2, False)
    pth.set_all(0, 0, 0, 0)
    pth.show()

    sys.exit()

def main(args):
    signal.signal(signal.SIGINT, signal_handler) # catch keyboard interrupt
    
    pth.servo_enable(1, True)
    pth.servo_enable(2, True)
        
    lighton()
    
    # start video stream
    pth.pan(0)
    pth.tilt(0)
    obj_vid = VideoStream(src = 0, usePiCamera=False, framerate = 60).start()
    time.sleep(2.0) # wait for camera to load
    vid = cv2.VideoWriter("./view1.avi", cv2.VideoWriter_fourcc(*'MJPG'), 20, (640, 480))
    model = get_instance_segmentation_model()
    obj_pid_pan = PID(p_pan, i_pan, d_pan)
    obj_pid_tilt = PID(p_tilt, i_tilt, d_tilt)
    output_pan = 0
    output_tilt = 0

    error_x = []
    error_y = []
    t = []

    ball_x = []
    ball_y = []

    starttime = time.time()

    while True:
        
        if time.time() - starttime > 15:
          break

        #time.sleep(0.01)
        img_frame = obj_vid.read() # read image from camera
#        img_frame = img_frame[:, :, ::-1]
#        print(len(img_frame), len(img_frame[0]), len(img_frame[0][0]))
        ballexist, b = ball_tracking_algo(img_frame, model)
        print("bounding box: ", b)

        if ballexist:
          ball_x.append((b[0] + b[2])/2)
          ball_y.append((b[1] + b[3])/2)

          if len(ball_x) > 3:
            ball_x.pop(0)
            ball_y.pop(0)

          ball_x_m = np.mean(ball_x)
          ball_y_m = np.mean(ball_y)
          print("center = ", ball_x_m, ball_y_m)
          img = Image.fromarray(img_frame)
          draw = ImageDraw.Draw(img)
          draw.rectangle(b, outline='red')
          draw.rectangle([ball_x_m-10, ball_y_m-10, ball_x_m+10, ball_y_m+10], outline='blue')
          
          vid.write(np.array(img))


#          img.save("img_frame.jpg")

          t.append(time.time())

          error = img_x - ball_x_m
          error_x.append(error)
          output_pan += obj_pid_pan.update(error) # calculate servo angle from controller

          error = img_y - ball_y_m
          error_y.append(error)
          output_tilt += obj_pid_tilt.update(error) # calculate servo angle from controller

          print(output_pan, output_tilt)

          pan_angle = -1*output_pan
          tilt_angle = output_tilt

          pan_angle = check_joint_limit(pan_angle, -1*SERVO_LIMIT, SERVO_LIMIT)
          pth.pan(pan_angle)

          tilt_angle = check_joint_limit(tilt_angle, -1*SERVO_LIMIT, SERVO_LIMIT)
          pth.tilt(tilt_angle)

          print("pan_angle = ", pan_angle, "tilt_angle = ", tilt_angle)

    pth.servo_enable(1, False)
    pth.servo_enable(2, False)
    pth.set_all(0, 0, 0, 0)
    pth.show()

    vid.release()

    showgraph(t, error_x, error_y)

    input()

    sys.exit()

# servo controller
def check_joint_limit(val, minimum, maximum):
    # checks joint limit
    # TODO: min-max algorithm
    if (val < minimum):
        val = minimum
    elif (val > maximum):
        val = maximum
    return val

def lighton():
    pth.light_mode(pth.WS2812)
#    pth.brightness(255)
#    pth.light_type(0)
    pth.set_all(255, 255, 255, 255)
    pth.show()

def showgraph(t, x, y):
    plt.subplot(2, 1, 1)
    plt.title("Pan")
    plt.plot(t, x, '.-')
    plt.grid()
    
    plt.subplot(2, 1, 2)
    plt.title("Tilt")
    plt.plot(t, y, '.-')
    plt.grid()

    plt.tight_layout()
    plt.savefig("pan(p={} i={} d={}) tilt(p={} i={} d={}).png".format(p_pan, i_pan, d_pan, p_tilt, i_tilt, d_tilt), dpi = 300)
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    args = parser.parse_args()
    main(args)
