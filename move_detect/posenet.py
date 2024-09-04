#!/usr/bin/env python3
#
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
#

import sys
import argparse
import numpy as np
from jetson_inference import poseNet
from jetson_utils import videoSource, videoOutput, Log, cudaDrawCircle

# parse the command line
parser = argparse.ArgumentParser(description="Run pose estimation DNN on a video/image stream.", 
                                 formatter_class=argparse.RawTextHelpFormatter, 
                                 epilog=poseNet.Usage() + videoSource.Usage() + videoOutput.Usage() + Log.Usage())

parser.add_argument("input", type=str, default="", nargs='?', help="URI of the input stream")
parser.add_argument("output", type=str, default="", nargs='?', help="URI of the output stream")
parser.add_argument("--network", type=str, default="resnet18-body", help="pre-trained model to load (see below for options)")
parser.add_argument("--overlay", type=str, default="links,keypoints", help="pose overlay flags (e.g. --overlay=links,keypoints)\nvalid combinations are:  'links', 'keypoints', 'boxes', 'none'")
parser.add_argument("--threshold", type=float, default=0.15, help="minimum detection threshold to use") 

try:
	args = parser.parse_known_args()[0]
except:
	print("")
	parser.print_help()
	sys.exit(0)

# load the pose estimation model
net = poseNet(args.network, sys.argv, args.threshold)

# create video sources & outputs
input = videoSource(args.input, argv=sys.argv)
output = videoOutput(args.output, argv=sys.argv)
FLENGTH = 5
left_filter = np.zeros(FLENGTH)
right_filter = np.zeros(FLENGTH)
cnt = 0
full = False

while True:
    img = input.Capture()
    if img is None: # timeout
        continue  
    poses = net.Process(img, overlay="none")
    print("detected {:d} objects in image".format(len(poses)))
    for pose in poses:
        if pose.ID == 0: # detekuj jen prvniho
            left_wrist_idx=pose.FindKeypoint(9) # leve zapesti -POZNAMKA - je nutne vzit v potaz otoceni obrazu
            right_wrist_idx=pose.FindKeypoint(10) # prave zapesti -POZNAMKA - je nutne vzit v potaz otoceni obrazu
            if left_wrist_idx>0 and right_wrist_idx>0:
                left_filter[cnt%FLENGTH] = pose.Keypoints[left_wrist_idx].y
                right_filter[cnt%FLENGTH] = pose.Keypoints[right_wrist_idx].y
                cnt += 1
                if cnt>=FLENGTH:
                    print("f{cnt}>>>{left_wrist_idx}<>{right_wrist_idx}")
                
                    if left_filter.mean() > right_filter.mean:
                        print("RIGHT")
                        cudaDrawCircle(img, (pose.Keypoints[right_wrist_idx].x,pose.Keypoints[right_wrist_idx].y), 50, (0,255, 127,200)) # (cx,cy), radius, color
                        cudaDrawCircle(img, (pose.Keypoints[left_wrist_idx].x,pose.Keypoints[left_wrist_idx].y), 50, (255,0,127,200)) # (cx,cy), radius, color
                    else:
                        print("LEFT")
                        cudaDrawCircle(img, (pose.Keypoints[right_wrist_idx].x,pose.Keypoints[right_wrist_idx].y), 50, (255,0, 127,200)) # (cx,cy), radius, color
                        cudaDrawCircle(img, (pose.Keypoints[left_wrist_idx].x,pose.Keypoints[left_wrist_idx].y), 50, (0,255,127,200)) # (cx,cy), radius, color
                    

    # render the image
    output.Render(img)

    # update the title bar
    output.SetStatus("{:s} | Network {:.0f} FPS".format(args.network, net.GetNetworkFPS()))

    # print out performance info
    #net.PrintProfilerTimes()

    # exit on input/output EOS
    if not input.IsStreaming() or not output.IsStreaming():
        break
