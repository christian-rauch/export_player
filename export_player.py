#!/usr/bin/env python
from __future__ import print_function
import rospy
import cv2
import argparse
import glob
import os
import json
from std_msgs.msg import Header
from sensor_msgs.msg import CompressedImage, Image, CameraInfo, JointState
from rosgraph_msgs.msg import Clock
from geometry_msgs.msg import TransformStamped, Vector3, Quaternion
import cv_bridge
from itertools import cycle
import time
import genpy
import tf2_ros
import numpy as np
from tf.transformations import *
from std_srvs.srv import Empty, EmptyRequest, EmptyResponse


class Player:
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("data_path", type=str, help="path to data")
        parser.add_argument("-l", "--loop", action='store_true', help="loop over files")
        parser.add_argument("-s", "--service", action='store_true', help="publish files on service")
        parser.add_argument("-f", "--frequency", type=float, default=30.0, help="replay frequency")
        parser.add_argument("-i", "--index", type=int, default=None, help="index of image")
        args = parser.parse_args()

        clist = sorted(glob.glob(os.path.join(args.data_path, "colour", "*.png")), key=lambda x: int(filter(str.isdigit, x)))
        dlist = sorted(glob.glob(os.path.join(args.data_path, "depth", "*.png")), key=lambda x: int(filter(str.isdigit, x)))

        js = np.genfromtxt(os.path.join(args.data_path, "sol_joints.csv"), dtype=None, encoding="utf8", names=True)
        self.joint_names = js.dtype.names

        if args.index is not None:
            clist = [clist[args.index]]
            dlist = [dlist[args.index]]
            js    = [js[args.index]]

        with open(os.path.join(args.data_path, "camera_parameters.json")) as f:
            cp = json.load(f)

        assert(len(clist)==len(dlist))

        print("samples:", len(clist))

        camera_pose_mat = np.loadtxt(os.path.join(args.data_path, "camera_pose.csv"), delimiter=' ')
        # transformation from camera to world
        camera_pose_mat = np.linalg.inv(camera_pose_mat)


        rospy.init_node("export_player")

        self.pub_clock = rospy.Publisher("/clock", Clock, queue_size=1)
        self.pub_colour = rospy.Publisher("/camera/rgb/image_rect_color/compressed", CompressedImage, queue_size=1)
        self.pub_ci_colour = rospy.Publisher("/camera/rgb/camera_info", CameraInfo, latch=True, queue_size=1)
        # pub_depth = rospy.Publisher("/camera/depth/image_rect_raw/compressed", CompressedImage, queue_size=1)
        self.pub_depth = rospy.Publisher("/camera/depth/image_rect_raw", Image, queue_size=1)
        self.pub_ci_depth = rospy.Publisher("/camera/depth/camera_info", CameraInfo, latch=True, queue_size=1)
        self.pub_joints = rospy.Publisher("/joint_states", JointState, latch=True, queue_size=1)

        self.ci = CameraInfo(width=cp['width'], height=cp['height'],
                        K=[cp['fu'],0,cp['cx'],0,cp['fv'], cp['cy'],0,0,1],
                        P=[cp['fu'],0,cp['cx'],0,0,cp['fv'],cp['cy'],0,0,0,1,0])

        camera_frame = "true/camera_rgb_optical_frame"
        base_frame = "true/world_frame"

        # T_cw
        camera_pose = TransformStamped()
        camera_pose.transform.translation = Vector3(x=camera_pose_mat[0,3], y=camera_pose_mat[1,3], z=camera_pose_mat[2,3])
        q = quaternion_from_matrix(camera_pose_mat) # xyzw
        camera_pose.transform.rotation = Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])
        camera_pose.child_frame_id = base_frame
        camera_pose.header.frame_id = camera_frame
        self.camera_pose = camera_pose

        self.cvbridge = cv_bridge.CvBridge()

        self.broadcaster = tf2_ros.StaticTransformBroadcaster()

        if args.loop:
            file_list = cycle(zip(clist, dlist, js))
        else:
            file_list = zip(clist, dlist, js)

        if args.service:
            self.list_iter = iter(file_list)
            rospy.Service("next", Empty, self.next_set)
            rospy.spin()
        else:
            for cpath, dpath, jp in file_list:
                self.send(cpath, dpath, jp)
                if rospy.is_shutdown():
                    break
                time.sleep(1/float(args.frequency))

    def next_set(self, req):
        try:
            cpath, dpath, jp = self.list_iter.next()
            self.send(cpath, dpath, jp)
        except StopIteration:
            print("end of list")
            rospy.signal_shutdown("end of file")
        return EmptyResponse()

    def send(self, cpath, dpath, jp):
        now = genpy.Time().from_sec(time.time())
        hdr = Header(stamp=now, frame_id=self.camera_pose.header.frame_id)

        self.pub_clock.publish(clock=now)

        # print("cimg:", cpath)

        self.camera_pose.header.stamp = hdr.stamp
        self.broadcaster.sendTransform(self.camera_pose)

        cimg = cv2.imread(cpath, cv2.IMREAD_UNCHANGED)  # bgr8
        msg_cimg = self.cvbridge.cv2_to_compressed_imgmsg(cimg, dst_format="jpg")
        msg_cimg.header = hdr

        dimg = cv2.imread(dpath, cv2.IMREAD_UNCHANGED)
        # msg_dimg = cvbridge.cv2_to_compressed_imgmsg(dimg, dst_format="png")
        msg_dimg = self.cvbridge.cv2_to_imgmsg(dimg, encoding="mono16")
        msg_dimg.header = hdr

        self.pub_colour.publish(msg_cimg)
        self.pub_depth.publish(msg_dimg)

        self.ci.header = hdr
        self.pub_ci_colour.publish(self.ci)
        self.pub_ci_depth.publish(self.ci)

        self.pub_joints.publish(name=self.joint_names, position=jp, header=hdr)


if __name__ == '__main__':
    Player()
