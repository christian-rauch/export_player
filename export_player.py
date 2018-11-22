#!/usr/bin/env python
from __future__ import print_function
import rospy
import cv2
import argparse
import glob
import os
import json
from std_msgs.msg import Header, Bool, UInt64
from sensor_msgs.msg import CompressedImage, Image, CameraInfo, JointState
from rosgraph_msgs.msg import Clock
from geometry_msgs.msg import TransformStamped, Vector3, Quaternion
import cv_bridge
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
        parser.add_argument("-j", "--end_range", type=int, default=None, help="last index of range")
        parser.add_argument("-p", "--print_file", action='store_true', default=False, help="print current file name")
        parser.add_argument("-c", "--cutoff", type=int, help="cutoff depth value in millimeter")
        parser.add_argument("-t", "--time", action='store_true', help="stamp messages with real wall clock time")
        parser.add_argument("--skip", type=int, help="skip frames")
        self.args = parser.parse_args()

        clist = sorted(glob.glob(os.path.join(self.args.data_path, "colour", "*.png")), key=lambda x: int(filter(str.isdigit, x)))
        dlist = sorted(glob.glob(os.path.join(self.args.data_path, "depth", "*.png")), key=lambda x: int(filter(str.isdigit, x)))

        if len(clist)!=len(dlist):
            raise UserWarning("number of images mismatch")
        index = np.arange(0, len(clist))

        js = None
        for jf in ["sol_joints.csv", "joints.csv"]:
            joint_file_path = os.path.join(self.args.data_path, jf)
            if os.path.isfile(joint_file_path):
                js = np.genfromtxt(joint_file_path, dtype=None, encoding="utf8", names=True)
                break

        if js is not None:
            self.joint_names = js.dtype.names
        else:
            raise UserWarning("no joint file found")

        if self.args.index is not None:
            if self.args.end_range is not None:
                clist = clist[self.args.index:self.args.end_range]
                dlist = dlist[self.args.index:self.args.end_range]
                js    = js[self.args.index:self.args.end_range]
                index = index[self.args.index:self.args.end_range]
            else:
                clist = [clist[self.args.index]]
                dlist = [dlist[self.args.index]]
                js    = [js[self.args.index]]
                index = [index[self.args.index]]

        if self.args.skip:
            clist = clist[::self.args.skip]
            dlist = dlist[::self.args.skip]
            js    = js[::self.args.skip]
            index = index[::self.args.skip]

        with open(os.path.join(self.args.data_path, "camera_parameters.json")) as f:
            cp = json.load(f)

        assert(len(clist)==len(dlist))

        self.i = 0
        self.N = len(clist)

        print("samples:", len(clist))

        # list of camera poses per image
        camera_poses = np.loadtxt(os.path.join(self.args.data_path, "camera_pose.csv"), skiprows=1, delimiter=' ')
        if camera_poses.shape[1]==7:
            # list of camera poses, px py pz, qw qx qy qz
            camera_pose = camera_poses[0]   # chose first camera pose as static
            # camera_pose_mat = np.identity(4)
            camera_pose_mat = quaternion_matrix([camera_pose[3+1], camera_pose[3+2], camera_pose[3+3], camera_pose[3+0]])  # xyzw
            camera_pose_mat[:3,3] = camera_pose[:3]
        else:
            # single static camera pose
            camera_pose_mat = np.loadtxt(os.path.join(self.args.data_path, "camera_pose.csv"), delimiter=' ')

        # transformation from camera to world
        camera_pose_mat = np.linalg.inv(camera_pose_mat)

        # time stamps in nanoseconds
        timestamps = np.loadtxt(os.path.join(self.args.data_path, "time.csv"), dtype=np.uint).tolist()

        rospy.init_node("export_player")

        self.pub_clock = rospy.Publisher("/clock", Clock, queue_size=1)
        self.pub_colour = rospy.Publisher("/camera/rgb/image_rect_color", Image, queue_size=1)
        self.pub_colour_compressed = rospy.Publisher("/camera/rgb/image_rect_color/compressed", CompressedImage, queue_size=1)
        self.pub_ci_colour = rospy.Publisher("/camera/rgb/camera_info", CameraInfo, latch=True, queue_size=1)
        self.pub_depth = rospy.Publisher("/camera/depth/image_rect_raw", Image, queue_size=1)
        self.pub_depth_compressed = rospy.Publisher("/camera/depth/image_rect_raw/compressed", CompressedImage, queue_size=1)
        self.pub_depth_compressed_depth = rospy.Publisher("/camera/depth/image_rect_raw/compressedDepth", CompressedImage, queue_size=1)
        self.pub_ci_depth = rospy.Publisher("/camera/depth/camera_info", CameraInfo, latch=True, queue_size=1)
        self.pub_joints = rospy.Publisher("/joint_states", JointState, latch=True, queue_size=1)
        self.pub_eol = rospy.Publisher("~end_of_log", Bool, queue_size=1, latch=True)

        self.pub_id = rospy.Publisher("~id", UInt64, queue_size=1, latch=True)

        self.ci = CameraInfo(width=cp['width'], height=cp['height'],
                        K=[cp['fu'],0,cp['cx'],0,cp['fv'], cp['cy'],0,0,1],
                        P=[cp['fu'],0,cp['cx'],0,0,cp['fv'],cp['cy'],0,0,0,1,0])

        self.last_filename = None
        self.new_file = True

        # T_cw
        self.camera_pose = TransformStamped()
        self.camera_pose.transform.translation = Vector3(x=camera_pose_mat[0, 3], y=camera_pose_mat[1, 3], z=camera_pose_mat[2, 3])
        q = quaternion_from_matrix(camera_pose_mat) # xyzw
        self.camera_pose.transform.rotation = Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])
        self.camera_pose.child_frame_id = "table"
        self.camera_pose.header.frame_id = "camera_rgb_optical_frame"

        self.cvbridge = cv_bridge.CvBridge()

        self.broadcaster = tf2_ros.StaticTransformBroadcaster()

        self.file_list = zip(index, timestamps, clist, dlist, js)

        if not self.args.loop:
            self.pub_eol.publish(data=False)

        if self.args.service:
            rospy.Service("~reset", Empty, self.reset_iter)
            self.reset_iter(EmptyRequest())
            rospy.Service("~next", Empty, self.next_set)
            rospy.spin()
        else:
            loop = True
            while loop and not rospy.is_shutdown():
                for self.i in range(self.N):
                    tstart = time.time()
                    ind, t, cpath, dpath, jp = self.file_list[self.i]
                    self.send(cpath, dpath, jp, t, ind)
                    dur = time.time()-tstart
                    if rospy.is_shutdown():
                        break
                    delay = (1/float(self.args.frequency)-dur)
                    delay = max(0,delay)
                    time.sleep(delay)
                loop = self.args.loop

            if not self.args.loop:
                self.pub_eol.publish(data=True)

    def reset_iter(self, req):
        self.i = 0
        if self.args.print_file and self.new_file:
            print("-----------------------------------------------")
        if not self.args.loop:
            self.pub_eol.publish(data=False)
        return EmptyResponse()

    def next_set(self, req):
        ind, t, cpath, dpath, jp = self.file_list[self.i]
        self.send(cpath, dpath, jp, t, ind)
        self.i += 1
        if self.i == self.N:
            if self.args.loop:
                # reset automatically
                self.reset_iter(EmptyRequest())
            else:
                self.pub_eol.publish(data=True)
                print("end of log")
                rospy.signal_shutdown("end of log")
        return EmptyResponse()

    def send(self, cpath, dpath, jp, time_ns, ind):
        if self.args.time:
            time_ns = int(time.time()*1e9)
        sec, nsec = divmod(time_ns, int(1e9))
        now = genpy.Time(secs=sec, nsecs=nsec)
        hdr = Header(stamp=now, frame_id=self.camera_pose.header.frame_id)

        # index of image in original list
        img_index = ind

        if self.args.print_file:
            filename = os.path.splitext(os.path.basename(cpath))[0]
            self.new_file = (self.last_filename != filename)
            if self.new_file:
                print("img("+str(img_index)+"):", filename)
            self.last_filename = filename

        self.camera_pose.header.stamp = hdr.stamp
        self.broadcaster.sendTransform(self.camera_pose)

        cimg = cv2.imread(cpath, cv2.IMREAD_UNCHANGED)  # bgr8
        msg_cimg = self.cvbridge.cv2_to_imgmsg(cimg, encoding="bgr8")
        msg_cimg.header = hdr
        msg_cimg_compr = self.cvbridge.cv2_to_compressed_imgmsg(cimg, dst_format="png")
        msg_cimg_compr.header = hdr

        dimg = cv2.imread(dpath, cv2.IMREAD_UNCHANGED)
        if self.args.cutoff:
            dimg[dimg>self.args.cutoff] = 0
        msg_dimg = self.cvbridge.cv2_to_imgmsg(dimg)
        msg_dimg.header = hdr
        msg_dimg_compr = self.cvbridge.cv2_to_compressed_imgmsg(dimg, dst_format="png")
        # should be   `format: "16UC1; png compressed "`
        # see: https://github.com/ros-perception/vision_opencv/issues/250
        msg_dimg_compr.format = "16UC1; png compressed"
        msg_dimg_compr.header = hdr
        msg_dimg_compr_depth = CompressedImage()
        msg_dimg_compr_depth.format = "16UC1; compressedDepth"
        # prepend 12bit fake header
        msg_dimg_compr_depth.data = "000000000000" + msg_dimg_compr.data
        msg_dimg_compr_depth.header = hdr

        try:
            self.pub_clock.publish(clock=now)
            self.pub_colour.publish(msg_cimg)
            self.pub_colour_compressed.publish(msg_cimg_compr)
            self.pub_depth.publish(msg_dimg)
            self.pub_depth_compressed.publish(msg_dimg_compr)
            self.pub_depth_compressed_depth.publish(msg_dimg_compr_depth)

            self.ci.header = hdr
            self.pub_ci_colour.publish(self.ci)
            self.pub_ci_depth.publish(self.ci)

            self.pub_joints.publish(name=self.joint_names, position=jp, header=hdr)

            self.pub_id.publish(data=img_index)
        except rospy.ROSException:
            pass


if __name__ == '__main__':
    Player()
