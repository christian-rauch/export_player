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
        parser.add_argument("-s", "--service", action='store_true', help="publish files on service call")
        parser.add_argument("-f", "--frequency", type=float, default=30.0, help="replay frequency")
        parser.add_argument("-i", "--index", type=int, default=None, help="start index of image range")
        parser.add_argument("-j", "--end_range", type=int, default=None, help="end index of image range")
        parser.add_argument("-p", "--print_file", action='store_true', default=False, help="print current file name")
        parser.add_argument("-c", "--cutoff", type=int, help="cutoff depth value in millimeter")
        parser.add_argument("-t", "--time", action='store_true', help="stamp messages with real wall clock time")
        parser.add_argument("-bf", "--base_frame", type=str, default="world", help="base frame")
        parser.add_argument("-cf", "--camera_frame", type=str, default="camera_rgb_optical_frame", help="camera frame")
        parser.add_argument("--skip", type=int, help="skip frames")
        parser.add_argument("--raw", action='store_true', help="decode PNG files and publish raw image messages")
        self.args = parser.parse_args()

        if not os.path.isdir(self.args.data_path):
            parser.error("directory '"+self.args.data_path+"' does not exist")

        clist = sorted(glob.glob(os.path.join(self.args.data_path, "colour", "*.png")), key=lambda x: int(filter(str.isdigit, x)))
        dlist = sorted(glob.glob(os.path.join(self.args.data_path, "depth", "*.png")), key=lambda x: int(filter(str.isdigit, x)))

        if len(clist)==0 or len(dlist)==0:
            raise UserWarning("no images found")

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

        # list of camera poses per image, order: px py pz, qw qx qy qz
        camera_poses = np.loadtxt(os.path.join(self.args.data_path, "camera_pose.csv"), skiprows=1, delimiter=' ')

        vicon_path = os.path.join(self.args.data_path, "vicon_pose_lwr_end_effector.csv")
        if os.path.exists(vicon_path):
            self.vicon_poses = np.loadtxt(os.path.join(self.args.data_path, "vicon_pose_lwr_end_effector.csv"), skiprows=0, delimiter=' ')
        else:
            self.vicon_poses = None

        rospy.init_node("export_player")

        self.pub_clock = rospy.Publisher("/clock", Clock, queue_size=1)
        if self.args.raw:
            self.pub_colour = rospy.Publisher("/camera/rgb/image_rect_color", Image, queue_size=1)
            self.pub_depth = rospy.Publisher("/camera/depth/image_rect_raw", Image, queue_size=1)
        self.pub_colour_compressed = rospy.Publisher("/camera/rgb/image_rect_color/compressed", CompressedImage, queue_size=1)
        self.pub_ci_colour = rospy.Publisher("/camera/rgb/camera_info", CameraInfo, latch=True, queue_size=1)
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

        self.cvbridge = cv_bridge.CvBridge()

        self.broadcaster = tf2_ros.TransformBroadcaster()

        if self.args.time:
            # timestamps will be ignored and replaced by current time
            timestamps = [None] * len(clist)
        else:
            # load time stamps in nanoseconds
            timestamps = np.loadtxt(os.path.join(self.args.data_path, "time.csv"), dtype=np.uint).tolist()

        self.file_list = zip(index, timestamps, clist, dlist, js, camera_poses)

        self.i = 0
        self.N = len(self.file_list)

        print("samples:", self.N)

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
                    ind, t, cpath, dpath, jp, cpose = self.file_list[self.i]
                    self.send(cpath, dpath, jp, t, ind, cpose)
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
        ind, t, cpath, dpath, jp, cpose = self.file_list[self.i]
        self.send(cpath, dpath, jp, t, ind, cpose)
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

    def send(self, cpath, dpath, jp, time_ns, ind, cpose):
        if self.args.time:
            time_ns = int(time.time()*1e9)
        sec, nsec = divmod(time_ns, int(1e9))
        now = genpy.Time(secs=sec, nsecs=nsec)
        hdr = Header(stamp=now, frame_id=self.args.camera_frame)

        # index of image in original list
        img_index = ind

        if self.args.print_file:
            filename = os.path.splitext(os.path.basename(cpath))[0]
            self.new_file = (self.last_filename != filename)
            if self.new_file:
                print("img("+str(img_index)+"):", filename)
            self.last_filename = filename

        # T_cw
        camera_pose = TransformStamped()
        camera_pose.header.stamp = hdr.stamp
        camera_pose.header.frame_id = self.args.camera_frame
        camera_pose.child_frame_id = self.args.base_frame
        # invert camera pose
        camera_pose_mat = np.dot(translation_matrix(cpose[:3]), quaternion_matrix([cpose[4], cpose[5], cpose[6], cpose[3]]))  # xyzw
        camera_pose_mat = np.linalg.inv(camera_pose_mat)
        p = translation_from_matrix(camera_pose_mat)
        q = quaternion_from_matrix(camera_pose_mat)
        camera_pose.transform.translation = Vector3(x=p[0], y=p[1], z=p[2]) # xyzw
        camera_pose.transform.rotation = Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])

        self.broadcaster.sendTransform(camera_pose)

        if self.vicon_poses is not None and img_index < len(self.vicon_poses):
            # px,py,pz,qw,qx,qy,qz
            vicon_pose = self.vicon_poses[img_index]
            vicon_tf = TransformStamped()
            vicon_tf.transform.translation = Vector3(x=vicon_pose[0], y=vicon_pose[1], z=vicon_pose[2])
            vicon_tf.transform.rotation = Quaternion(w=vicon_pose[3], x=vicon_pose[4], y=vicon_pose[5], z=vicon_pose[6])
            vicon_tf.header.frame_id = self.args.base_frame
            vicon_tf.child_frame_id = "vicon_end_effector"
            self.broadcaster.sendTransform(vicon_tf)

        with open(cpath, mode='rb') as f:
            cimg_buf = f.read()
            msg_cimg_compr = CompressedImage()
            msg_cimg_compr.header = hdr
            msg_cimg_compr.format = "png"
            msg_cimg_compr.data = cimg_buf

            if self.args.raw:
                cimg = cv2.imdecode(np.fromstring(cimg_buf, dtype=np.uint8), cv2.IMREAD_UNCHANGED) # bgr8
                msg_cimg = self.cvbridge.cv2_to_imgmsg(cimg, encoding="bgr8")

        with open(dpath, mode='rb') as f:
            dimg_buf = f.read()
            msg_dimg_compr = CompressedImage()
            msg_dimg_compr.header = hdr
            msg_dimg_compr.format = "16UC1; png compressed"
            msg_dimg_compr.data = dimg_buf

            msg_dimg_compr_depth = CompressedImage()
            msg_dimg_compr_depth.header = hdr
            msg_dimg_compr_depth.format = "16UC1; compressedDepth"
            # prepend 12bit fake header
            msg_dimg_compr_depth.data = "000000000000" + dimg_buf

            if self.args.raw:
                dimg = cv2.imdecode(np.fromstring(dimg_buf, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
                msg_dimg = self.cvbridge.cv2_to_imgmsg(dimg, encoding="16UC1")

        try:
            self.pub_clock.publish(clock=now)
            if self.args.raw:
                self.pub_colour.publish(msg_cimg)
                self.pub_depth.publish(msg_dimg)
            self.pub_colour_compressed.publish(msg_cimg_compr)
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
    try:
        Player()
    except UserWarning as e:
        print(e.message)
