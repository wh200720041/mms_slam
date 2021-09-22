#!/usr/bin/env python
import os
import sys
import threading
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pdb
import rospy
from sensor_msgs.msg import Image
from mmdet.apis import init_detector, inference_detector, show_result_ins, show_result
import time
from datetime import datetime

# Local path to trained weights file
ROOT_PATH = os.path.join(os.environ['HOME'], 'catkin_ws/src/mms_slam/')
MODEL_PATH = 'config/trained_model.pth'
CONFIG_PATH = 'config/training_param.py'
ROS_HOME = os.environ.get('ROS_HOME', os.path.join(os.environ['HOME'], ROOT_PATH))
RGB_TOPIC = '/camera/color/image_raw'
# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: CLASS_NAMES.index('teddy bear')
CLASS_NAMES = ['BG', 'person', 'agv']


class SOLOv2Node(object):
    def __init__(self):
        self.skip_frame = rospy.get_param('~skip_frame', 0)
        self.counter = self.skip_frame
        rospy.loginfo('skip %d frame' % self.skip_frame)
        # Get root path and model path
        root_path = rospy.get_param('~root_path', ROOT_PATH)
        model_path = rospy.get_param('~model_path', MODEL_PATH)
        checkpoint_file = os.path.join(root_path, model_path)

        config_path = rospy.get_param('~config_path', CONFIG_PATH)
        config_file = os.path.join(root_path, config_path)

        # Get input RGB topic.
        self._rgb_input_topic = rospy.get_param('~input', RGB_TOPIC)

        self._visualization = rospy.get_param('~visualization', True)

        self._score_thr = 0.30

        # Create model object in inference mode.
        self._model = init_detector(config_file, checkpoint_file, device='cuda:0')

        self._class_names = rospy.get_param('~class_names', CLASS_NAMES)

        self._last_msg = None
        self._msg_lock = threading.Lock()

        # self._class_colors = visualize.random_colors(len(CLASS_NAMES))

        self._publish_rate = rospy.get_param('~publish_rate', 100)

    def imgmsg_to_cv2(self, img_msg):
        if img_msg.encoding != "rgb8":
            rospy.logwarn_once("This Coral detect node has been hardcoded to the 'bgr8' encoding.  Come change the code if you're actually trying to implement a new camera")
        dtype = np.dtype("uint8") # Hardcode to 8 bits...
        dtype = dtype.newbyteorder('>' if img_msg.is_bigendian else '<')
        image_opencv = np.ndarray(shape=(img_msg.height, img_msg.width, 3), # and three channels of data. Since OpenCV works with bgr natively, we don't need to reorder the channels.
                        dtype=dtype, buffer=img_msg.data)
        # If the byt order is different between the message and the system.
        if img_msg.is_bigendian == (sys.byteorder == 'little'):
            image_opencv = image_opencv.byteswap().newbyteorder()
        image_reverse = np.flip(image_opencv, axis=2)

        return image_reverse

    def cv2_to_imgmsg(self, cv_image):
        img_msg = Image()
        img_msg.height = cv_image.shape[0]
        img_msg.width = cv_image.shape[1]
        img_msg.encoding = "bgr8"
        img_msg.is_bigendian = 0
        img_msg.data = cv_image.tostring()
        img_msg.step = len(img_msg.data) // img_msg.height # That double line is actually integer division, not a comment
        return img_msg

    def run(self):
        self._mask_pub = rospy.Publisher('~mask', Image, queue_size=1)
        vis_pub = rospy.Publisher('~visualization', Image, queue_size=1)

        rospy.Subscriber(self._rgb_input_topic, Image,
                         self._image_callback, queue_size=1)

        rate = rospy.Rate(self._publish_rate)
        while not rospy.is_shutdown():
            if self._msg_lock.acquire(False):
                msg = self._last_msg
                self._last_msg = None
                self._msg_lock.release()
            else:
                rate.sleep()
                continue

            if msg is not None:
                np_image = self.imgmsg_to_cv2(msg)
                # print(np_image.shape)
                # Run detection
                start_time = time.time()
                result = inference_detector(self._model, np_image)
                detect_time = time.time() - start_time
                rospy.logwarn_once('Time needed for segmentation: %.3f s' % detect_time)
                result_msg = self._build_result_msg(msg, result)
                self._mask_pub.publish(result_msg)

                # Visualize results
                if self._visualization:  
                    if not result or result == [None]:
                        vis_pub.publish(msg)
                    else:
                        cv_result = show_result_ins(np_image, result, self._model.CLASSES, self._score_thr)
                        image_msg = self.cv2_to_imgmsg(cv_result)
                        vis_pub.publish(image_msg)
                        #print('Time needed for segmentation:')
                        
            rate.sleep()

    def _build_result_msg(self, msg, result):
        result_msg = Image()
        result_msg.header = msg.header
        #print('Time needed for segmentation: %.3f s' % msg.header)
        result_msg.encoding = "mono8"
        if not result or result == [None]:
            result_msg.height = 720
            result_msg.width = 1280
            result_msg.step = result_msg.width
            result_msg.is_bigendian = False
            mask_sum = np.zeros(shape=(1280,720),dtype=np.uint8)
            result_msg.data = mask_sum.tobytes()
            return result_msg
        cur_result = result[0]
        seg_label = cur_result[0]
        seg_label = seg_label.cpu().numpy().astype(np.uint8)
        cate_label = cur_result[1]
        cate_label = cate_label.cpu().numpy()
        score = cur_result[2].cpu().numpy()


        vis_inds = score > self._score_thr
        seg_label = seg_label[vis_inds]
        result_msg.height = seg_label.shape[1]
        result_msg.width = seg_label.shape[2]
        result_msg.step = result_msg.width
        result_msg.is_bigendian = False
        num_mask = seg_label.shape[0]
        cate_label = cate_label[vis_inds]
        cate_score = score[vis_inds]

        mask_sum = np.zeros(shape=(result_msg.height,result_msg.width),dtype=np.uint8)
        for i in range(num_mask):
            class_id = cate_label[i] # 0,1,2
            class_name = self._class_names[class_id]  #class name
            score = cate_score[i] #correponding score
            mask_sum += seg_label[i, :, :] * (class_id+1)
            # if class_id==1:
            #     mask_sum += seg_label[i, :, :] * (class_id+1)*20 + seg_label[i, :, :] * 150
            # else:
            #     mask_sum += seg_label[i, :, :] * (class_id+1)*20 + seg_label[i, :, :] * 50

        result_msg.data = mask_sum.tobytes()
        return result_msg



    def _image_callback(self, msg):
        # rospy.logwarn("Get an image")
        self.counter -= 1
        if self.counter >= 0:
            return
        self.counter = self.skip_frame 
        if self._msg_lock.acquire(False):
            self._last_msg = msg
            self._msg_lock.release()


def main():
    rospy.init_node('solo_v2')

    node = SOLOv2Node()
    node.run()


if __name__ == '__main__':
    main()
