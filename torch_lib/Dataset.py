import cv2
import numpy as np
import os
import random

import torch
from torchvision import transforms
from torch.utils import data

from library.File import *

from .ClassAverages import ClassAverages


def read_flo_file(filename):
    f = open(filename, 'rb')
    magic = np.fromfile(f, np.float32, count=1)
    data2d = None
    if 202021.25 != magic:
        print('Magic number incorrect. Invalid .flo file')
    else:
        w = np.fromfile(f, np.int32, count=1)
        h = np.fromfile(f, np.int32, count=1)
        data2d = np.fromfile(f, np.float32, count=int(2 * w * h))
        data2d = np.resize(data2d, (h[0], w[0], 2))
    f.close()
    return data2d


# TODO: clean up where this is
def generate_bins(bins):
    angle_bins = np.zeros(bins)
    interval = 2 * np.pi / bins
    for i in range(1,bins):
        angle_bins[i] = i * interval
    angle_bins += interval / 2 # center of the bin

    return angle_bins

class Dataset(data.Dataset):
    def __init__(self, path, bins=2, overlap=0.1):

        self.top_label_path = path + "/label_2/"
        self.top_img_path = path + "/image_2/"
        self.prev_img_path = path + "/prev_2/"
        self.top_calib_path = path + "/calib/"
        self.flow_path = '/mnt/lustre/dingmingyu/Research/geo/pytorch-pwc/output_kitti_1/'
        # use a relative path instead?

        # TODO: which camera cal to use, per frame or global one?
        self.proj_matrix = get_P(os.path.abspath(os.path.dirname(os.path.dirname(__file__)) + '/camera_cal/calib_cam_to_cam.txt'))

        self.ids = [x.split('.')[0] for x in sorted(os.listdir(self.top_img_path))] # name of file
        self.num_images = len(self.ids)

        # create angle bins
        self.bins = bins
        self.angle_bins = np.zeros(bins)
        self.interval = 2 * np.pi / bins
        for i in range(1,bins):
            self.angle_bins[i] = i * self.interval
        self.angle_bins += self.interval / 2 # center of the bin

        self.overlap = overlap
        # ranges for confidence
        # [(min angle in bin, max angle in bin), ... ]
        self.bin_ranges = []
        for i in range(0,bins):
            self.bin_ranges.append(( (i*self.interval - overlap) % (2*np.pi), \
                                (i*self.interval + self.interval + overlap) % (2*np.pi)) )

        # hold average dimensions
        class_list = ['Car', 'Van', 'Truck', 'Pedestrian','Person_sitting', 'Cyclist', 'Tram', 'Misc']
        self.averages = ClassAverages(class_list)

        self.object_list = self.get_objects(self.ids)

        # pre-fetch all labels
        self.labels = {}
        last_id = ""
        for obj in self.object_list:
            id = obj[0]
            line_num = obj[1]
            label = self.get_label(id, line_num)
            if id != last_id:
                self.labels[id] = {}
                last_id = id

            self.labels[id][str(line_num)] = label

        # hold one image at a time
        self.curr_id = ""
        self.curr_img = None


    # should return (Input, Label)
    def __getitem__(self, index):
        id = self.object_list[index][0]
        line_num = self.object_list[index][1]

        if id != self.curr_id:
            self.curr_id = id
            self.curr_img = cv2.imread(self.top_img_path + '%s.png'%id)
            self.pre_img = cv2.imread(self.prev_img_path + '%s_01.png'%id)
            self.flow = read_flo_file(self.flow_path + '%d.flo'%int(id))

        label = self.labels[id][str(line_num)]
        # P doesn't matter here
        obj = DetectedObject(self.curr_img, self.pre_img, label['Class'], label['Box_2D'], self.proj_matrix, label=label, flow = self.flow)

        return obj.img, obj.prev, label

    def __len__(self):
        return len(self.object_list)

    def get_objects(self, ids):
        objects = []
        for id in ids:
            with open(self.top_label_path + '%s.txt'%id) as file:
                for line_num,line in enumerate(file):
                    line = line[:-1].split(' ')
                    obj_class = line[0]
                    if obj_class == "DontCare":
                        continue

                    dimension = np.array([float(line[8]), float(line[9]), float(line[10])], dtype=np.double)
                    self.averages.add_item(obj_class, dimension)

                    objects.append((id, line_num))


        self.averages.dump_to_file()
        return objects


    def get_label(self, id, line_num):
        lines = open(self.top_label_path + '%s.txt'%id).read().splitlines()
        label = self.format_label(lines[line_num])

        return label

    def get_bin(self, angle):

        bin_idxs = []

        def is_between(min, max, angle):
            max = (max - min) if (max - min) > 0 else (max - min) + 2*np.pi
            angle = (angle - min) if (angle - min) > 0 else (angle - min) + 2*np.pi
            return angle < max

        for bin_idx, bin_range in enumerate(self.bin_ranges):
            if is_between(bin_range[0], bin_range[1], angle):
                bin_idxs.append(bin_idx)

        return bin_idxs

    def format_label(self, line):
        line = line[:-1].split(' ')

        Class = line[0]

        for i in range(1, len(line)):
            line[i] = float(line[i])

        Alpha = line[3] # what we will be regressing
        Ry = line[14]
        top_left = (int(round(line[4])), int(round(line[5])))
        bottom_right = (int(round(line[6])), int(round(line[7])))
        Box_2D = [top_left, bottom_right]

        Dimension = np.array([line[8], line[9], line[10]], dtype=np.double) # height, width, length
        # modify for the average
        Dimension -= self.averages.get_item(Class)

        Location = [line[11], line[12], line[13]] # x, y, z
        Location[1] -= Dimension[0] / 2 # bring the KITTI center up to the middle of the object

        Orientation = np.zeros((self.bins, 2))
        Confidence = np.zeros(self.bins)

        # alpha is [-pi..pi], shift it to be [0..2pi]
        angle = Alpha + np.pi

        bin_idxs = self.get_bin(angle)

        for bin_idx in bin_idxs:
            angle_diff = angle - self.angle_bins[bin_idx]

            Orientation[bin_idx,:] = np.array([np.cos(angle_diff), np.sin(angle_diff)])
            Confidence[bin_idx] = 1

        label = {
                'Class': Class,
                'Box_2D': Box_2D,
                'Dimensions': Dimension,
                'Alpha': Alpha,
                'Orientation': Orientation,
                'Confidence': Confidence
                }

        return label

    # will be deprc soon
    def parse_label(self, label_path):
        buf = []
        with open(label_path, 'r') as f:
            for line in f:
                line = line[:-1].split(' ')

                Class = line[0]
                if Class == "DontCare":
                    continue

                for i in range(1, len(line)):
                    line[i] = float(line[i])

                Alpha = line[3] # what we will be regressing
                Ry = line[14]
                top_left = (int(round(line[4])), int(round(line[5])))
                bottom_right = (int(round(line[6])), int(round(line[7])))
                Box_2D = [top_left, bottom_right]

                Dimension = [line[8], line[9], line[10]] # height, width, length
                Location = [line[11], line[12], line[13]] # x, y, z
                Location[1] -= Dimension[0] / 2 # bring the KITTI center up to the middle of the object

                buf.append({
                        'Class': Class,
                        'Box_2D': Box_2D,
                        'Dimensions': Dimension,
                        'Location': Location,
                        'Alpha': Alpha,
                        'Ry': Ry
                    })
        return buf

#     # will be deprc soon
#     def all_objects(self):
#         data = {}
#         for id in self.ids:
#             data[id] = {}
#             img_path = self.top_img_path + '%s.png'%id
#             prev_path = self.prev_img_path + '%s_01.png'%id
#             img = cv2.imread(img_path)
#             prev = cv2.imread(prev_path)
#             data[id]['Image'] = img
#             data[id]['Prev'] = prev

#             # using p per frame
#             calib_path = self.top_calib_path + '%s.txt'%id
#             proj_matrix = get_calibration_cam_to_image(calib_path)

#             # using P_rect from global calib file
#             proj_matrix = self.proj_matrix

#             data[id]['Calib'] = proj_matrix

#             label_path = self.top_label_path + '%s.txt'%id
#             labels = self.parse_label(label_path)
#             objects = []
#             for label in labels:
#                 box_2d = label['Box_2D']
#                 detection_class = label['Class']
#                 objects.append(DetectedObject(img, detection_class, box_2d, proj_matrix, label=label))

#             data[id]['Objects'] = objects

#         return data


"""
What is *sorta* the input to the neural net. Will hold the cropped image and
the angle to that image, and (optionally) the label for the object. The idea
is to keep this abstract enough so it can be used in combination with YOLO
"""
class DetectedObject:
    def __init__(self, img, prev, detection_class, box_2d, proj_matrix, label=None, flow=None):

        if isinstance(proj_matrix, str): # filename
        # Change here: which camera cal to use, per frame or global one? ##############
#             proj_matrix = get_P(proj_matrix)
            proj_matrix = get_calibration_cam_to_image(proj_matrix)

        self.proj_matrix = proj_matrix
        self.theta_ray = self.calc_theta_ray(img, box_2d, proj_matrix)
        self.img = self.format_img(img, box_2d, None)
        self.prev = self.format_img(prev, box_2d, flow)
        self.label = label
        self.detection_class = detection_class

    def calc_theta_ray(self, img, box_2d, proj_matrix):
        width = img.shape[1]
        fovx = 2 * np.arctan(width / (2 * proj_matrix[0][0]))
        center = (box_2d[1][0] + box_2d[0][0]) / 2
        dx = center - (width / 2)

        mult = 1
        if dx < 0:
            mult = -1
        dx = abs(dx)
        angle = np.arctan( (2*dx*np.tan(fovx/2)) / width )
        angle = angle * mult

        return angle

    def format_img(self, img, box_2d, flow = None):

        # Should this happen? or does normalize take care of it. YOLO doesnt like
        # img=img.astype(np.float) / 255

        # torch transforms
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                std=[0.229, 0.224, 0.225])
        process = transforms.Compose ([
            transforms.ToTensor(),
            normalize
        ])
        width = img.shape[1]
        height = img.shape[0]
        if flow is None:
        # crop image
            pt1 = box_2d[0]
            pt2 = box_2d[1]
            print(pt1,pt2,'img',flush=True)
            crop = img[pt1[1]:pt2[1]+1, pt1[0]:pt2[0]+1]
            crop = cv2.resize(src = crop, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)
        else:
            pt1 = (int(box_2d[0][0] + flow[min(max(0,box_2d[0][1]),height-1),min(max(0,box_2d[0][0]),width-1),0]), int(box_2d[0][1] + flow[min(max(0,box_2d[0][1]),height-1),min(max(0,box_2d[0][0]),width-1),1]))
            pt2 = (int(box_2d[1][0] + flow[min(max(0,box_2d[1][1]),height-1),min(max(0,box_2d[1][0]),width-1),0]), int(box_2d[1][1] + flow[min(max(0,box_2d[1][1]),height-1),min(max(0,box_2d[1][0]),width-1),1]))
            pt1 = (min(max(0,pt1[0]),width-1),min(max(0,pt1[1]),height-1))
            pt2 = (min(max(0,pt2[0]),width-1),min(max(0,pt2[1]),height-1))
            print(pt1,pt2,'prev',flush=True)
            crop = img[pt1[1]:pt2[1]+1, pt1[0]:pt2[0]+1]
            if crop.shape[0] < 3 or crop.shape[1] <3:
                pt1 = box_2d[0]
                pt2 = box_2d[1]
                print(pt1,pt2,'prev_gg',flush=True)
                crop = img[pt1[1]:pt2[1]+1, pt1[0]:pt2[0]+1]
            crop = cv2.resize(src = crop, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)
            
        # recolor, reformat
        batch = process(crop)

        return batch
