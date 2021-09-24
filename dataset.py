import torch.utils.data as data
from PIL import Image
import os
import os.path
import torch
import numpy as np
import torchvision.transforms as transforms
from libs.transformations import euler_matrix
import argparse
import time
import random
import numpy.ma as ma
import copy
import math
import scipy.misc
import scipy.io as scio
import cv2
import _pickle as cPickle
from skimage.transform import resize
import matplotlib.pyplot as plt


class Dataset(data.Dataset):
    def __init__(self, mode, root, add_noise, num_pt, num_cates, count, cate_id, w_size, occlude=False):
        # num_cates is the total number of categories gonna be preloaded from dataset, cate_id is the category need to be trained.
        self.root = root
        self.add_noise = add_noise
        self.mode = mode
        self.num_pt = num_pt
        self.occlude = occlude
        self.num_cates = num_cates
        self.back_root = '{0}/train2017/'.format(self.root)
        self.w_size = w_size + 1
        self.cate_id = cate_id
        # Path list: obj_list[], real_obj_list[], back_list[],
        self.obj_list = {}
        self.obj_name_list = {}
        self.cate_set = [1, 2, 3, 4, 5, 6]
        self.real_oc_list = {}
        self.real_oc_name_set = {}
        for ca in self.cate_set:
            if ca != self.cate_id:
                real_oc_name_list = os.listdir('{0}/data_list/real_{1}/{2}/'.format(self.root, self.mode, str(ca)))
                self.real_oc_name_set[ca] = real_oc_name_list
        del self.cate_set[self.cate_id - 1]
        # Get all the occlusions.
        # print(self.real_oc_name_set)
        for key in self.real_oc_name_set.keys():
            for item in self.real_oc_name_set[key]:
                self.real_oc_list[item] = []

                input_file = open(
                    '{0}/data_list/real_{1}/{2}/{3}/list.txt'.format(self.root, self.mode, str(key), item), 'r')
                while 1:
                    input_line = input_file.readline()
                    if not input_line:
                        break
                    if input_line[-1:] == '\n':
                        input_line = input_line[:-1]
                    self.real_oc_list[item].append('{0}/data/{1}'.format(self.root, input_line))
                input_file.close()

        if self.mode == 'train':
            for tmp_cate_id in range(1, self.num_cates + 1):
                # (nxm)obj_name_list[] contains the name list of the super dir(1a9e1fb2a51ffd065b07a27512172330) of training list txt file(train/16069/0008)
                listdir = os.listdir('{0}/data_list/train/{1}/'.format(self.root, tmp_cate_id))
                self.obj_name_list[tmp_cate_id] = []
                for i in listdir:
                    if os.path.isdir('{0}/data_list/train/{1}/{2}'.format(self.root, tmp_cate_id, i)):
                        self.obj_name_list[tmp_cate_id].append(i)
                # self.obj_name_list[tmp_cate_id] = os.listdir('{0}/data_list/train/{1}/'.format(self.root, tmp_cate_id))
                self.obj_list[tmp_cate_id] = {}

                for item in self.obj_name_list[tmp_cate_id]:
                    # print(tmp_cate_id, item)# item: 1a9e1fb2a51ffd065b07a27512172330
                    self.obj_list[tmp_cate_id][item] = []

                    input_file = open('{0}/data_list/train/{1}/{2}/list.txt'.format(self.root, tmp_cate_id, item), 'r')
                    while 1:
                        input_line = input_file.readline()  # read list.txt(train/16069/0008)
                        if not input_line:
                            break
                        if input_line[-1:] == '\n':
                            input_line = input_line[:-1]
                        # (nxmxk)obj_list is the real training data from {root}/data/train/16069/0008， 0008 here is just a prefix without the 5 suffix indicate the different file like _color.png/mask.png/depth.png/meta.txt_coord.png in 16069 dir.
                        self.obj_list[tmp_cate_id][item].append('{0}/data/{1}'.format(self.root, input_line))
                    input_file.close()

        self.real_obj_list = {}
        self.real_obj_name_list = {}

        for tmp_cate_id in range(1, self.num_cates + 1):
            # real_obj_name_list contains the real obj names from {}/data_list/real_train/1/ like bottle_blue_google_norm, bottle_starbuck_norm
            self.real_obj_name_list[tmp_cate_id] = []
            listdir = os.listdir('{0}/data_list/real_{1}/{2}/'.format(self.root, self.mode, tmp_cate_id))
            for i in listdir:
                if os.path.isdir('{0}/data_list/real_{1}/{2}/{3}'.format(self.root, self.mode, tmp_cate_id, i)):
                    self.real_obj_name_list[tmp_cate_id].append(i)

            # self.real_obj_name_list[tmp_cate_id] = os.listdir('{0}/data_list/real_{1}/{2}/'.format(self.root, self.mode, tmp_cate_id))
            self.real_obj_list[tmp_cate_id] = {}

            for item in self.real_obj_name_list[tmp_cate_id]:
                # print(tmp_cate_id, item) #item : bottle_blue_google_norm
                self.real_obj_list[tmp_cate_id][item] = []
                # real_train/scene_2/0000
                input_file = open(
                    '{0}/data_list/real_{1}/{2}/{3}/list.txt'.format(self.root, self.mode, tmp_cate_id, item), 'r')

                while 1:
                    input_line = input_file.readline()
                    if not input_line:
                        break
                    if input_line[-1:] == '\n':
                        input_line = input_line[:-1]
                    # real_obj_list contains the prefix of files under the dir {}/data/real_train/scene_2/, which are all consecutive frames in video squence.
                    self.real_obj_list[tmp_cate_id][item].append('{0}/data/{1}'.format(self.root, input_line))
                input_file.close()

        self.back_list = []

        input_file = open('dataset/train2017.txt', 'r')
        while 1:
            input_line = input_file.readline()
            if not input_line:
                break
            if input_line[-1:] == '\n':
                input_line = input_line[:-1]
            # back_list is the path list of the images in COCO dataset 2017 are about to be used in the training.
            self.back_list.append(self.back_root + input_line)  # back_root is the dir of COCO dataset train2017
        input_file.close()

        self.mesh = []
        input_file = open('dataset/sphere.xyz', 'r')
        while 1:
            input_line = input_file.readline()
            if not input_line:
                break
            if input_line[-1:] == '\n':
                input_line = input_line[:-1]
            input_line = input_line.split(' ')
            self.mesh.append([float(input_line[0]), float(input_line[1]), float(input_line[2])])
        input_file.close()
        self.mesh = np.array(self.mesh) * 0.6

        self.cam_cx_1 = 322.52500
        self.cam_cy_1 = 244.11084
        self.cam_fx_1 = 591.01250
        self.cam_fy_1 = 590.16775

        self.cam_cx_2 = 319.5
        self.cam_cy_2 = 239.5
        self.cam_fx_2 = 577.5
        self.cam_fy_2 = 577.5

        self.xmap = np.array([[j for i in range(640)] for j in range(480)])
        self.ymap = np.array([[i for i in range(640)] for j in range(480)])

        self.norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        self.trancolor = transforms.ColorJitter(0.8, 0.5, 0.5, 0.05)
        self.length = count

    def get_occlusion(self, oc_obj, oc_frame, syn_or_real):
        if syn_or_real:
            cam_cx = self.cam_cx_1
            cam_cy = self.cam_cy_1
            cam_fx = self.cam_fx_1
            cam_fy = self.cam_fy_1
        else:
            cam_cx = self.cam_cx_2
            cam_cy = self.cam_cy_2
            cam_fx = self.cam_fx_2
            cam_fy = self.cam_fy_2
        cam_scale = 1.0
        oc_target = []
        oc_input_file = open('{0}/model_scales/{1}.txt'.format(self.root, oc_obj), 'r')
        for i in range(8):
            oc_input_line = oc_input_file.readline()
            if oc_input_line[-1:] == '\n':
                oc_input_line = oc_input_line[:-1]
            oc_input_line = oc_input_line.split(' ')
            oc_target.append([float(oc_input_line[0]), float(oc_input_line[1]), float(oc_input_line[2])])
        oc_input_file.close()
        oc_target = np.array(oc_target)
        r, t, _ = self.get_pose(oc_frame, oc_obj)

        oc_target_tmp = np.dot(oc_target, r.T) + t
        oc_target_tmp[:, 0] *= -1.0
        oc_target_tmp[:, 1] *= -1.0
        oc_rmin, oc_rmax, oc_cmin, oc_cmax = get_2dbbox(oc_target_tmp, cam_cx, cam_cy, cam_fx, cam_fy, cam_scale)

        oc_img = Image.open('{0}_color.png'.format(oc_frame))
        oc_depth = np.array(self.load_depth('{0}_depth.png'.format(oc_frame)))
        oc_mask = (cv2.imread('{0}_mask.png'.format(oc_frame))[:, :, 0] == 255)  # White is True and Black is False
        oc_img = np.array(oc_img)[:, :, :3]  # (3, 640, 480)
        oc_img = np.transpose(oc_img, (2, 0, 1))  # (3, 640, 480)
        oc_img = oc_img / 255.0
        oc_img = oc_img * (~oc_mask)
        oc_depth = oc_depth * (~oc_mask)

        oc_img = oc_img[:, oc_rmin:oc_rmax, oc_cmin:oc_cmax]
        oc_depth = oc_depth[oc_rmin:oc_rmax, oc_cmin:oc_cmax]
        oc_mask = oc_mask[oc_rmin:oc_rmax, oc_cmin:oc_cmax]
        return oc_img, oc_depth, oc_mask

    def divide_scale(self, scale, pts):
        pts[:, 0] = pts[:, 0] / scale[0]
        pts[:, 1] = pts[:, 1] / scale[1]
        pts[:, 2] = pts[:, 2] / scale[2]

        return pts

    def get_anchor_box(self, ori_bbox):
        bbox = ori_bbox
        limit = np.array(search_fit(bbox))
        num_per_axis = 5
        gap_max = num_per_axis - 1

        small_range = [1, 3]

        gap_x = (limit[1] - limit[0]) / float(gap_max)
        gap_y = (limit[3] - limit[2]) / float(gap_max)
        gap_z = (limit[5] - limit[4]) / float(gap_max)

        ans = []
        scale = [max(limit[1], -limit[0]), max(limit[3], -limit[2]), max(limit[5], -limit[4])]

        for i in range(0, num_per_axis):
            for j in range(0, num_per_axis):
                for k in range(0, num_per_axis):
                    ans.append([limit[0] + i * gap_x, limit[2] + j * gap_y, limit[4] + k * gap_z])

        ans = np.array(ans)
        scale = np.array(scale)

        ans = self.divide_scale(scale, ans)

        return ans, scale

    def change_to_scale(self, scale, cloud_fr, cloud_to):
        cloud_fr = self.divide_scale(scale, cloud_fr)
        cloud_to = self.divide_scale(scale, cloud_to)

        return cloud_fr, cloud_to

    def enlarge_bbox(self, target):

        limit = np.array(search_fit(target))
        longest = max(limit[1] - limit[0], limit[3] - limit[2], limit[5] - limit[4])
        longest = longest * 1.3

        scale1 = longest / (limit[1] - limit[0])
        scale2 = longest / (limit[3] - limit[2])
        scale3 = longest / (limit[5] - limit[4])

        target[:, 0] *= scale1
        target[:, 1] *= scale2
        target[:, 2] *= scale3

        return target

    def load_depth(self, depth_path):
        depth = cv2.imread(depth_path, -1)

        if len(depth.shape) == 3:
            depth16 = np.uint16(depth[:, :, 1] * 256) + np.uint16(depth[:, :, 2])
            depth16 = depth16.astype(np.uint16)
        elif len(depth.shape) == 2 and depth.dtype == 'uint16':
            depth16 = depth
        else:
            assert False, '[ Error ]: Unsupported depth type.'

        return depth16

    def get_pose(self, choose_frame, choose_obj):
        has_pose = []
        pose = {}
        if self.mode == "train":
            input_file = open('{0}_pose.txt'.format(choose_frame.replace("data/", "data_pose/")), 'r')
            while 1:
                input_line = input_file.readline()
                if not input_line:
                    break
                if input_line[-1:] == '\n':
                    input_line = input_line[:-1]
                input_line = input_line.split(' ')
                if len(input_line) == 1:
                    idx = int(input_line[0])
                    has_pose.append(idx)
                    pose[idx] = []
                    for i in range(4):
                        input_line = input_file.readline()
                        if input_line[-1:] == '\n':
                            input_line = input_line[:-1]
                        input_line = input_line.split(' ')
                        pose[idx].append(
                            [float(input_line[0]), float(input_line[1]), float(input_line[2]), float(input_line[3])])
            input_file.close()
        if self.mode == "val":
            with open('{0}/data/gts/real_test/results_real_test_{1}_{2}.pkl'.format(self.root,
                                                                                    choose_frame.split("/")[-2],
                                                                                    choose_frame.split("/")[-1]),
                      'rb') as f:
                nocs_data = cPickle.load(f)
            for idx in range(nocs_data['gt_RTs'].shape[0]):
                idx = idx + 1
                pose[idx] = nocs_data['gt_RTs'][idx - 1]
                pose[idx][:3, :3] = pose[idx][:3, :3] / np.cbrt(np.linalg.det(pose[idx][:3, :3]))
                z_180_RT = np.zeros((4, 4), dtype=np.float32)
                z_180_RT[:3, :3] = np.diag([-1, -1, 1])
                z_180_RT[3, 3] = 1
                pose[idx] = z_180_RT @ pose[idx]
                pose[idx][:3, 3] = pose[idx][:3, 3] * 1000

        input_file = open('{0}_meta.txt'.format(choose_frame), 'r')
        while 1:
            input_line = input_file.readline()
            if not input_line:
                break
            if input_line[-1:] == '\n':
                input_line = input_line[:-1]
            input_line = input_line.split(' ')
            if input_line[-1] == choose_obj:
                ans = pose[int(input_line[0])]
                ans_idx = int(input_line[0])
                break
        input_file.close()

        ans = np.array(ans)
        ans_r = ans[:3, :3]
        ans_t = ans[:3, 3].flatten()

        return ans_r, ans_t, ans_idx

    # choose_obj: the code of the object, choose_frame: the samples prefix.
    def get_frame(self, choose_frame, choose_obj, syn_or_real):

        if syn_or_real:
            mesh_bbox = []
            input_file = open('{0}/model_pts/{1}.txt'.format(self.root, choose_obj), 'r')
            for i in range(8):
                input_line = input_file.readline()
                if input_line[-1:] == '\n':
                    input_line = input_line[:-1]
                input_line = input_line.split(' ')
                mesh_bbox.append([float(input_line[0]), float(input_line[1]), float(input_line[2])])
            input_file.close()
            mesh_bbox = np.array(mesh_bbox)

            mesh_pts = []
            input_file = open('{0}/model_pts/{1}.xyz'.format(self.root, choose_obj), 'r')
            for i in range(2800):
                input_line = input_file.readline()
                if input_line[-1:] == '\n':
                    input_line = input_line[:-1]
                input_line = input_line.split(' ')
                mesh_pts.append([float(input_line[0]), float(input_line[1]), float(input_line[2])])
            input_file.close()
        img = Image.open('{0}_color.png'.format(choose_frame))
        target_r, target_t, idx = self.get_pose(choose_frame, choose_obj)

        if syn_or_real:
            cam_cx = self.cam_cx_2
            cam_cy = self.cam_cy_2
            cam_fx = self.cam_fx_2
            cam_fy = self.cam_fy_2
        else:
            cam_cx = self.cam_cx_1
            cam_cy = self.cam_cy_1
            cam_fx = self.cam_fx_1
            cam_fy = self.cam_fy_1
        cam_scale = 1.0

        if syn_or_real:
            target = []
            input_file = open('{0}_bbox.txt'.format(choose_frame.replace("data/", "data_pose/")), 'r')
            while 1:
                input_line = input_file.readline()
                if not input_line:
                    break
                if input_line[-1:] == '\n':
                    input_line = input_line[:-1]
                input_line = input_line.split(' ')
                if len(input_line) == 1 and int(input_line[0]) == idx:
                    for i in range(8):
                        input_line = input_file.readline()
                        if input_line[-1:] == '\n':
                            input_line = input_line[:-1]
                        input_line = input_line.split(' ')
                        target.append([float(input_line[0]), float(input_line[1]), float(input_line[2])])
                    break
            input_file.close()
            target = np.array(target)
        else:
            target = []
            input_file = open('{0}/model_scales/{1}.txt'.format(self.root, choose_obj), 'r')
            for i in range(8):
                input_line = input_file.readline()
                if input_line[-1:] == '\n':
                    input_line = input_line[:-1]
                input_line = input_line.split(' ')
                target.append([float(input_line[0]), float(input_line[1]), float(input_line[2])])
            input_file.close()
            target = np.array(target)

        target = self.enlarge_bbox(copy.deepcopy(target))

        delta = math.pi / 10.0
        noise_trans = 0.05
        target_tmp = target - (np.array([random.uniform(-noise_trans, noise_trans) for i in range(3)]) * 3000.0)
        target_tmp = np.dot(target_tmp, target_r.T) + target_t
        target_tmp[:, 0] *= -1.0
        target_tmp[:, 1] *= -1.0
        rmin, rmax, cmin, cmax = get_2dbbox(target_tmp, cam_cx, cam_cy, cam_fx, cam_fy,
                                            cam_scale)  # These four values is the boundaries of 2d bounding box.
        if self.add_noise:
            img = self.trancolor(img)

            if random.randint(1, 20) > 3:
                back_frame = random.sample(self.back_list, 1)[0]

                back_img = np.array(self.trancolor(Image.open(back_frame).resize((640, 480), Image.ANTIALIAS)))
                back_img = np.transpose(back_img, (2, 0, 1))

                mask = (cv2.imread('{0}_mask.png'.format(choose_frame))[:, :, 0] == 255)
                img = np.transpose(np.array(img), (2, 0, 1))
                # Here use the object from /data/train/.png to be foreground and the background from /train2017 to synethize a new image.
                img = img * (~mask) + back_img * mask

                img = np.transpose(img, (1, 2, 0))
            else:
                img = np.array(img)
        else:
            img = np.array(img)

        mask_target = (cv2.imread('{0}_mask.png'.format(choose_frame))[:, :, 2] == idx)[rmin:rmax, cmin:cmax]
        choose = (mask_target.flatten() != False).nonzero()[0]
        if len(choose) == 0:
            return 0
        # [:, rmin:rmax, cmin:cmax]
        img = np.transpose(img[:, :, :3], (2, 0, 1))[:, rmin:rmax, cmin:cmax]
        img = img / 255.0

        # Deep copy
        img_r = img
        img_r = np.transpose(img_r, (1, 2, 0))
        img_r = resize(img_r, (320, 320, 3))
        img_r = np.transpose(img_r, (2, 0, 1))
        return img_r

    def re_scale(self, target_fr, target_to):
        ans_scale = target_fr / target_to
        ans_target = target_fr
        ans_scale = ans_scale[0][0]

        return ans_target, ans_scale

    def __getitem__(self, index):
        syn_or_real = (random.randint(1, 20) < 15)  # True(syn): 3/4, False(real): 1/4

        if self.mode == 'val':
            syn_or_real = False

        if syn_or_real:
            # Synthetic data 3/4

            choose_obj = random.sample(self.obj_name_list[self.cate_id], 1)[0]  # Select one object.


            while 1:
                try:
                    choose_frame = random.sample(self.obj_list[self.cate_id][choose_obj],2)  # Path like data/train/06652/0003
                    img_fr_r = self.get_frame(choose_frame[0], choose_obj, syn_or_real)
                    img_to_r = self.get_frame(choose_frame[1], choose_obj, syn_or_real)
                    img_fr_r = self.norm(torch.from_numpy(img_fr_r.astype(np.float32)))
                    img_to_r = self.norm(torch.from_numpy(img_to_r.astype(np.float32)))
                    break
                except:

                    continue

        else:
            # Real data from video sequence, 1/4
            choose_obj = random.sample(self.real_obj_name_list[self.cate_id], 1)[0]

            while 1:
                try:

                    choose_frame = random.sample(self.real_obj_list[self.cate_id][choose_obj], 2)

                    img_fr_r = self.get_frame(choose_frame[0], choose_obj,syn_or_real)
                    img_to_r = self.get_frame(choose_frame[1],choose_obj,syn_or_real)

                    img_fr_r = self.norm(torch.from_numpy(img_fr_r.astype(np.float32)))
                    img_to_r = self.norm(torch.from_numpy(img_to_r.astype(np.float32)))

                    break
                except:
                    continue
        class_gt = np.array([self.cate_id - 1])
        return img_fr_r, img_to_r, torch.LongTensor(class_gt.astype(np.int32))

    def __len__(self):

            return self.length


border_list = [-1, 80, 120, 160, 200, 240, 280, 320, 360, 400, 440, 480, 520, 560, 600, 640, 680]
img_width = 480
img_length = 640


def get_2dbbox(cloud, cam_cx, cam_cy, cam_fx, cam_fy, cam_scale):
    rmin = 10000
    rmax = -10000
    cmin = 10000
    cmax = -10000
    for tg in cloud:
        p1 = int(tg[0] * cam_fx / tg[2] + cam_cx)
        p0 = int(tg[1] * cam_fy / tg[2] + cam_cy)
        if p0 < rmin:
            rmin = p0
        if p0 > rmax:
            rmax = p0
        if p1 < cmin:
            cmin = p1
        if p1 > cmax:
            cmax = p1
    rmax += 1
    cmax += 1
    if rmin < 0:
        rmin = 0
    if cmin < 0:
        cmin = 0
    if rmax >= 480:
        rmax = 479
    if cmax >= 640:
        cmax = 639

    r_b = rmax - rmin
    for tt in range(len(border_list)):
        if r_b > border_list[tt] and r_b < border_list[tt + 1]:
            r_b = border_list[tt + 1]
            break
    c_b = cmax - cmin
    for tt in range(len(border_list)):
        if c_b > border_list[tt] and c_b < border_list[tt + 1]:
            c_b = border_list[tt + 1]
            break
    center = [int((rmin + rmax) / 2), int((cmin + cmax) / 2)]
    rmin = center[0] - int(r_b / 2)
    rmax = center[0] + int(r_b / 2)
    cmin = center[1] - int(c_b / 2)
    cmax = center[1] + int(c_b / 2)

    if rmin < 0:
        delt = -rmin
        rmin = 0
        rmax += delt
    if cmin < 0:
        delt = -cmin
        cmin = 0
        cmax += delt
    if rmax > img_width:
        delt = rmax - img_width
        rmax = img_width
        rmin -= delt
    if cmax > img_length:
        delt = cmax - img_length
        cmax = img_length
        cmin -= delt

    if ((rmax - rmin) in border_list) and ((cmax - cmin) in border_list):
        return rmin, rmax, cmin, cmax
    else:
        return 0


def search_fit(points):
    min_x = min(points[:, 0])
    max_x = max(points[:, 0])
    min_y = min(points[:, 1])
    max_y = max(points[:, 1])
    min_z = min(points[:, 2])
    max_z = max(points[:, 2])

    return [min_x, max_x, min_y, max_y, min_z, max_z]
