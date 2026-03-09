import pdb

# import config
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).absolute().parents[0].absolute()
sys.path.insert(0, str(PROJECT_ROOT))
import os

import cv2
import einops
import numpy as np
import random
import time
import json

# from pytorch_lightning import seed_everything
from preprocess.openpose.annotator.util import resize_image, HWC3
from preprocess.openpose.annotator.openpose import OpenposeDetector

import argparse
from PIL import Image
import torch
import pdb

# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

import cv2

def draw_pose(image, keypoints):
    img = image.copy()

    # Vẽ điểm
    for x, y in keypoints:
        if x > 0 and y > 0:
            cv2.circle(img, (int(x), int(y)), 4, (0, 255, 0), -1)

    # Vẽ xương
    skeleton = [
        (0, 1), (1, 2), (2, 3), (3, 4),
        (1, 5), (5, 6), (6, 7),
        (1, 8), (8, 9), (9, 10),
        (1, 11), (11, 12), (12, 13),
        (0, 14), (14, 16),
        (0, 15), (15, 17)
    ]

    for i, j in skeleton:
        if keypoints[i][0] > 0 and keypoints[j][0] > 0:
            pt1 = tuple(map(int, keypoints[i]))
            pt2 = tuple(map(int, keypoints[j]))
            cv2.line(img, pt1, pt2, (255, 0, 0), 2)

    return img


class OpenPose:
    def __init__(self, gpu_id: int):
        # self.gpu_id = gpu_id
        # torch.cuda.set_device(gpu_id)
        self.preprocessor = OpenposeDetector()

    # def __call__(self, input_image, resolution=384):
    #     # torch.cuda.set_device(self.gpu_id)
    #     if isinstance(input_image, Image.Image):
    #         input_image = np.asarray(input_image)
    #     elif type(input_image) == str:
    #         input_image = np.asarray(Image.open(input_image))
    #     else:
    #         raise ValueError
    #     with torch.no_grad():
    #         input_image = HWC3(input_image)
    #         # input_image = resize_image(input_image, resolution)
    #         # H, W, C = input_image.shape
    #         # assert (H == 512 and W == 384), 'Incorrect input image shape'
    #         pose, detected_map = self.preprocessor(input_image, hand_and_face=False)

    #         candidate = pose['bodies']['candidate']
    #         subset = pose['bodies']['subset'][0][:18]
    #         for i in range(18):
    #             if subset[i] == -1:
    #                 candidate.insert(i, [0, 0])
    #                 for j in range(i, 18):
    #                     if(subset[j]) != -1:
    #                         subset[j] += 1
    #             elif subset[i] != i:
    #                 candidate.pop(i)
    #                 for j in range(i, 18):
    #                     if(subset[j]) != -1:
    #                         subset[j] -= 1

    #         candidate = candidate[:18]


    #         for i in range(len(candidate)):
    #             candidate[i][0] *= input_image.shape[1]
    #             candidate[i][1] *= input_image.shape[0]

    #         print(len(candidate))
    #         exit(0)

    #         keypoints = {"pose_keypoints_2d": candidate}

        
    #         return keypoints

            # out = draw_pose(input_image, keypoints["pose_keypoints_2d"])
            # cv2.imwrite("pose_result.jpg", out)
            # with open("/home/aigc/ProjectVTON/OpenPose/keypoints/keypoints.json", "w") as f:
            #     json.dump(keypoints, f)
            #
            # # print(candidate)
            # output_image = cv2.resize(cv2.cvtColor(detected_map, cv2.COLOR_BGR2RGB), (768, 1024))
            # cv2.imwrite('out_pose.jpg', output_image)
    def __call__(self, input_image):

        if isinstance(input_image, Image.Image):
            input_image = np.asarray(input_image)
        elif isinstance(input_image, str):
            input_image = np.asarray(Image.open(input_image))
        else:
            raise ValueError

        with torch.no_grad():
            input_image = HWC3(input_image)

            pose, _ = self.preprocessor(input_image, hand_and_face=False)

            candidate = pose['bodies']['candidate']
            subset = pose['bodies']['subset']

            if len(subset) == 0:
                return {"pose_keypoints_2d": [[0,0]] * 20}

            subset = subset[0]  # người đầu tiên

            H, W = input_image.shape[:2]
            keypoints = []

            for idx in subset:

                idx = int(idx)

                if idx == -1 or idx >= len(candidate):
                    keypoints.append([0.0, 0.0])
                else:
                    x = float(candidate[idx][0] * W)
                    y = float(candidate[idx][1] * H)
                    keypoints.append([x, y])


            # detected_map = draw_pose(input_image, keypoints)
            # output_image = cv2.resize(cv2.cvtColor(detected_map, cv2.COLOR_BGR2RGB), (768, 1024))
            # cv2.imwrite('out_pose.jpg', output_image)

            return {"pose_keypoints_2d": keypoints}


if __name__ == '__main__':

    model = OpenPose()
    model('./images/bad_model.jpg')
