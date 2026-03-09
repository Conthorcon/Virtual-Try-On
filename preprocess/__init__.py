from preprocess.openpose.run_openpose import OpenPose
from preprocess.humanparsing.run_parsing import Parsing
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import cv2

import random
import argparse
import os.path as osp
from PIL import Image
from PIL import ImageDraw
import numpy as np
import json
import os

from .utils import *

def CPVTON(opt, person, cloth):
    parsing_model = Parsing(0)
    openpose_model = OpenPose(0)

    device = torch.device('cpu')

    openpose_model.preprocessor.body_estimation.model.to(device)

    cloth = cloth.resize((opt.fine_width, opt.fine_height))
    person = person.resize((opt.fine_width, opt.fine_height))

    # POSE ESTIMATION
    transform = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            transforms.Normalize((0.5,), (0.5,))])

    cloth_parsing, _ = parsing_model(cloth)
    cloth_array = np.array(cloth_parsing)

    cloth_array = (cloth_array == 4).astype(np.float32) + \
    (cloth_array == 7).astype(np.float32) + \
    (cloth_array == 17).astype(np.float32) + \
    (cloth_array == 16).astype(np.float32)
    # (cloth_array == 18).astype(np.float32) 

    cm_array = (cloth_array > 0).astype(np.float32)
    cm = torch.from_numpy(cm_array)  # [0,1]
    cm.unsqueeze_(0)


    c = transform(cloth)
    im = transform(person)

    # PARSING IMAGE
    model_parse, _ = parsing_model(person)
    parse_array = np.array(model_parse)
    
    parse_shape = (parse_array > 0).astype(np.float32)


    """
    Labels

    0: Background
    1: Hat
    2: Hair
    3: Headwear
    4: Upper clothes
    5: Skirt
    6: Short
    7: Dress
    8: Socks
    9: Left Shoes
    10: Right Shoes
    11: Face
    12: Left Leg 
    13: Right Leg
    14: Left Arm
    15: Right Arm
    16: Bag
    17: Scarf
    18: Neck

    """

    if opt.stage == 'GMM':
        parse_head = (parse_array == 1).astype(np.float32) + \
            (parse_array == 3).astype(np.float32) + \
            (parse_array == 18).astype(np.float32) + \
            (parse_array == 2).astype(np.float32) + \
            (parse_array == 11).astype(
                np.float32)  # CP-VTON+ GMM input (reserved regions)
    else:
        parse_head = (parse_array == 1).astype(np.float32) + \
            (parse_array == 2).astype(np.float32) + \
            (parse_array == 3).astype(np.float32) + \
            (parse_array == 5).astype(np.float32) + \
            (parse_array == 6).astype(np.float32) + \
            (parse_array == 11).astype(np.float32) + \
            (parse_array == 12).astype(np.float32) + \
            (parse_array == 13).astype(
            np.float32)  # CP-VTON+ TOM input (reserved regions)

    parse_cloth = (parse_array == 4).astype(np.float32) + \
        (parse_array == 17).astype(np.float32) + \
        (parse_array == 16).astype(np.float32)    # upper-clothes labels


    # plt.imshow(parse_array)
    # plt.colorbar()
    # plt.show()

    # labels, counts = np.unique(parse_array, return_counts=True)
    # print(dict(zip(labels, counts)))

    # shape downsample
    parse_shape_ori = Image.fromarray((parse_shape*255).astype(np.uint8))
    parse_shape = parse_shape_ori.resize(
        (opt.fine_width//16, opt.fine_height//16), Image.BILINEAR)
    parse_shape = parse_shape.resize(
        (opt.fine_width, opt.fine_height), Image.BILINEAR)
    parse_shape_ori = parse_shape_ori.resize(
        (opt.fine_width, opt.fine_height), Image.BILINEAR)
    shape_ori = transform(parse_shape_ori)  # [-1,1]
    shape = transform(parse_shape)  # [-1,1]
    phead = torch.from_numpy(parse_head)  # [0,1]
    # phand = torch.from_numpy(parse_hand)  # [0,1]
    pcm = torch.from_numpy(parse_cloth)  # [0,1]

    # upper cloth
    im_c = im * pcm + (1 - pcm)  # [-1,1], fill 1 for other parts
    im_h = im * phead - (1 - phead)  # [-1,1], fill -1 for other parts

    
    # img = im_h.detach().cpu()

    # # CHW → HWC
    # img = img.permute(1, 2, 0).numpy()
    
    # plt.imshow(img)
    # plt.show()

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    pose_data = openpose_model(person)
    pose_data = pose_data["pose_keypoints_2d"]
    pose_data = pose_data[:18]
    pose_data = np.array(pose_data)
    pose_data = pose_data.reshape((-1, 2))

    # data_path = osp.join(opt.dataroot, opt.datamode)
    # with open(osp.join(data_path, 'pose', "000001_0_keypoints.json"), 'r') as f:
    #     pose_label = json.load(f)
    #     pose_data = pose_label['people'][0]['pose_keypoints']
    #     pose_data = np.array(pose_data)
    #     pose_data = pose_data.reshape((-1, 3))

    point_num = pose_data.shape[0]
    pose_map = torch.zeros(point_num, opt.fine_height, opt.fine_width)
    r = opt.radius
    im_pose = Image.new('L', (opt.fine_width, opt.fine_height))
    pose_draw = ImageDraw.Draw(im_pose)
    for i in range(point_num):
        one_map = Image.new('L', (opt.fine_width, opt   .fine_height))
        draw = ImageDraw.Draw(one_map)
        pointx = pose_data[i, 0]
        pointy = pose_data[i, 1]
        if pointx > 1 and pointy > 1:
            draw.rectangle((pointx-r, pointy-r, pointx +
                            r, pointy+r), 'white', 'white')
            pose_draw.rectangle(
                (pointx-r, pointy-r, pointx+r, pointy+r), 'white', 'white')
        one_map = transform(one_map)
        pose_map[i] = one_map[0]

        # just for visualization
        # im_pose = transform(im_pose)

    agnostic = torch.cat([shape, im_h, pose_map], 0)

    if opt.stage == 'GMM':
        im_g = Image.open('grid.png')
        im_g = transform(im_g)
    else:
        im_g = ''


    # img_vis = np.array(person.copy())

    # for (x, y) in pose_data:
    #     x, y = int(x), int(y)
    #     cv2.circle(img_vis, (x, y), radius=4, color=(0, 255, 0), thickness=-1)

    # cv2.imshow("pose", img_vis)
    # cv2.waitKey(0)


    result = {
            'im_name': ['test_image.jpg'],  # dummy name
            'cloth':    c,          # for input
            'cloth_mask':     cm,   # for input
            'image':    im,         # for visualization
            'agnostic': agnostic,   # for input
            'parse_cloth': im_c,    # for ground truth
            'shape': shape,         # for visualization
            'head': im_h,           # for visualization
            'pose_image': im_pose,  # for visualization
            'grid_image': im_g,     # for visualization
            'parse_cloth_mask': pcm,     # for CP-VTON+, TOM input
            'shape_ori': shape_ori,     # original body shape without resize
        }

    return result

def MVTON0(opt, person0, cloth, warped_inputs):
    parsing_model = Parsing(0)
    openpose_model = OpenPose(0)

    device = torch.device('cpu')

    openpose_model.preprocessor.body_estimation.model.to(device)

    crop_size = (opt.fine_height, opt.fine_width)
    toTensor = transforms.ToTensor()
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    to_pil = transforms.ToPILImage()
    clip_normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                                   (0.26862954, 0.26130258, 0.27577711))

    im_pil = transforms.Resize(crop_size, interpolation=2)(person0)
    im = transform(im_pil)

    # 1. Parsing image
    # load parsing image
    parse_name = 'ip.png'
    im_parse_pil_big = Image.open(osp.join('test_data/t', parse_name))
    im_parse_pil = transforms.Resize(crop_size, interpolation=0)(im_parse_pil_big)
    parse = torch.from_numpy(np.array(im_parse_pil)[None]).long()

    # person_parse, _ = parsing_model(person0)
    # im_parse_pil = transforms.Resize(crop_size, interpolation=0)(person_parse)
    # parse = torch.from_numpy(np.array(im_parse_pil)[None]).long()


    # Parse map one-hot
    parse_map = torch.FloatTensor(20, opt.fine_height, opt.fine_width).zero_()
    parse_map = parse_map.scatter_(0, parse, 1.0)
    new_parse_map = torch.FloatTensor(13, opt.fine_height, opt.fine_width).zero_()

    # parse map
    # labels = {
    #     0: ['background', [0]],
    #     1: ['hair', [1, 2, 3]],
    #     2: ['face', [11, 18]],
    #     3: ['upper', [4, 7, 16, 17]],
    #     4: ['bottom', [5, 6]],
    #     5: ['left_arm', [14]],
    #     6: ['right_arm', [15]],
    #     7: ['left_leg', [12]],
    #     8: ['right_leg', [13]],
    #     9: ['left_shoe', [9]],
    #     10: ['right_shoe', [10]],
    #     11: ['socks', [8]],
    #     12: ['noise', [19]]
    # }
    labels = {
            0: ['background', [0, 10]],
            1: ['hair', [1, 2]],
            2: ['face', [4, 13]],
            3: ['upper', [5, 6, 7]],
            4: ['bottom', [9, 12]],
            5: ['left_arm', [14]],
            6: ['right_arm', [15]],
            7: ['left_leg', [16]],
            8: ['right_leg', [17]],
            9: ['left_shoe', [18]],
            10: ['right_shoe', [19]],
            11: ['socks', [8]],
            12: ['noise', [3, 11]]
        }


    for i in range(len(labels)):
        for label in labels[i][1]:
            new_parse_map[i] += parse_map[label]

    parse_onehot = torch.FloatTensor(1, opt.fine_height, opt.fine_width).zero_()
    for i in range(len(labels)):
        for label in labels[i][1]:
            parse_onehot[0] += parse_map[label] * i

    mask_id = torch.Tensor([3, 5, 6])
    mask = torch.isin(parse_onehot[0], mask_id).numpy()

    kernel_size = int(5 * (opt.fine_width / 256))
    mask = cv2.dilate(mask.astype(np.uint8), kernel=np.ones((kernel_size, kernel_size)), iterations=3)
    mask = cv2.erode(mask.astype(np.uint8), kernel=np.ones((kernel_size, kernel_size)), iterations=1)
    mask = mask.astype(np.float32)
    inpaint_mask = 1 - toTensor(mask)

    # 2. Warped cloth
    warped_cloth = Image.open('test_data/t/cw.jpg').convert("RGB")

    # warped_cloth = warped_inputs['cloth']

    # warped_cloth = to_pil(warped_cloth[0])
    warped_cloth = transforms.Resize(crop_size, interpolation=2)(warped_cloth)
    warped_cloth = transform(warped_cloth)

    # warped_cloth_mask = warped_inputs['cloth_mask'] 

    # 3. Warped cloth mask
    warped_cloth_mask = Image.open('test_data/t/cwm.jpg')
    # warped_cloth_mask = to_pil(warped_cloth_mask[0])
    warped_cloth_mask = transforms.Resize(crop_size, interpolation=transforms.InterpolationMode.NEAREST) \
            (warped_cloth_mask)
    warped_cloth_mask = toTensor(warped_cloth_mask)
    warped_cloth = warped_cloth * warped_cloth_mask


    feat = warped_cloth * (1 - inpaint_mask) + im * inpaint_mask


    c = transforms.Resize(crop_size, interpolation=2)(cloth)
    c_img = c
    controlnet_cond = transform(c_img)

    # 4. Cloth mask
    cm = Image.open(osp.join('test_data/t', 'cm.jpg'))
    cm = transforms.Resize(crop_size, interpolation=0)(cm)
    cm_img = cm

    c = transform(c)  # [-1,1]
    cm_array = np.array(cm)
    cm_array = (cm_array >= 128).astype(np.float32)
    cm = torch.from_numpy(cm_array)  # [0,1]
    cm.unsqueeze_(0)


    # cloth_parsing, _ = parsing_model(cloth)
    # cloth_array = np.array(cloth_parsing)   # shape (H, W)

    # mask = (
    #     (cloth_array == 4) |
    #     (cloth_array == 7) |
    #     (cloth_array == 16) |
    #     (cloth_array == 17)
    # ).astype(np.float32)   # (H, W), giá trị 0 hoặc 1

    # cm = torch.from_numpy(mask).unsqueeze_(0)   # shape (1, H, W)
    # cm = transforms.Resize(
    #     crop_size,
    #     interpolation=transforms.InterpolationMode.NEAREST
    # )(cm)


    cm_img = cm.clone()

    c = transform(cloth)   # [-1, 1]

    down, up, left, right = mask2bbox(cm[0].numpy())
    ref_image = c[:, down:up, left:right]
    ref_image = (ref_image + 1.0) / 2.0
    ref_image = transforms.Resize((224, 224))(ref_image)
    ref_image = clip_normalize(ref_image)


    # 5. Agnostic representation
    # from preprocess.openpose.run_openpose import draw_pose
    # load pose points
    pose_name = 'test_data/t/k.json'
    with open(pose_name, 'r') as f:
        pose_label = json.load(f)
        pose_data = pose_label['people'][0]['pose_keypoints_2d']
        pose_data = np.array(pose_data)
        pose_data = pose_data.reshape((-1, 3))[:, :2]

    agnostic = get_agnostic0(person0, im_parse_pil_big, pose_data)
    agnostic = transforms.Resize(crop_size, interpolation=2)(agnostic)
    agnostic = transform(agnostic)


    # pose_data = openpose_model(person0)
    # pose_data = pose_data["pose_keypoints_2d"]
    # pose_data = np.array(pose_data)
    # pose_data = pose_data.reshape((-1, 2))[:, :2]


    # agnostic = get_agnostic(person0, person_parse, pose_data)
    # agnostic = transforms.Resize(crop_size, interpolation=2)(agnostic)
    # agnostic = transform(agnostic)

    # load image-parse-agnostic
    # 6. Parsing Agnostic map
    parse = Image.open(osp.join('test_data/t','ipa.png'))

    # parse = np.array(person_parse)

    # parse[np.isin(parse, [4,7,16,17,18])] = 0

    # parse = Image.fromarray(parse.astype(np.uint8))
    parse = transforms.Resize(crop_size, interpolation=Image.NEAREST)(parse)

    parse = torch.from_numpy(np.array(parse)).long().unsqueeze(0)

    parse_map = torch.zeros(20, opt.fine_height, opt.fine_width)
    parse_map.scatter_(0, parse, 1.0)

    new_parse_agnostic_map = torch.FloatTensor(13, opt.fine_height, opt.fine_width).zero_()
    
    for i in range(len(labels)):
        for label in labels[i][1]:
            new_parse_agnostic_map[i] += parse_map[label]
    hands_mask = torch.sum(new_parse_agnostic_map[5:7], dim=0, keepdim=True)
    hands_mask = torch.clamp(hands_mask, min=0.0, max=1.0)

    inpaint = feat * (1 - hands_mask) + agnostic * hands_mask

    output_path = opt.output
    os.makedirs(output_path,exist_ok=True)

    show_tensor_image(im, save_path=osp.join(output_path,"im.jpg"))
    show_tensor_image(feat, save_path=osp.join(output_path,"feat.jpg"))
    show_tensor_image(hands_mask, save_path=osp.join(output_path,"hands_mask.jpg"))
    show_tensor_image(agnostic, save_path=osp.join(output_path,"agnostic.jpg"))
    show_tensor_image(inpaint, save_path=osp.join(output_path,"inpaint.jpg"))
    show_tensor_image(inpaint_mask, save_path=osp.join(output_path,"inpaint_mask.jpg"))
    show_tensor_image(warped_cloth_mask, save_path=osp.join(output_path,"warped_cloth_mask.jpg"))
    show_tensor_image(warped_cloth, save_path=osp.join(output_path,"warped_cloth.jpg"))



    result = {
            "GT": im,
            "inpaint_image": inpaint,
            "inpaint_mask": inpaint_mask,
            "ref_imgs": ref_image,
            'warp_feat': feat,
            # "file_name": self.im_names[index],
            "controlnet_cond": controlnet_cond,
            "hands_mask": hands_mask,
        }
    
    return result
    

def MVTON1(opt, person0, cloth, inputs):
    parsing_model = Parsing(0)
    openpose_model = OpenPose(0)

    device = torch.device('cpu')

    openpose_model.preprocessor.body_estimation.model.to(device)

    crop_size = (opt.fine_height, opt.fine_width)
    toTensor = transforms.ToTensor()
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    to_pil = transforms.ToPILImage()
    clip_normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                                   (0.26862954, 0.26130258, 0.27577711))

    im_pil = transforms.Resize(crop_size, interpolation=2)(person0)
    im = transform(im_pil)

    # 1. Parsing image
    # load parsing image
    # parse_name = 'ip.png'
    # im_parse_pil_big = Image.open(osp.join('test_data/t', parse_name))
    # im_parse_pil = transforms.Resize(crop_size, interpolation=0)(im_parse_pil_big)
    # parse = torch.from_numpy(np.array(im_parse_pil)[None]).long()

    person_parse, _ = parsing_model(person0)
    im_parse_pil = transforms.Resize(crop_size, interpolation=0)(person_parse)
    parse = torch.from_numpy(np.array(im_parse_pil)[None]).long()


    # Parse map one-hot
    parse_map = torch.FloatTensor(20, opt.fine_height, opt.fine_width).zero_()
    parse_map = parse_map.scatter_(0, parse, 1.0)
    new_parse_map = torch.FloatTensor(13, opt.fine_height, opt.fine_width).zero_()

    # parse map
    labels = {
        0: ['background', [0]],
        1: ['hair', [1, 2, 3]],
        2: ['face', [11, 18]],
        3: ['upper', [4, 7, 16, 17]],
        4: ['bottom', [5, 6]],
        5: ['left_arm', [14]],
        6: ['right_arm', [15]],
        7: ['left_leg', [12]],
        8: ['right_leg', [13]],
        9: ['left_shoe', [9]],
        10: ['right_shoe', [10]],
        11: ['socks', [8]],
        12: ['noise', [19]]
    }
    # labels = {
    #         0: ['background', [0, 10]],
    #         1: ['hair', [1, 2]],
    #         2: ['face', [4, 13]],
    #         3: ['upper', [5, 6, 7]],
    #         4: ['bottom', [9, 12]],
    #         5: ['left_arm', [14]],
    #         6: ['right_arm', [15]],
    #         7: ['left_leg', [16]],
    #         8: ['right_leg', [17]],
    #         9: ['left_shoe', [18]],
    #         10: ['right_shoe', [19]],
    #         11: ['socks', [8]],
    #         12: ['noise', [3, 11]]
    #     }


    for i in range(len(labels)):
        for label in labels[i][1]:
            new_parse_map[i] += parse_map[label]

    parse_onehot = torch.FloatTensor(1, opt.fine_height, opt.fine_width).zero_()
    for i in range(len(labels)):
        for label in labels[i][1]:
            parse_onehot[0] += parse_map[label] * i

    mask_id = torch.Tensor([3, 5, 6])
    mask = torch.isin(parse_onehot[0], mask_id).numpy()

    kernel_size = int(5 * (opt.fine_width / 256))
    mask = cv2.dilate(mask.astype(np.uint8), kernel=np.ones((kernel_size, kernel_size)), iterations=3)
    mask = cv2.erode(mask.astype(np.uint8), kernel=np.ones((kernel_size, kernel_size)), iterations=1)
    mask = mask.astype(np.float32)
    inpaint_mask = 1 - toTensor(mask)

    # 2. Warped cloth
    # warped_cloth = Image.open('test_data/t/cw.jpg').convert("RGB")
    warped_cloth = inputs['warped_cloth']

    # warped_cloth = to_pil(warped_cloth[0])
    # warped_cloth = transforms.Resize(crop_size, interpolation=2)(warped_cloth)
    # warped_cloth = transform(warped_cloth)

    warped_cloth = F.interpolate(
        warped_cloth,
        size=crop_size,
        mode="bilinear",
        align_corners=False
    )[0]

    # 3. Warped cloth mask
    # warped_cloth_mask = Image.open('test_data/t/cwm.jpg')
    warped_cloth_mask = inputs['warped_cloth_mask'] 
    warped_cloth_mask = to_pil(warped_cloth_mask[0])
    warped_cloth_mask = transforms.Resize(crop_size, interpolation=transforms.InterpolationMode.NEAREST) \
            (warped_cloth_mask)
    warped_cloth_mask = toTensor(warped_cloth_mask)
    warped_cloth = warped_cloth * warped_cloth_mask


    feat = warped_cloth * (1 - inpaint_mask) + im * inpaint_mask

    c = transforms.Resize(crop_size, interpolation=2)(cloth)
    c_img = c
    controlnet_cond = transform(c_img)

    # 4. Cloth mask
    cm = tensor_to_pil_mask(inputs["cloth_mask"]) 
    cm = transforms.Resize(crop_size, interpolation=0)(cm)
    cm_img = cm

    c = transform(c)  # [-1,1]
    cm_array = np.array(cm)
    cm_array = (cm_array >= 128).astype(np.float32)
    cm = torch.from_numpy(cm_array)  # [0,1]
    cm.unsqueeze_(0)


    # cloth_parsing, _ = parsing_model(cloth)
    # cloth_array = np.array(cloth_parsing)   # shape (H, W)

    # mask = (
    #     (cloth_array == 4) |
    #     (cloth_array == 7) |
    #     (cloth_array == 16) |
    #     (cloth_array == 17)
    # ).astype(np.float32)   # (H, W), giá trị 0 hoặc 1

    # cm = torch.from_numpy(mask).unsqueeze_(0)   # shape (1, H, W)
    # cm = transforms.Resize(
    #     crop_size,
    #     interpolation=transforms.InterpolationMode.NEAREST
    # )(cm)


    cm_img = cm.clone()

    c = transform(cloth)   # [-1, 1]

    down, up, left, right = mask2bbox(cm[0].numpy())
    ref_image = c[:, down:up, left:right]
    ref_image = (ref_image + 1.0) / 2.0
    ref_image = transforms.Resize((224, 224))(ref_image)
    ref_image = clip_normalize(ref_image)


    # 5. Agnostic representation
    # from preprocess.openpose.run_openpose import draw_pose
    # load pose points
    # pose_name = 'test_data/t/k.json'
    # with open(pose_name, 'r') as f:
    #     pose_label = json.load(f)
    #     pose_data = pose_label['people'][0]['pose_keypoints_2d']
    #     pose_data = np.array(pose_data)
    #     pose_data = pose_data.reshape((-1, 3))[:, :2]

    # agnostic = get_agnostic0(person0, im_parse_pil_big, pose_data)

    # agnostic = transforms.Resize(crop_size, interpolation=2)(agnostic)
    # agnostic = transform(agnostic)


    pose_data = openpose_model(person0)
    pose_data = pose_data["pose_keypoints_2d"]
    pose_data = np.array(pose_data)
    pose_data = pose_data.reshape((-1, 2))[:, :2]


    agnostic = get_agnostic1(person0, person_parse, pose_data)
    agnostic = transforms.Resize(crop_size, interpolation=2)(agnostic)
    agnostic = transform(agnostic)


    # load image-parse-agnostic
    # 6. Parsing Agnostic map
    # parse = Image.open(osp.join('test_data/t','ipa.png'))

    parse = np.array(person_parse)

    parse[np.isin(parse, [4,7,16,17,18])] = 0

    parse = Image.fromarray(parse.astype(np.uint8))
    parse = transforms.Resize(crop_size, interpolation=Image.NEAREST)(parse)

    parse = torch.from_numpy(np.array(parse)).long().unsqueeze(0)

    parse_map = torch.zeros(20, opt.fine_height, opt.fine_width)
    parse_map.scatter_(0, parse, 1.0)

    new_parse_agnostic_map = torch.FloatTensor(13, opt.fine_height, opt.fine_width).zero_()

    for i in range(len(labels)):
        for label in labels[i][1]:
            new_parse_agnostic_map[i] += parse_map[label]
    hands_mask = torch.sum(new_parse_agnostic_map[5:7], dim=0, keepdim=True)
    hands_mask = torch.clamp(hands_mask, min=0.0, max=1.0)
    hands_mask = hands_mask * (1 - warped_cloth_mask)

    inpaint = feat * (1 - hands_mask) + agnostic * hands_mask
    
    # output_path = opt.output
    # os.makedirs(output_path, exist_ok=True)
    
    # show_pil(person_parse, save_path=osp.join(output_path,"person_parser.jpg"))
    # show_tensor_image(im, save_path=osp.join(output_path,"im.jpg"))
    # show_tensor_image(feat, save_path=osp.join(output_path,"feat.jpg"))
    # show_tensor_image(hands_mask, save_path=osp.join(output_path,"hands_mask.jpg"))
    # show_tensor_image(agnostic, save_path=osp.join(output_path,"agnostic.jpg"))
    # show_tensor_image(inpaint, save_path=osp.join(output_path,"inpaint.jpg"))
    # show_tensor_image(inpaint_mask, save_path=osp.join(output_path,"inpaint_mask.jpg"))
    # show_tensor_image(warped_cloth_mask, save_path=osp.join(output_path,"warped_cloth_mask.jpg"))
    # show_tensor_image(warped_cloth, save_path=osp.join(output_path,"warped_cloth.jpg"))


    result = {
            "GT": im,
            "inpaint_image": inpaint,
            "inpaint_mask": inpaint_mask,
            "ref_imgs": ref_image,
            'warp_feat': feat,
            # "file_name": self.im_names[index],
            "controlnet_cond": controlnet_cond,
            "hands_mask": hands_mask,
        }
    
    return result
from model import build_model
from engine import test_gmm, test_tom
from model.MVTON.utils import get_args

def test():
    # opt = get_opt()

    # cloth = Image.open("test_data/000048_1.jpg").convert("RGB")
    # person = Image.open("test_data/human.jpg").convert("RGB")

    # result = MVTON(opt, person, cloth)

    opt = get_args()

    cloth = Image.open("test_data/c.jpg").convert("RGB")
    person = Image.open("test_data/p.jpg").convert("RGB")

    opt.stage = 'GMM'
    inputs = CPVTON(opt, person, cloth)

    gmm, tom = build_model(opt)
    gmm.eval()

    inputs = test_gmm(opt, inputs, gmm)

    MVTON(opt, person, cloth, inputs)

    

# if __name__ == "__main__":
#     main()