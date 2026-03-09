import torch

import matplotlib.pyplot as plt

import random
import argparse
from PIL import Image
from PIL import ImageDraw
import numpy as np


def show_tensor_image(tensor, save_path=None, denorm=True):
    """
    tensor: (C,H,W) hoặc (1,C,H,W)
    save_path: nếu có -> lưu ảnh ra file
    denorm: nếu True -> tự động convert [-1,1] về [0,1]
    """

    # Nếu có batch dimension
    if tensor.dim() == 4:
        tensor = tensor[0]

    # Đưa về CPU
    tensor = tensor.detach().cpu()

    # Nếu đang ở [-1,1] (thường gặp trong VTON)
    if denorm:
        tensor = (tensor + 1) / 2

    # Clamp về [0,1] để tránh clipping warning
    tensor = torch.clamp(tensor, 0, 1)

    # (C,H,W) -> (H,W,C)
    tensor = tensor.permute(1, 2, 0)

    plt.imshow(tensor)
    plt.axis("off")

    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        print(f"Saved to {save_path}")
    else:
        plt.show()

    plt.close()

def show_pil(img, save_path=None):
    import matplotlib.pyplot as plt
    
    plt.imshow(img)
    plt.axis("off")
    
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
        print(f"Saved to {save_path}")
    else:
        plt.show()
        
    plt.close()

def mask2bbox(mask):
    up = np.max(np.where(mask)[0])
    down = np.min(np.where(mask)[0])
    left = np.min(np.where(mask)[1])
    right = np.max(np.where(mask)[1])
    center = ((up + down) // 2, (left + right) // 2)

    factor = random.random() * 0.1 + 0.1

    up = int(min(up * (1 + factor) - center[0] * factor + 1, mask.shape[0]))
    down = int(max(down * (1 + factor) - center[0] * factor, 0))
    left = int(max(left * (1 + factor) - center[1] * factor, 0))
    right = int(min(right * (1 + factor) - center[1] * factor + 1, mask.shape[1]))
    return (down, up, left, right)

def tensor_to_pil_mask(t):
    if t.ndim == 4:      # [1, 1, H, W]
        t = t[0]

    if t.ndim == 3:      # [1, H, W]
        t = t.squeeze(0)

    if t.min() < 0:      # [-1,1] → [0,1]
        t = (t + 1) / 2

    t = t.clamp(0, 1)
    t = (t * 255).byte()

    t = t.cpu().numpy()     # (H, W)
    return Image.fromarray(t, mode='L')

def tensor_to_pil(t):
    if t.ndim == 4:      # [1, C, H, W]
        t = t[0]

    if t.min() < 0:     # [-1,1] → [0,1]
        t = (t + 1) / 2

    t = t.clamp(0, 1)
    t = (t * 255).byte()

    t = t.permute(1, 2, 0).cpu().numpy()  # HWC
    return Image.fromarray(t)


from types import SimpleNamespace# for test
def get_opt():
    parser = SimpleNamespace()

    

    return parser

    
def get_agnostic0(im, im_parse, pose_data):
    parse_array = np.array(im_parse)
    parse_head = ((parse_array == 4).astype(np.float32) +
                    (parse_array == 13).astype(np.float32))
    parse_lower = ((parse_array == 9).astype(np.float32) +
                    (parse_array == 12).astype(np.float32) +
                    (parse_array == 16).astype(np.float32) +
                    (parse_array == 17).astype(np.float32) +
                    (parse_array == 18).astype(np.float32) +
                    (parse_array == 19).astype(np.float32))

    agnostic = im.copy()
    agnostic_draw = ImageDraw.Draw(agnostic)

    length_a = np.linalg.norm(pose_data[5] - pose_data[2])
    length_b = np.linalg.norm(pose_data[12] - pose_data[9])
    point = (pose_data[9] + pose_data[12]) / 2
    pose_data[9] = point + (pose_data[9] - point) / length_b * length_a
    pose_data[12] = point + (pose_data[12] - point) / length_b * length_a

    r = int(length_a / 16) + 1

    # mask torso
    for i in [9, 12]:
        pointx, pointy = pose_data[i]
        agnostic_draw.ellipse((pointx - r * 3, pointy - r * 6, pointx + r * 3, pointy + r * 6), 'gray', 'gray')
    agnostic_draw.line([tuple(pose_data[i]) for i in [2, 9]], 'gray', width=r * 6)
    agnostic_draw.line([tuple(pose_data[i]) for i in [5, 12]], 'gray', width=r * 6)
    agnostic_draw.line([tuple(pose_data[i]) for i in [9, 12]], 'gray', width=r * 12)
    agnostic_draw.polygon([tuple(pose_data[i]) for i in [2, 5, 12, 9]], 'gray', 'gray')

    # mask neck
    pointx, pointy = pose_data[1]
    agnostic_draw.rectangle((pointx - r * 5, pointy - r * 9, pointx + r * 5, pointy), 'gray', 'gray')

    # mask arms
    agnostic_draw.line([tuple(pose_data[i]) for i in [2, 5]], 'gray', width=r * 12)
    for i in [2, 5]:
        pointx, pointy = pose_data[i]
        agnostic_draw.ellipse((pointx - r * 5, pointy - r * 6, pointx + r * 5, pointy + r * 6), 'gray', 'gray')
    for i in [3, 4, 6, 7]:
        if (pose_data[i - 1, 0] == 0.0 and pose_data[i - 1, 1] == 0.0) or (
                pose_data[i, 0] == 0.0 and pose_data[i, 1] == 0.0):
            continue
        agnostic_draw.line([tuple(pose_data[j]) for j in [i - 1, i]], 'gray', width=r * 10)
        pointx, pointy = pose_data[i]
        agnostic_draw.ellipse((pointx - r * 5, pointy - r * 5, pointx + r * 5, pointy + r * 5), 'gray', 'gray')

    for parse_id, pose_ids in [(14, [5, 6, 7]), (15, [2, 3, 4])]:
        # mask_arm = Image.new('L', (self.fine_width, self.fine_height), 'white')
        mask_arm = Image.new('L', (768, 1024), 'white')
        mask_arm_draw = ImageDraw.Draw(mask_arm)
        pointx, pointy = pose_data[pose_ids[0]]
        mask_arm_draw.ellipse((pointx - r * 5, pointy - r * 6, pointx + r * 5, pointy + r * 6), 'black', 'black')
        for i in pose_ids[1:]:
            if (pose_data[i - 1, 0] == 0.0 and pose_data[i - 1, 1] == 0.0) or (
                    pose_data[i, 0] == 0.0 and pose_data[i, 1] == 0.0):
                continue
            mask_arm_draw.line([tuple(pose_data[j]) for j in [i - 1, i]], 'black', width=r * 10)
            pointx, pointy = pose_data[i]
            if i != pose_ids[-1]:
                mask_arm_draw.ellipse((pointx - r * 5, pointy - r * 5, pointx + r * 5, pointy + r * 5), 'black',
                                        'black')
        mask_arm_draw.ellipse((pointx - r * 4, pointy - r * 4, pointx + r * 4, pointy + r * 4), 'black', 'black')

        parse_arm = (np.array(mask_arm) / 255) * (parse_array == parse_id).astype(np.float32)
        agnostic.paste(im, None, Image.fromarray(np.uint8(parse_arm * 255), 'L'))

    agnostic.paste(im, None, Image.fromarray(np.uint8(parse_head * 255), 'L'))
    agnostic.paste(im, None, Image.fromarray(np.uint8(parse_lower * 255), 'L'))
    return agnostic


def get_agnostic1(im, im_parse, pose_data):
        parse_array = np.array(im_parse)
        parse_head = ((parse_array == 1).astype(np.float32) +
                    # (parse_array == 2).astype(np.float32) +
                    (parse_array == 3).astype(np.float32) +
                    (parse_array == 11).astype(np.float32))
        parse_lower = ((parse_array == 5).astype(np.float32) +
                       (parse_array == 6).astype(np.float32) +
                       (parse_array == 7).astype(np.float32) +
                       (parse_array == 8).astype(np.float32) +
                       (parse_array == 12).astype(np.float32) +
                       (parse_array == 13).astype(np.float32))

        agnostic = im.copy()
        agnostic_draw = ImageDraw.Draw(agnostic)

        length_a = np.linalg.norm(pose_data[5] - pose_data[2])
        length_b = np.linalg.norm(pose_data[11] - pose_data[8])
        point = (pose_data[8] + pose_data[11]) / 2
        pose_data[8] = point + (pose_data[8] - point) / length_b * length_a
        pose_data[11] = point + (pose_data[11] - point) / length_b * length_a

        r = int(length_a / 16) + 1

        # mask torso
        for i in [8, 11]:
            pointx, pointy = pose_data[i]
            agnostic_draw.ellipse((pointx - r * 3, pointy - r * 6, pointx + r * 3, pointy + r * 6), 'gray', 'gray')
        agnostic_draw.line([tuple(pose_data[i]) for i in [2, 8]], 'gray', width=r * 6)
        agnostic_draw.line([tuple(pose_data[i]) for i in [5, 11]], 'gray', width=r * 6)
        agnostic_draw.line([tuple(pose_data[i]) for i in [8, 11]], 'gray', width=r * 12)
        agnostic_draw.polygon([tuple(pose_data[i]) for i in [2, 5, 11, 8]], 'gray', 'gray')

        # mask neck
        pointx, pointy = pose_data[1]
        agnostic_draw.rectangle((pointx - r * 5, pointy - r * 9, pointx + r * 5, pointy), 'gray', 'gray')

        # mask arms
        agnostic_draw.line([tuple(pose_data[i]) for i in [2, 5]], 'gray', width=r * 12)
        for i in [2, 5]:
            pointx, pointy = pose_data[i]
            agnostic_draw.ellipse((pointx - r * 5, pointy - r * 6, pointx + r * 5, pointy + r * 6), 'gray', 'gray')
        for i in [3, 4, 6, 7]:
            if (pose_data[i - 1, 0] == 0.0 and pose_data[i - 1, 1] == 0.0) or (
                    pose_data[i, 0] == 0.0 and pose_data[i, 1] == 0.0):
                continue
            agnostic_draw.line([tuple(pose_data[j]) for j in [i - 1, i]], 'gray', width=r * 10)
            pointx, pointy = pose_data[i]
            agnostic_draw.ellipse((pointx - r * 5, pointy - r * 5, pointx + r * 5, pointy + r * 5), 'gray', 'gray')


        w, h = im.size
        for parse_id, pose_ids in [(14, [5, 6, 7]), (15, [2, 3, 4])]:
            mask_arm = Image.new('L', (w, h), 'white')
            # mask_arm = Image.new('L', (768, 1024), 'white')
            mask_arm_draw = ImageDraw.Draw(mask_arm)
            pointx, pointy = pose_data[pose_ids[0]]
            mask_arm_draw.ellipse((pointx - r * 5, pointy - r * 6, pointx + r * 5, pointy + r * 6), 'black', 'black')
            for i in pose_ids[1:]:
                if (pose_data[i - 1, 0] == 0.0 and pose_data[i - 1, 1] == 0.0) or (
                        pose_data[i, 0] == 0.0 and pose_data[i, 1] == 0.0):
                    continue
                mask_arm_draw.line([tuple(pose_data[j]) for j in [i - 1, i]], 'black', width=r * 10)
                pointx, pointy = pose_data[i]
                if i != pose_ids[-1]:
                    mask_arm_draw.ellipse((pointx - r * 5, pointy - r * 5, pointx + r * 5, pointy + r * 5), 'black',
                                          'black')
            mask_arm_draw.ellipse((pointx - r * 4, pointy - r * 4, pointx + r * 4, pointy + r * 4), 'black', 'black')

            parse_arm = (np.array(mask_arm) / 255) * (parse_array == parse_id).astype(np.float32)
            agnostic.paste(im, None, Image.fromarray(np.uint8(parse_arm * 255), 'L'))

        agnostic.paste(im, None, Image.fromarray(np.uint8(parse_head * 255), 'L'))
        agnostic.paste(im, None, Image.fromarray(np.uint8(parse_lower * 255), 'L'))
        return agnostic
