import argparse, os, sys, glob
import cv2
import torch
import numpy as np
from PIL import Image

from itertools import islice

import torchvision

from model.MVTON.ldm.util import instantiate_from_config


def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def get_tensor_clip(normalize=True, toTensor=True):
    transform_list = []
    if toTensor:
        transform_list += [torchvision.transforms.ToTensor()]

    if normalize:
        transform_list += [torchvision.transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                                            (0.26862954, 0.26130258, 0.27577711))]
    return torchvision.transforms.Compose(transform_list)


def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]

    return pil_images


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model
    model.eval()
    return model


def put_watermark(img, wm_encoder=None):
    if wm_encoder is not None:
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        img = wm_encoder.encode(img, 'dwtDct')
        img = Image.fromarray(img[:, :, ::-1])
    return img


def load_replacement(x):
    try:
        hwc = x.shape
        y = Image.open("assets/rick.jpeg").convert("RGB").resize((hwc[1], hwc[0]))
        y = (np.array(y) / 255.0).astype(x.dtype)
        assert y.shape == x.shape
        return y
    except Exception:
        return x


def get_tensor(normalize=True, toTensor=True):
    transform_list = []
    if toTensor:
        transform_list += [torchvision.transforms.ToTensor()]

    if normalize:
        transform_list += [torchvision.transforms.Normalize((0.5, 0.5, 0.5),
                                                            (0.5, 0.5, 0.5))]
    return torchvision.transforms.Compose(transform_list)


def get_tensor_clip(normalize=True, toTensor=True):
    transform_list = []
    if toTensor:
        transform_list += [torchvision.transforms.ToTensor()]

    if normalize:
        transform_list += [torchvision.transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                                            (0.26862954, 0.26130258, 0.27577711))]
    return torchvision.transforms.Compose(transform_list)

from types import SimpleNamespace


def get_args():

    args = SimpleNamespace(
        output="test_data",
        name="tryon",
        outdir="static",
        skip_grid=False,
        skip_save=False,
        gpu_id=0,
        ddim_steps=10,
        plms=False,
        fixed_code=False,
        ddim_eta=0.0,
        n_iter=2,
        H=512,
        W=512,
        n_imgs=100,
        C=4,
        f=8,
        n_samples=1,
        n_rows=0,
        scale=1,
        config="configs/viton512.yaml",
        ckpt="ckpt/MVTON/mvton.ckpt",
        seed=23,
        precision="autocast",
        unpaired=False,
        dataroot="",
        workers=1,
        batch_size=4,
        datamode="test",
        stage="GMM",
        data_list="test_pairs.txt",
        fine_width=192,
        fine_height=256,
        radius=5,
        grid_size=5,
        tensorboard_dir='tensorboard',
        result_dir='result',
        gmm_checkpoint='ckpt/GMM/gmm_final.pth',
        tom_checkpoint='ckpt/TOM/tom_final.pth',
        display_count=1,
        shuffle=True
    )

    return args