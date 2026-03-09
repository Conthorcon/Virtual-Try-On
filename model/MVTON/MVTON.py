import os
import cv2
import torch
import numpy as np

from omegaconf import OmegaConf
from tqdm import tqdm, trange
from PIL import Image
from einops import rearrange

from torchvision.transforms import Resize
from torchvision import transforms

from model.MVTON.ldm.models.diffusion.ddim import DDIMSampler
from model.MVTON.ldm.models.diffusion.plms import PLMSSampler

from pytorch_lightning import seed_everything
from model.MVTON.utils import load_model_from_config, get_args

import preprocess as pre
from model import build_model
from engine import test_gmm, test_tom


device = torch.device("cpu")


def setup_mvton():
    opt = get_args()

    seed_everything(opt.seed)

    config = OmegaConf.load(f"{opt.config}")
    version = opt.config.split('/')[-1].split('.')[0]
    model = load_model_from_config(config, f"{opt.ckpt}")

    if opt.plms:
        sampler = PLMSSampler(model)
    else:
        sampler = DDIMSampler(model)

    start_code = None
    if opt.fixed_code:
        start_code = torch.randn([opt.n_samples, opt.C, opt.H // opt.f, opt.W // opt.f], device=device)

    return model, sampler, opt, start_code

def run_mvton(model, sampler, data, opt, start_code=None):
    print("Running on device: {}".format(device))

    with torch.no_grad():
        with model.ema_scope():
            mask_tensor = data['inpaint_mask'].unsqueeze(0)
            inpaint_image = data['inpaint_image'].unsqueeze(0)
            ref_tensor = data['ref_imgs'].unsqueeze(0)
            feat_tensor = data['warp_feat'].unsqueeze(0)
            image_tensor = data['GT'].unsqueeze(0)
            controlnet_cond = data['controlnet_cond'].unsqueeze(0)
            controlnet_cond = controlnet_cond.to(device)

            test_model_kwargs = {}
            test_model_kwargs['inpaint_mask'] = mask_tensor.to(device)
            test_model_kwargs['inpaint_image'] = inpaint_image.to(device)
            feat_tensor = feat_tensor.to(device)
            ref_tensor = ref_tensor.to(device)

            uc = None
            if opt.scale != 1.0:
                uc = model.learnable_vector
                uc = uc.repeat(ref_tensor.size(0), 1, 1)
            c = model.get_learned_conditioning(ref_tensor.float())
            c = model.proj_out(c)

            z_inpaint = model.encode_first_stage(test_model_kwargs['inpaint_image'])
            z_inpaint = model.get_first_stage_encoding(z_inpaint).detach()
            test_model_kwargs['inpaint_image'] = z_inpaint
            test_model_kwargs['inpaint_mask'] = Resize([z_inpaint.shape[-2], z_inpaint.shape[-1]])(
                test_model_kwargs['inpaint_mask'])

            warp_feat = model.encode_first_stage(feat_tensor)
            warp_feat = model.get_first_stage_encoding(warp_feat).detach()

            ts = torch.full((1,), 999, device=device, dtype=torch.long)
            start_code = model.q_sample(warp_feat, ts)

            # local_controlnet
            x_noisy = torch.cat(
                (start_code, test_model_kwargs['inpaint_image'], test_model_kwargs['inpaint_mask']), dim=1)
            down_samples, _ = model.local_controlnet(x_noisy, ts, encoder_hidden_states=torch.zeros(
                (c.shape[0], 1, 768)), controlnet_cond=controlnet_cond)

            shape = [opt.C, opt.H // opt.f, opt.W // opt.f]
            samples_ddim, _ = sampler.sample(S=opt.ddim_steps,
                                                conditioning=c,
                                                batch_size=opt.n_samples,
                                                shape=shape,
                                                verbose=False,
                                                unconditional_guidance_scale=opt.scale,
                                                unconditional_conditioning=uc,
                                                eta=opt.ddim_eta,
                                                x_T=start_code,
                                                down_samples=down_samples,
                                                test_model_kwargs=test_model_kwargs)

            x_samples_ddim = model.decode_first_stage(samples_ddim)
            x_sample_result = x_samples_ddim
            x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
            x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()

            x_checked_image = x_samples_ddim
            x_checked_image_torch = torch.from_numpy(x_checked_image).permute(0, 3, 1, 2)
            x_source = torch.clamp((image_tensor + 1.0) / 2.0, min=0.0, max=1.0)
            x_result = x_checked_image_torch * (1 - mask_tensor) + mask_tensor * x_source
            
            # hands_mask = data['hands_mask'].unsqueeze(0)
            # x_result = x_result * (1- hands_mask) + hands_mask * x_source

            resize = transforms.Resize((opt.H, int(opt.H / 256 * 192)))

            save_x = resize(x_result[0])
            save_x = 255. * rearrange(save_x.cpu().numpy(), 'c h w -> h w c')
            img = Image.fromarray(save_x.astype(np.uint8))

            return img