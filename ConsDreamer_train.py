from __future__ import annotations
import sys
import yaml
import imageio
from random import randint
from utils.loss_utils import l1_loss, ssim, tv_loss
from gaussian_renderer import render, network_gui
from scene import Scene, GaussianModel
import uuid
from tqdm import tqdm
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams, GenerateCamParams, GuidanceParams
import math
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False
from tqdm.notebook import tqdm
from transformers import CLIPTextModel, CLIPTokenizer, logging
from pathlib import Path
import random
logging.set_verbosity_error()
from torchvision.utils import save_image
from torch.cuda.amp import custom_bwd, custom_fwd
from guidance.perpneg_utils import weighted_perpendicular_aggregator
from guidance.sd_step import *
import clip_utils
import os
import torch
import torch.nn.functional as F

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

def rgb2sat(img, T=None):
    max_ = torch.max(img, dim=1, keepdim=True).values + 1e-5
    min_ = torch.min(img, dim=1, keepdim=True).values
    sat = (max_ - min_) / max_
    if T is not None:
        sat = (1 - T) * sat
    return sat


class SpecifyGradient(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, input_tensor, gt_grad):
        ctx.save_for_backward(gt_grad)
        return torch.ones([1], device=input_tensor.device, dtype=input_tensor.dtype)

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_scale):
        gt_grad, = ctx.saved_tensors
        gt_grad = gt_grad * grad_scale
        return gt_grad, None


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


class StableDiffusion(nn.Module):
    flag = 0

    def __init__(self, device, fp16, vram_O, t_range=[0.02, 0.98], max_t_range=0.98, num_train_timesteps=None,
                 ddim_inv=False, textual_inversion_path=None,
                 LoRA_path=None, guidance_opt=None):
        super().__init__()
        self.flag = 0
        self.device = device
        self.precision_t = torch.float16 if fp16 else torch.float32

        print(f'[INFO] loading stable diffusion...')

        model_key = guidance_opt.model_key
        if model_key == None:
            model_key = "stabilityai/stable-diffusion-2-1-base"

        is_safe_tensor = guidance_opt.is_safe_tensor
        base_model_key = model_key

        if is_safe_tensor:
            pipe = StableDiffusionPipeline.from_single_file(model_key, use_safetensors=True,
                                                            torch_dtype=self.precision_t, load_safety_checker=False)
        else:
            pipe = StableDiffusionPipeline.from_pretrained(model_key, torch_dtype=self.precision_t)
            self.pipe = pipe
        self.ism = not guidance_opt.sds
        self.scheduler = DDIMScheduler.from_pretrained(model_key if not is_safe_tensor else base_model_key,
                                                       subfolder="scheduler", torch_dtype=self.precision_t)
        self.sche_func = ddim_step

        if vram_O:
            pipe.enable_sequential_cpu_offload()
            pipe.enable_vae_slicing()
            pipe.unet.to(memory_format=torch.channels_last)
            pipe.enable_attention_slicing(1)
            pipe.enable_model_cpu_offload()

        pipe.enable_xformers_memory_efficient_attention()

        pipe = pipe.to(self.device)
        if textual_inversion_path is not None:
            pipe.load_textual_inversion(textual_inversion_path)
            print("load textual inversion in:.{}".format(textual_inversion_path))

        self.pipe = pipe
        self.vae = pipe.vae
        self.tokenizer = pipe.tokenizer
        self.text_encoder = pipe.text_encoder
        self.unet = pipe.unet
        self.num_train_timesteps = num_train_timesteps if num_train_timesteps is not None else self.scheduler.config.num_train_timesteps
        self.scheduler.set_timesteps(self.num_train_timesteps, device=device)
        self.timesteps = torch.flip(self.scheduler.timesteps, dims=(0,))
        self.min_step = int(self.num_train_timesteps * t_range[0])
        self.max_step = int(self.num_train_timesteps * t_range[1])
        self.warmup_step = int(self.num_train_timesteps * (max_t_range - t_range[1]))
        self.noise_temp = None
        self.noise_gen = torch.Generator(self.device)
        self.noise_gen.manual_seed(guidance_opt.noise_seed)
        self.alphas = self.scheduler.alphas_cumprod.to(self.device)
        self.rgb_latent_factors = torch.tensor([
            [0.298, 0.207, 0.208],
            [0.187, 0.286, 0.173],
            [-0.158, 0.189, 0.264],
            [-0.184, -0.271, -0.473]
        ], device=self.device)
        print(f'[INFO] loaded stable diffusion!')

    def augmentation(self, *tensors):
        augs = T.Compose([
            T.RandomHorizontalFlip(p=0.5),
        ])

        channels = [ten.shape[1] for ten in tensors]
        tensors_concat = torch.concat(tensors, dim=1)
        tensors_concat = augs(tensors_concat)

        results = []
        cur_c = 0
        for i in range(len(channels)):
            results.append(tensors_concat[:, cur_c:cur_c + channels[i], ...])
            cur_c += channels[i]
        return (ten for ten in results)

    def add_noise_with_cfg(self, latents, noise,
                           ind_t, ind_prev_t,
                           text_embeddings=None, cfg=1.0,
                           delta_t=1, inv_steps=1,
                           is_noisy_latent=False,
                           eta=0.0):
        text_embeddings = text_embeddings.to(self.precision_t)
        if cfg <= 1.0:
            uncond_text_embedding = \
                text_embeddings.reshape(2, -1, text_embeddings.shape[-2], text_embeddings.shape[-1])[1]

        unet = self.unet

        if is_noisy_latent:
            prev_noisy_lat = latents
        else:
            prev_noisy_lat = self.scheduler.add_noise(latents, noise, self.timesteps[ind_prev_t])

        cur_ind_t = ind_prev_t
        cur_noisy_lat = prev_noisy_lat
        pred_scores = []
        for i in range(inv_steps):
            # pred noise
            cur_noisy_lat_ = self.scheduler.scale_model_input(cur_noisy_lat, self.timesteps[cur_ind_t]).to(
                self.precision_t)

            if cfg > 1.0:
                latent_model_input = torch.cat([cur_noisy_lat_, cur_noisy_lat_])
                timestep_model_input = self.timesteps[cur_ind_t].reshape(1, 1).repeat(latent_model_input.shape[0],
                                                                                      1).reshape(-1)
                unet_output = unet(latent_model_input, timestep_model_input,
                                   encoder_hidden_states=text_embeddings).sample

                uncond, cond = torch.chunk(unet_output, chunks=2)

                unet_output = cond + cfg * (uncond - cond)
            else:
                timestep_model_input = self.timesteps[cur_ind_t].reshape(1, 1).repeat(cur_noisy_lat_.shape[0],
                                                                                      1).reshape(-1)

                StableDiffusion.flag = 0
                unet_output = unet(cur_noisy_lat_, timestep_model_input,
                                   encoder_hidden_states=uncond_text_embedding).sample
            pred_scores.append((cur_ind_t, unet_output))
            next_ind_t = min(cur_ind_t + delta_t, ind_t)
            cur_t, next_t = self.timesteps[cur_ind_t], self.timesteps[next_ind_t]
            delta_t_ = next_t - cur_t if isinstance(self.scheduler, DDIMScheduler) else next_ind_t - cur_ind_t
            cur_noisy_lat = self.sche_func(self.scheduler, unet_output, cur_t, cur_noisy_lat, -delta_t_,
                                           eta).prev_sample
            cur_ind_t = next_ind_t
            del unet_output
            torch.cuda.empty_cache()
            if cur_ind_t == ind_t:
                break
        return prev_noisy_lat, cur_noisy_lat, pred_scores[::-1]

    @torch.no_grad()
    def get_text_embeds(self, prompt, resolution=(512, 512)):
        inputs = self.tokenizer(prompt, padding='max_length', max_length=self.tokenizer.model_max_length,
                                truncation=True, return_tensors='pt')
        embeddings = self.text_encoder(inputs.input_ids.to(self.device))[0]
        return embeddings

    def train_step_perpneg(self, text_embeddings, pred_rgb, pred_depth=None, pred_alpha=None,
                           grad_scale=1,
                           save_folder: Path = None, iteration=0, warm_up_rate=0, weights=0,
                           resolution=(512, 512), guidance_opt=None, as_latent=False, embedding_inverse=None,
                           iteration_flag1=2500, iteration_flag2=4000, gaussians=None):
        pred_rgb, pred_depth, pred_alpha = self.augmentation(pred_rgb, pred_depth, pred_alpha)
        B = pred_rgb.shape[0]
        K = text_embeddings.shape[0] - 1
        if as_latent:
            latents, _ = self.encode_imgs(pred_depth.repeat(1, 3, 1, 1).to(self.precision_t))
        else:
            latents, _ = self.encode_imgs(pred_rgb.to(self.precision_t))
        weights = weights.reshape(-1)
        noise = torch.randn((latents.shape[0], 4, resolution[0] // 8, resolution[1] // 8,), dtype=latents.dtype,
                            device=latents.device, generator=self.noise_gen) + 0.1 * torch.randn((1, 4, 1, 1),
                                                                                                 device=latents.device).repeat(
            latents.shape[0], 1, 1, 1)
        inverse_text_embeddings = embedding_inverse.unsqueeze(1).repeat(1, B, 1, 1).reshape(-1,
                                                                                            embedding_inverse.shape[-2],
                                                                                            embedding_inverse.shape[-1])
        text_embeddings = text_embeddings.reshape(-1, text_embeddings.shape[-2], text_embeddings.shape[-1])
        if guidance_opt.annealing_intervals:
            current_delta_t = int(
                guidance_opt.delta_t + np.ceil((warm_up_rate) * (guidance_opt.delta_t_start - guidance_opt.delta_t)))
        else:
            current_delta_t = guidance_opt.delta_t
        ind_t = \
        torch.randint(self.min_step, self.max_step + int(self.warmup_step * warm_up_rate), (1,), dtype=torch.long,
                      generator=self.noise_gen, device=self.device)[0]
        ind_prev_t = max(ind_t - current_delta_t, torch.ones_like(ind_t) * 0)
        t = self.timesteps[ind_t]
        prev_t = self.timesteps[ind_prev_t]

        if (1200 > iteration > 1500 and random.random() < 0.3 and prev_t <= 300 and not as_latent) or (
                iteration < 3000 and random.random() < 0.2 and as_latent and prev_t <= 400):

            with torch.no_grad():
                # Step 1: sample x_s with larger steps
                xs_delta_t = guidance_opt.xs_delta_t if guidance_opt.xs_delta_t is not None else current_delta_t
                xs_inv_steps = guidance_opt.xs_inv_steps if guidance_opt.xs_inv_steps is not None else int(
                    np.ceil(ind_prev_t / xs_delta_t))
                starting_ind = max(ind_prev_t - xs_delta_t * xs_inv_steps, torch.ones_like(ind_t) * 0)

                _, prev_latents_noisy, pred_scores_xs = self.add_noise_with_cfg(latents, noise, ind_prev_t,
                                                                                starting_ind, inverse_text_embeddings,
                                                                                guidance_opt.denoise_guidance_scale,
                                                                                xs_delta_t, xs_inv_steps,
                                                                                eta=guidance_opt.xs_eta)
                # Step 2: sample x_t
                _, latents_noisy, pred_scores_xt = self.add_noise_with_cfg(prev_latents_noisy, noise, ind_t, ind_prev_t,
                                                                           inverse_text_embeddings,
                                                                           guidance_opt.denoise_guidance_scale,
                                                                           current_delta_t, 1, is_noisy_latent=True)
                pred_scores = pred_scores_xt + pred_scores_xs
                target = pred_scores[0][1]
            with torch.no_grad():
                latent_model_input = latents_noisy[None, :, ...].repeat(1 + K, 1, 1, 1, 1).reshape(-1, 4,
                                                                                                   resolution[0] // 8,
                                                                                                   resolution[1] // 8, )
                tt = t.reshape(1, 1).repeat(latent_model_input.shape[0], 1).reshape(-1)
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, tt[0])
                StableDiffusion.flag = 1
                unet_output = self.unet(latent_model_input.to(self.precision_t), tt.to(self.precision_t),
                                        encoder_hidden_states=text_embeddings.to(self.precision_t)).sample
                StableDiffusion.flag = 0
                unet_output = unet_output.reshape(1 + K, -1, 4, resolution[0] // 8, resolution[1] // 8, )
                noise_pred_uncond, noise_pred_text = unet_output[:1].reshape(-1, 4, resolution[0] // 8,
                                                                             resolution[1] // 8, ), unet_output[
                                                                                                    1:].reshape(-1, 4,
                                                                                                                resolution[
                                                                                                                    0] // 8,
                                                                                                                resolution[
                                                                                                                    1] // 8, )
                pred_x0_latent_sp = pred_original(self.scheduler, noise_pred_uncond, prev_t, prev_latents_noisy)
                pred_x0_sp = self.decode_latents(pred_x0_latent_sp.type(self.precision_t))
                w = lambda alphas: (((1 - alphas) / alphas) ** 0.5)
                grad = w(self.alphas[t]) * (pred_x0_latent_sp - latents)
                grad = torch.nan_to_num(grad_scale * grad)
                loss = SpecifyGradient.apply(latents, grad)
            pred_x0_latent_sp = pred_original(self.scheduler, noise_pred_uncond, prev_t, prev_latents_noisy)
            pred_x0_sp = self.decode_latents(pred_x0_latent_sp.type(self.precision_t))
            flaggg = 1
        else:
            with torch.no_grad():
                # Step 1: sample x_s with larger steps
                xs_delta_t = guidance_opt.xs_delta_t if guidance_opt.xs_delta_t is not None else current_delta_t
                xs_inv_steps = guidance_opt.xs_inv_steps if guidance_opt.xs_inv_steps is not None else int(
                    np.ceil(ind_prev_t / xs_delta_t))
                starting_ind = max(ind_prev_t - xs_delta_t * xs_inv_steps, torch.ones_like(ind_t) * 0)

                _, prev_latents_noisy, pred_scores_xs = self.add_noise_with_cfg(latents, noise, ind_prev_t,
                                                                                starting_ind, inverse_text_embeddings,
                                                                                guidance_opt.denoise_guidance_scale,
                                                                                xs_delta_t, xs_inv_steps,
                                                                                eta=guidance_opt.xs_eta)
                # Step 2: sample x_t
                _, latents_noisy, pred_scores_xt = self.add_noise_with_cfg(prev_latents_noisy, noise, ind_t, ind_prev_t,
                                                                           inverse_text_embeddings,
                                                                           guidance_opt.denoise_guidance_scale,
                                                                           current_delta_t, 1, is_noisy_latent=True)
                pred_scores = pred_scores_xt + pred_scores_xs
                target = pred_scores[0][1]
            with torch.no_grad():
                latent_model_input = latents_noisy[None, :, ...].repeat(1 + K, 1, 1, 1, 1).reshape(-1, 4,
                                                                                                   resolution[0] // 8,
                                                                                                   resolution[1] // 8, )
                tt = t.reshape(1, 1).repeat(latent_model_input.shape[0], 1).reshape(-1)
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, tt[0])
                unet_output = self.unet(latent_model_input.to(self.precision_t), tt.to(self.precision_t),
                                        encoder_hidden_states=text_embeddings.to(self.precision_t)).sample
                unet_output = unet_output.reshape(1 + K, -1, 4, resolution[0] // 8, resolution[1] // 8, )
                noise_pred_uncond, noise_pred_text = unet_output[:1].reshape(-1, 4, resolution[0] // 8,
                                                                             resolution[1] // 8, ), unet_output[
                                                                                                    1:].reshape(
                    -1, 4, resolution[0] // 8, resolution[1] // 8, )
                delta_noise_preds = noise_pred_text - noise_pred_uncond.repeat(K, 1, 1, 1)
                delta_DSD = weighted_perpendicular_aggregator(delta_noise_preds, \
                                                              weights, \
                                                              B)

            pred_noise = noise_pred_uncond + guidance_opt.guidance_scale * delta_DSD
            w = lambda alphas: (((1 - alphas) / alphas) ** 0.5)
            grad = w(self.alphas[t]) * (pred_noise - target)
            grad = torch.nan_to_num(grad_scale * grad)
            loss = SpecifyGradient.apply(latents, grad)
            flaggg = 0
            pred_x0_latent_sp = pred_original(self.scheduler, noise_pred_uncond, prev_t, prev_latents_noisy)
            pred_x0_sp = self.decode_latents(pred_x0_latent_sp.type(self.precision_t))

            def ordered_similarity_loss(sim_A, sim_B, sim_C):
                diff_AB = sim_B - sim_A
                diff_BC = sim_C - sim_B
                loss_AB = torch.relu(diff_AB)
                loss_BC = torch.relu(diff_BC)
                lossABC = loss_AB + loss_BC
                return lossABC

            def compute_alex_loss(pred_x0_sp):
                total_score = 0
                for i in range(3):
                    alex_x2next = lpips_alex(pred_x0_sp[i + 1][0], pred_x0_sp[i][0])
                    total_score += alex_x2next
                return total_score / 3

            def compute_P_loss(embed_rgb):
                sim = []
                for i in range(3):
                    consistency_x_to_1 = torch.cosine_similarity(embed_rgb[0][0], embed_rgb[i + 1][0], dim=-1)
                    sim.append(consistency_x_to_1)
                lossABC = ordered_similarity_loss(sim[0], sim[1], sim[2])
                return lossABC

            if not as_latent:
                lambda_sum_cos_loss = 1
                lambda_ABC_cos_loss = 1
                if iteration >= iteration_flag2:
                    clip_loss1 = compute_alex_loss(pred_rgb)
                    clip_loss = clip_loss1 * lambda_sum_cos_loss
                    loss = loss + clip_loss
                elif iteration <= iteration_flag1:
                    pred_rgb_resize = F.interpolate(pred_x0_sp, size=(224, 224), mode="bicubic")
                    embed_rgb = embed(pred_rgb_resize)
                    clip_loss1 = compute_P_loss(embed_rgb)
                    clip_loss = clip_loss1 * lambda_ABC_cos_loss
                    loss = loss + clip_loss

        if iteration % 10 == 0:
            if not flaggg:
                noise_pred_post = noise_pred_uncond + guidance_opt.guidance_scale * delta_DSD
                lat2rgb = lambda x: torch.clip(
                    (x.permute(0, 2, 3, 1) @ self.rgb_latent_factors.to(x.dtype)).permute(0, 3, 1, 2), 0., 1.)
                save_path_iter = os.path.join(save_folder, "iter_{}_step_{}.jpg".format(iteration, prev_t.item()))

                with torch.no_grad():
                    pred_x0_latent_pos = pred_original(self.scheduler, noise_pred_post, prev_t, prev_latents_noisy)
                    pred_x0_pos = self.decode_latents(pred_x0_latent_pos.type(self.precision_t))
                    grad_abs = torch.abs(grad.detach())
                    norm_grad = F.interpolate((grad_abs / grad_abs.max()).mean(dim=1, keepdim=True),
                                              (resolution[0], resolution[1]), mode='bilinear',
                                              align_corners=False).repeat(
                        1, 3, 1, 1)
                    latents_rgb = F.interpolate(lat2rgb(latents), (resolution[0], resolution[1]), mode='bilinear',
                                                align_corners=False)
                    latents_sp_rgb = F.interpolate(lat2rgb(pred_x0_latent_sp), (resolution[0], resolution[1]),
                                                   mode='bilinear', align_corners=False)
                    viz_images = torch.cat([pred_rgb,
                                            pred_depth.repeat(1, 3, 1, 1),
                                            pred_alpha.repeat(1, 3, 1, 1),
                                            rgb2sat(pred_rgb, pred_alpha).repeat(1, 3, 1, 1),
                                            latents_rgb, latents_sp_rgb,
                                            pred_x0_pos, pred_x0_sp,
                                            norm_grad], dim=0)
                    save_image(viz_images, save_path_iter)
            else:
                lat2rgb = lambda x: torch.clip(
                    (x.permute(0, 2, 3, 1) @ self.rgb_latent_factors.to(x.dtype)).permute(0, 3, 1, 2), 0., 1.)
                save_path_iter = os.path.join(save_folder, "iter_{}_step_{}.jpg".format(iteration, prev_t.item()))
                with torch.no_grad():
                    grad_abs = torch.abs(grad.detach())
                    norm_grad = F.interpolate((grad_abs / grad_abs.max()).mean(dim=1, keepdim=True),
                                              (resolution[0], resolution[1]), mode='bilinear',
                                              align_corners=False).repeat(
                        1, 3, 1, 1)
                    latents_rgb = F.interpolate(lat2rgb(latents), (resolution[0], resolution[1]), mode='bilinear',
                                                align_corners=False)
                    latents_sp_rgb = F.interpolate(lat2rgb(pred_x0_latent_sp), (resolution[0], resolution[1]),
                                                   mode='bilinear', align_corners=False)

                    viz_images = torch.cat([pred_rgb,
                                            pred_depth.repeat(1, 3, 1, 1),
                                            pred_alpha.repeat(1, 3, 1, 1),
                                            rgb2sat(pred_rgb, pred_alpha).repeat(1, 3, 1, 1),
                                            latents_rgb, latents_sp_rgb,
                                            pred_x0_sp, norm_grad], dim=0)
                    save_image(viz_images, save_path_iter)
        return loss

    def train_step(self, text_embeddings, pred_rgb, pred_depth=None, pred_alpha=None,
                   grad_scale=1,
                   save_folder: Path = None, iteration=0, warm_up_rate=0,
                   resolution=(512, 512), guidance_opt=None, as_latent=False, embedding_inverse=None):

        pred_rgb, pred_depth, pred_alpha = self.augmentation(pred_rgb, pred_depth, pred_alpha)

        B = pred_rgb.shape[0]
        K = text_embeddings.shape[0] - 1

        if as_latent:
            latents, _ = self.encode_imgs(pred_depth.repeat(1, 3, 1, 1).to(self.precision_t))
        else:
            latents, _ = self.encode_imgs(pred_rgb.to(self.precision_t))

        if self.noise_temp is None:
            self.noise_temp = torch.randn((latents.shape[0], 4, resolution[0] // 8, resolution[1] // 8,),
                                          dtype=latents.dtype, device=latents.device,
                                          generator=self.noise_gen) + 0.1 * torch.randn((1, 4, 1, 1),
                                                                                        device=latents.device).repeat(
                latents.shape[0], 1, 1, 1)

        if guidance_opt.fix_noise:
            noise = self.noise_temp
        else:
            noise = torch.randn((latents.shape[0], 4, resolution[0] // 8, resolution[1] // 8,), dtype=latents.dtype,
                                device=latents.device, generator=self.noise_gen) + 0.1 * torch.randn((1, 4, 1, 1),
                                                                                                     device=latents.device).repeat(
                latents.shape[0], 1, 1, 1)

        text_embeddings = text_embeddings[:, :, ...]
        text_embeddings = text_embeddings.reshape(-1, text_embeddings.shape[-2],
                                                  text_embeddings.shape[-1])  # make it k+1, c * t, ...

        inverse_text_embeddings = embedding_inverse.unsqueeze(1).repeat(1, B, 1, 1).reshape(-1,
                                                                                            embedding_inverse.shape[-2],
                                                                                            embedding_inverse.shape[-1])

        if guidance_opt.annealing_intervals:
            current_delta_t = int(
                guidance_opt.delta_t + (warm_up_rate) * (guidance_opt.delta_t_start - guidance_opt.delta_t))
        else:
            current_delta_t = guidance_opt.delta_t

        ind_t = \
            torch.randint(self.min_step, self.max_step + int(self.warmup_step * warm_up_rate), (1,), dtype=torch.long,
                          generator=self.noise_gen, device=self.device)[0]
        ind_prev_t = max(ind_t - current_delta_t, torch.ones_like(ind_t) * 0)

        t = self.timesteps[ind_t]
        prev_t = self.timesteps[ind_prev_t]

        with torch.no_grad():
            # step unroll via ddim inversion
            if self.ism:
                prev_latents_noisy = self.scheduler.add_noise(latents, noise, prev_t)
                latents_noisy = self.scheduler.add_noise(latents, noise, t)
                target = noise
            else:
                # Step 1: sample x_s with larger steps
                xs_delta_t = guidance_opt.xs_delta_t if guidance_opt.xs_delta_t is not None else current_delta_t
                xs_inv_steps = guidance_opt.xs_inv_steps if guidance_opt.xs_inv_steps is not None else int(
                    np.ceil(ind_prev_t / xs_delta_t))
                starting_ind = max(ind_prev_t - xs_delta_t * xs_inv_steps, torch.ones_like(ind_t) * 0)

                _, prev_latents_noisy, pred_scores_xs = self.add_noise_with_cfg(latents, noise, ind_prev_t,
                                                                                starting_ind, inverse_text_embeddings,
                                                                                guidance_opt.denoise_guidance_scale,
                                                                                xs_delta_t, xs_inv_steps,
                                                                                eta=guidance_opt.xs_eta)
                # Step 2: sample x_t
                _, latents_noisy, pred_scores_xt = self.add_noise_with_cfg(prev_latents_noisy, noise, ind_t, ind_prev_t,
                                                                           inverse_text_embeddings,
                                                                           guidance_opt.denoise_guidance_scale,
                                                                           current_delta_t, 1, is_noisy_latent=True)

                pred_scores = pred_scores_xt + pred_scores_xs
                target = pred_scores[0][1]

        with torch.no_grad():
            latent_model_input = latents_noisy[None, :, ...].repeat(2, 1, 1, 1, 1).reshape(-1, 4, resolution[0] // 8,
                                                                                           resolution[1] // 8, )
            tt = t.reshape(1, 1).repeat(latent_model_input.shape[0], 1).reshape(-1)
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, tt[0])
            unet_output = self.unet(latent_model_input.to(self.precision_t), tt.to(self.precision_t),
                                    encoder_hidden_states=text_embeddings.to(self.precision_t)).sample
            unet_output = unet_output.reshape(2, -1, 4, resolution[0] // 8, resolution[1] // 8, )
            noise_pred_uncond, noise_pred_text = unet_output[:1].reshape(-1, 4, resolution[0] // 8,
                                                                         resolution[1] // 8, ), unet_output[1:].reshape(
                -1, 4, resolution[0] // 8, resolution[1] // 8, )
            delta_DSD = noise_pred_text - noise_pred_uncond
        pred_noise = noise_pred_uncond + guidance_opt.guidance_scale * delta_DSD
        w = lambda alphas: (((1 - alphas) / alphas) ** 0.5)
        grad = w(self.alphas[t]) * (pred_noise - target)
        grad = torch.nan_to_num(grad_scale * grad)
        loss = SpecifyGradient.apply(latents, grad)

        if iteration % 10 == 0:
            noise_pred_post = noise_pred_uncond + guidance_opt.guidance_scale * delta_DSD
            lat2rgb = lambda x: torch.clip(
                (x.permute(0, 2, 3, 1) @ self.rgb_latent_factors.to(x.dtype)).permute(0, 3, 1, 2), 0., 1.)
            save_path_iter = os.path.join(save_folder, "iter_{}_step_{}.jpg".format(iteration, prev_t.item()))
            with torch.no_grad():
                pred_x0_latent_sp = pred_original(self.scheduler, noise_pred_uncond, prev_t, prev_latents_noisy)
                pred_x0_latent_pos = pred_original(self.scheduler, noise_pred_post, prev_t, prev_latents_noisy)
                pred_x0_pos = self.decode_latents(pred_x0_latent_pos.type(self.precision_t))
                pred_x0_sp = self.decode_latents(pred_x0_latent_sp.type(self.precision_t))
                grad_abs = torch.abs(grad.detach())
                norm_grad = F.interpolate((grad_abs / grad_abs.max()).mean(dim=1, keepdim=True),
                                          (resolution[0], resolution[1]), mode='bilinear', align_corners=False).repeat(
                    1, 3, 1, 1)
                latents_rgb = F.interpolate(lat2rgb(latents), (resolution[0], resolution[1]), mode='bilinear',
                                            align_corners=False)
                latents_sp_rgb = F.interpolate(lat2rgb(pred_x0_latent_sp), (resolution[0], resolution[1]),
                                               mode='bilinear', align_corners=False)
                viz_images = torch.cat([pred_rgb,
                                        pred_depth.repeat(1, 3, 1, 1),
                                        pred_alpha.repeat(1, 3, 1, 1),
                                        rgb2sat(pred_rgb, pred_alpha).repeat(1, 3, 1, 1),
                                        latents_rgb, latents_sp_rgb,
                                        pred_x0_pos, pred_x0_sp,
                                        norm_grad], dim=0)
                save_image(viz_images, save_path_iter)
        return loss

    def decode_latents(self, latents):
        target_dtype = latents.dtype
        latents = latents / self.vae.config.scaling_factor
        imgs = self.vae.decode(latents.to(self.vae.dtype)).sample
        imgs = (imgs / 2 + 0.5).clamp(0, 1)
        return imgs.to(target_dtype)

    def encode_imgs(self, imgs):
        target_dtype = imgs.dtype
        # imgs: [B, 3, H, W]
        imgs = 2 * imgs - 1
        posterior = self.vae.encode(imgs.to(self.vae.dtype)).latent_dist
        kl_divergence = posterior.kl()
        latents = posterior.sample() * self.vae.config.scaling_factor
        return latents.to(target_dtype), kl_divergence


from dataclasses import dataclass, field
from typing import Tuple, Dict, List
import torch
from torch import nn
from jaxtyping import Float


@dataclass
class PromptEmbedding:
    prompt: str
    tokenwise_embeddings: Dict[str, Float[torch.Tensor, 'n d']]
    tokenwise_embedding_spans: Dict[str, List[Tuple[int, int]]]

    @staticmethod
    def merge(*embs: PromptEmbedding) -> PromptEmbedding:
        emb_joined = None
        for emb in embs:
            if emb_joined is None:
                emb_joined = PromptEmbedding(
                    prompt=emb.prompt,
                    tokenwise_embeddings={k: v for k, v in emb.tokenwise_embeddings.items()},
                    tokenwise_embedding_spans={k: v for k, v in emb.tokenwise_embedding_spans.items()},
                )
            else:
                assert emb.prompt == emb_joined.prompt
                emb_joined.tokenwise_embeddings = emb_joined.tokenwise_embeddings | emb.tokenwise_embeddings
                emb_joined.tokenwise_embedding_spans = emb_joined.tokenwise_embedding_spans | emb.tokenwise_embedding_spans
        return emb_joined

    def get_tokenwise_mask(self, characterwise_mask: List[bool]) -> Dict[str, List[bool]]:
        tokenwise_masks = {}
        for k, t_embs, t_spans in ((k, self.tokenwise_embeddings[k], self.tokenwise_embedding_spans[k]) for k in
                                   self.tokenwise_embeddings):
            token_mask = [False] * len(t_embs)
            for i_t, (t_span_start, t_span_end) in enumerate(t_spans):
                if t_span_start != t_span_end:
                    m = characterwise_mask[t_span_start:t_span_end]
                    assert all(m) or not any(m), 'Inconsistent mask'
                    if all(m):
                        token_mask[i_t] = True
            tokenwise_masks[k] = token_mask
        return tokenwise_masks


class EmbeddingDelta(nn.Module):
    def __init__(self, dims: Dict[str, int]) -> None:
        super().__init__()
        self.tokenwise_delta_front = nn.ParameterDict(
            {k: nn.Parameter(torch.zeros(d), requires_grad=True) for k, d in dims.items()})
        self.tokenwise_delta_side = nn.ParameterDict(
            {k: nn.Parameter(torch.zeros(d), requires_grad=True) for k, d in dims.items()})
        self.tokenwise_delta_back = nn.ParameterDict(
            {k: nn.Parameter(torch.zeros(d), requires_grad=True) for k, d in dims.items()})

    def apply_back(self, emb: PromptEmbedding, characterwise_mask: List[bool], alpha: float = 1.,
                   flag="") -> PromptEmbedding:
        tokenwise_embeddings = {}
        matching_keys = [k for k in self.tokenwise_delta_front if k in emb.tokenwise_embeddings]
        directions = ['front', 'side', 'back']
        for k in matching_keys:
            t_embs = emb.tokenwise_embeddings[k]
            t_spans = emb.tokenwise_embedding_spans[k]
            projections = {direction: [] for direction in directions}
            for direction in directions:
                token_mask = [0] * len(t_embs)
                for i_t, (t_span_start, t_span_end) in enumerate(t_spans):
                    if t_span_start != t_span_end:
                        m = characterwise_mask[t_span_start:t_span_end]
                        if all(m):
                            token_mask[i_t] = 1
                            delta_vector = getattr(self, f'tokenwise_delta_{direction}')[k].unsqueeze(0).to(
                                t_embs.dtype)
                            token_emb = t_embs[i_t].unsqueeze(0)
                            if direction == "front" or direction == "side" or direction == "back":
                                projection = (torch.sum(token_emb * delta_vector) / torch.sum(
                                    delta_vector * delta_vector)) * delta_vector
                                t_embs[i_t] -= projection.squeeze(0)
            tokenwise_embeddings[k] = (t_embs + alpha * torch.tensor(token_mask, dtype=t_embs.dtype,
                                                                     device=t_embs.device).unsqueeze(-1) * \
                                       self.tokenwise_delta_back[k].unsqueeze(0).to(t_embs.dtype))
        return PromptEmbedding(
            prompt=emb.prompt,
            tokenwise_embeddings=tokenwise_embeddings,
            tokenwise_embedding_spans=emb.tokenwise_embedding_spans,
        )

    def apply_side(self, emb: PromptEmbedding, characterwise_mask: List[bool], alpha: float = 1.,
                   flag="") -> PromptEmbedding:
        tokenwise_embeddings = {}
        matching_keys = [k for k in self.tokenwise_delta_front if k in emb.tokenwise_embeddings]
        directions = ['front', 'side', 'back']
        for k in matching_keys:
            t_embs = emb.tokenwise_embeddings[k]
            t_spans = emb.tokenwise_embedding_spans[k]
            projections = {direction: [] for direction in directions}
            for direction in directions:
                token_mask = [0] * len(t_embs)
                for i_t, (t_span_start, t_span_end) in enumerate(t_spans):
                    if t_span_start != t_span_end:
                        m = characterwise_mask[t_span_start:t_span_end]
                        if all(m):
                            token_mask[i_t] = 1
                            delta_vector = getattr(self, f'tokenwise_delta_{direction}')[k].unsqueeze(0).to(
                                t_embs.dtype)
                            token_emb = t_embs[i_t].unsqueeze(0)
                            if direction == "front" or direction == "side" or direction == "back":
                                projection = (torch.sum(token_emb * delta_vector) / torch.sum(
                                    delta_vector * delta_vector)) * delta_vector
                                t_embs[i_t] -= projection.squeeze(0)
            tokenwise_embeddings[k] = (t_embs + alpha * torch.tensor(token_mask, dtype=t_embs.dtype,
                                                                     device=t_embs.device).unsqueeze(-1) * \
                                       self.tokenwise_delta_side[k].unsqueeze(0).to(t_embs.dtype))
        return PromptEmbedding(
            prompt=emb.prompt,
            tokenwise_embeddings=tokenwise_embeddings,
            tokenwise_embedding_spans=emb.tokenwise_embedding_spans,
        )


import re
from typing import List


def get_mask_regex(prompt: str, keyword: str, verbose: bool = True) -> List[bool]:
    characterwise_mask = [False] * len(prompt)
    num_matches = 0
    for m in re.finditer(keyword, prompt):
        num_matches += 1
        characterwise_mask[m.span()[0]:m.span()[1]] = [True] * len(m.group(0))
    return characterwise_mask


def get_mask(prompt: str, target: str, verbose: bool = True) -> List[bool]:
    return get_mask_regex(prompt=prompt, keyword=re.escape(target), verbose=verbose)


def Token_spans(prompt: str) -> Dict[str, List[Tuple[int, int]]]:
    SPECIAL_TOKENS = ['<|startoftext|>', '<|endoftext|>']
    all_token_spans: Dict[str, List[Tuple[int, int]]] = {}
    for name in {'clip_encoder'}:
        d = guidance.tokenizer(prompt)
        token_ids = d['input_ids']
        vocab = guidance.tokenizer.get_vocab()
        vocab_reverse = {v: k for k, v in vocab.items()}
        tokens = [vocab_reverse[i] for i in token_ids]
        token_spans = []
        i = 0
        skipped_chars = []
        for token in tokens:
            if token in SPECIAL_TOKENS:
                token_spans.append((i, i))
                continue
            if token.endswith('</w>'):
                token = token[:-len('</w>')]
            while prompt[i] != token[0]:
                skipped_chars.append(prompt[i])
                i += 1
            assert i + len(token) <= len(prompt)
            assert prompt[i:(i + len(token))] == token, 'Misaligned'
            token_spans.append((i, i + len(token)))
            i += len(token)
        skipped_chars = ''.join(skipped_chars) + prompt[i:]
        assert not any(l.isalpha() for l in
                       skipped_chars), f'Failed to assign some alphabetical characters to tokens for tokenizer {name}. Prompt: "{prompt}", unassigned characters: "{skipped_chars}".'
        all_token_spans[name] = token_spans
    return all_token_spans


def adjust_text_embeddings(embeddings, azimuth, guidance_opt, iteration=0):
    text_z_list = []
    weights_list = []
    K = 0
    text_z_, weights_ = get_pos_neg_text_embeddings(embeddings, azimuth, guidance_opt, iteration=iteration)

    text_z_depth, weights_depth = get_pos_neg_text_embeddings(embeddings, azimuth, guidance_opt)

    K = max(K, weights_.shape[0])
    text_z_list.append(text_z_)
    weights_list.append(weights_)

    text_embeddings = []
    for i in range(K):
        for text_z in text_z_list:
            text_embeddings.append(text_z[i] if i < len(text_z) else text_z[0])
    text_embeddings = torch.stack(text_embeddings, dim=0)  # [B * K, 77, 768]

    weights = []
    for i in range(K):
        for weights_ in weights_list:
            weights.append(weights_[i] if i < len(weights_) else torch.zeros_like(weights_[0]))
    weights = torch.stack(weights, dim=0)
    return text_embeddings, weights


def get_pos_neg_text_embeddings(embeddings, azimuth_val, opt, iteration=0):
    if azimuth_val >= -90 and azimuth_val < 90:
        if azimuth_val >= 0:
            r = 1 - azimuth_val / 90
        else:
            r = 1 + azimuth_val / 90

        if random.random() < 0.7 and iteration < 5000:
            embs = delta.apply_side(emb_side, characterwise_mask_side, alpha=(1 - r), flag="side")
            embeddings['side'] = embs.tokenwise_embeddings["clip_encoder"].unsqueeze(0)

        start_z = embeddings['front']
        end_z = embeddings['side']
        pos_z = r * start_z + (1 - r) * end_z
        text_z = torch.cat([pos_z, embeddings['front'], embeddings['side']], dim=0)
        if r > 0.8:
            front_neg_w = 0.0
        else:
            front_neg_w = math.exp(-r * opt.front_decay_factor) * opt.negative_w
        if r < 0.2:
            side_neg_w = 0.0
        else:
            side_neg_w = math.exp(-(1 - r) * opt.side_decay_factor) * opt.negative_w

        weights = torch.tensor([1.0, front_neg_w, side_neg_w])
    else:

        if azimuth_val >= 0:
            r = 1 - (azimuth_val - 90) / 90
        else:
            r = 1 + (azimuth_val + 90) / 90
        if random.random() < 0.7 and iteration < 5000:
            embs1 = delta.apply_back(emb_back, characterwise_mask_back, alpha=2 * (1 - r), flag="back")
            embs2 = delta.apply_side(emb_side, characterwise_mask_side, alpha=r, flag="side")
            embeddings['back'] = embs1.tokenwise_embeddings["clip_encoder"].unsqueeze(0)
            embeddings['side'] = embs2.tokenwise_embeddings["clip_encoder"].unsqueeze(0)

        start_z = embeddings['side']
        end_z = embeddings['back']

        pos_z = r * start_z + (1 - r) * end_z
        text_z = torch.cat([pos_z, embeddings['side'], embeddings['front']], dim=0)
        front_neg_w = opt.negative_w
        if r > 0.8:
            side_neg_w = 0.0
        else:
            side_neg_w = math.exp(-r * opt.side_decay_factor) * opt.negative_w / 2

        weights = torch.tensor([1.0, side_neg_w, front_neg_w])
    return text_z, weights.to(text_z.device)


def prepare_embeddings(guidance_opt, guidance, vdm_prompts):
    embeddings = {}
    embeddings['default'] = guidance.get_text_embeds([guidance_opt.text])
    embeddings['uncond'] = guidance.get_text_embeds([guidance_opt.negative])
    embeddings['none'] = guidance.get_text_embeds("")

    for d in ['front', 'side', 'back']:
        embeddings[d] = guidance.get_text_embeds([f"{guidance_opt.text}, {d} view"])
        embeddings[f'prompt_{d}'] = guidance.get_text_embeds([f"{d} view of {vdm_prompts['prompt_target']}"])

    embeddings['prompt_target'] = guidance.get_text_embeds(vdm_prompts['prompt_target'])
    embeddings['inverse_text'] = guidance.get_text_embeds(guidance_opt.inverse_text)

    return embeddings


def guidance_setup(guidance_opt, vdm_prompts):
    if guidance_opt.guidance == "SD":
        guidance = StableDiffusion(guidance_opt.g_device, guidance_opt.fp16, guidance_opt.vram_O,
                                   guidance_opt.t_range, guidance_opt.max_t_range,
                                   num_train_timesteps=guidance_opt.num_train_timesteps,
                                   ddim_inv=guidance_opt.ddim_inv,
                                   textual_inversion_path=guidance_opt.textual_inversion_path,
                                   LoRA_path=guidance_opt.LoRA_path,
                                   guidance_opt=guidance_opt)
    else:
        raise ValueError(f'{guidance_opt.guidance} not supported.')
    if guidance is not None:
        for p in guidance.parameters():
            p.requires_grad = False
    embeddings = prepare_embeddings(guidance_opt, guidance, vdm_prompts)
    return guidance, embeddings


def prepare_output_and_logger(args):
    if not args._model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str = os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args._model_path = os.path.join("./output/", args.workspace)

    # Set up output folder
    print("Output folder: {}".format(args._model_path))
    os.makedirs(args._model_path, exist_ok=True)

    # copy configs
    if args.opt_path is not None:
        os.system(' '.join(['cp', args.opt_path, os.path.join(args._model_path, 'config.yaml')]))

    with open(os.path.join(args._model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args._model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


def training_report(tb_writer, iteration, elapsed, testing_iterations, scene: Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('iter_time', elapsed, iteration)
    # Report test and samples of training set
    if iteration in testing_iterations:
        save_folder = os.path.join(scene.args._model_path, "test_six_views/{}_iteration".format(iteration))
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
            print('test views is in :', save_folder)
        torch.cuda.empty_cache()

        config = ({'name': 'test', 'cameras': scene.getTestCameras()})
        if config['cameras'] and len(config['cameras']) > 0:
            for idx, viewpoint in enumerate(config['cameras']):
                render_out = renderFunc(viewpoint, scene.gaussians, *renderArgs, test=True)
                rgb, depth = render_out["render"], render_out["depth"]
                if depth is not None:
                    depth_norm = depth / depth.max()
                    save_image(depth_norm, os.path.join(save_folder, "render_depth_{}.png".format(viewpoint.uid)))

                image = torch.clamp(rgb, 0.0, 1.0)
                save_image(image, os.path.join(save_folder, "render_view_{}.png".format(viewpoint.uid)))
                if tb_writer:
                    tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.uid), image[None],
                                         global_step=iteration)
            print("\n[ITER {}] Eval Done!".format(iteration))
        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()


def video_inference(iteration, scene: Scene, renderFunc, renderArgs):
    sharp = T.RandomAdjustSharpness(3, p=1.0)

    save_folder = os.path.join(scene.args._model_path, "videos/{}_iteration".format(iteration))
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)  # makedirs
        print('videos is in :', save_folder)
    torch.cuda.empty_cache()
    config = ({'name': 'test', 'cameras': scene.getCircleVideoCameras()})
    if config['cameras'] and len(config['cameras']) > 0:
        img_frames = []
        depth_frames = []
        print("Generating Video using", len(config['cameras']), "different view points")
        for idx, viewpoint in enumerate(config['cameras']):
            render_out = renderFunc(viewpoint, scene.gaussians, *renderArgs, test=True)
            rgb, depth = render_out["render"], render_out["depth"]
            if depth is not None:
                depth_norm = depth / depth.max()
                depths = torch.clamp(depth_norm, 0.0, 1.0)
                depths = depths.detach().cpu().permute(1, 2, 0).numpy()
                depths = (depths * 255).round().astype('uint8')
                depth_frames.append(depths)

            image = torch.clamp(rgb, 0.0, 1.0)
            image = image.detach().cpu().permute(1, 2, 0).numpy()
            image = (image * 255).round().astype('uint8')
            img_frames.append(image)
        # Img to Numpy
        imageio.mimwrite(os.path.join(save_folder, "video_rgb_{}.mp4".format(iteration)), img_frames, fps=30, quality=8)
        if len(depth_frames) > 0:
            imageio.mimwrite(os.path.join(save_folder, "video_depth_{}.mp4".format(iteration)), depth_frames, fps=30,
                             quality=8)
        print("\n[ITER {}] Video Save Done!".format(iteration))
    torch.cuda.empty_cache()


def latent2image(vae, latents):
    latents = 1 / 0.18215 * latents
    image = vae.decode(latents)['sample']
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).numpy()
    image = (image * 255).astype(np.uint8)
    return image


def training(guidance, embeddings, tokenizer, dataset, opt, pipe, gcams, guidance_opt, testing_iterations,
             saving_iterations, checkpoint_iterations, checkpoint, debug_from, save_video, iteration_flag1,
             iteration_flag2):

    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gcams, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)
        print(f"use checkpoint, first_iter = {first_iter}")
    bg_color = [1, 1, 1] if dataset._white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device=dataset.data_device)
    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)
    save_folder = os.path.join(dataset._model_path, "train_process/")
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)  # makedirs
        print('train_process is in :', save_folder)
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    if opt.save_process:
        save_folder_proc = os.path.join(scene.args._model_path, "process_videos/")
        if not os.path.exists(save_folder_proc):
            os.makedirs(save_folder_proc)  # makedirs
        process_view_points = scene.getCircleVideoCameras(batch_size=opt.pro_frames_num,
                                                          render45=opt.pro_render_45).copy()
        save_process_iter = opt.iterations // len(process_view_points)
        pro_img_frames = []
    for iteration in range(first_iter, opt.iterations + 1):
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2,
                                                                                                               0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, guidance_opt.text)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None
        iter_start.record()
        gaussians.update_learning_rate(iteration)
        gaussians.update_feature_learning_rate(iteration)
        gaussians.update_rotation_learning_rate(iteration)
        gaussians.update_scaling_learning_rate(iteration)
        if iteration % 500 == 0:
            gaussians.oneupSHdegree()
        if not opt.use_progressive:
            if iteration >= opt.progressive_view_iter and iteration % opt.scale_up_cameras_iter == 0 and iteration > 1700:
                scene.pose_args.fovy_range[0] = max(scene.pose_args.max_fovy_range[0],
                                                    scene.pose_args.fovy_range[0] * opt.fovy_scale_up_factor[0])
                scene.pose_args.fovy_range[1] = min(scene.pose_args.max_fovy_range[1],
                                                    scene.pose_args.fovy_range[1] * opt.fovy_scale_up_factor[1])
                scene.pose_args.radius_range[1] = max(scene.pose_args.max_radius_range[1],
                                                      scene.pose_args.radius_range[1] * opt.scale_up_factor)
                scene.pose_args.radius_range[0] = max(scene.pose_args.max_radius_range[0],
                                                      scene.pose_args.radius_range[0] * opt.scale_up_factor)
                scene.pose_args.theta_range[1] = min(scene.pose_args.max_theta_range[1],
                                                     scene.pose_args.theta_range[1] * opt.phi_scale_up_factor)
                scene.pose_args.theta_range[0] = max(scene.pose_args.max_theta_range[0],
                                                     scene.pose_args.theta_range[0] * 1 / opt.phi_scale_up_factor)
                scene.pose_args.phi_range[0] = max(scene.pose_args.max_phi_range[0],
                                                   scene.pose_args.phi_range[0] * opt.phi_scale_up_factor)
                scene.pose_args.phi_range[1] = min(scene.pose_args.max_phi_range[1],
                                                   scene.pose_args.phi_range[1] * opt.phi_scale_up_factor)
        C_batch_size = guidance_opt.C_batch_size
        viewpoint_cams = []
        images = []
        text_z_ = []
        weights_ = []
        depths = []
        alphas = []
        scales = []
        text_z_inverse = torch.cat([embeddings['uncond'], embeddings['inverse_text']], dim=0)

        def get_apart_cameras(base_cam, other_cameras, num_cameras):
            base_azimuth = base_cam.delta_azimuth
            other_azimuths = np.array([cam.delta_azimuth for cam in other_cameras])

            def angular_distance(a, b):
                return np.abs(a - b)

            if base_azimuth < 0:
                base_azimuth += 360
            ideal_azimuths = [(base_azimuth + 5) % 360, (base_azimuth + 10) % 360, (base_azimuth + 15) % 360]
            ideal_azimuths = [(az - 360 if az > 180 else az) for az in ideal_azimuths]
            far_apart_indices = []
            for ideal_azimuth in ideal_azimuths:
                closest_idx = np.argmin(angular_distance(other_azimuths, ideal_azimuth))
                far_apart_indices.append(closest_idx)
                other_azimuths[closest_idx] = np.inf
            far_apart_cameras = [other_cameras[i] for i in far_apart_indices]
            return far_apart_cameras

        def rand_order_cameras(base_cam, other_cameras):
            def signed_azimuth_distance(azimuth):
                adjusted_azimuth = (azimuth + 180) % 360
                if 0 <= adjusted_azimuth < 180:
                    distance = adjusted_azimuth - 90
                else:
                    distance = 270 - adjusted_azimuth
                weight = calculate_weight(adjusted_azimuth)
                return distance * weight

            def calculate_weight(azimuth):
                min_diff = min(abs(azimuth - 135), abs(azimuth - 225), abs(azimuth - 45), abs(azimuth - 315))
                if min_diff > 90:
                    return 0.8
                else:
                    return 0.8 + 0.2 * (1 - min_diff / 90)

            base_azimuth = base_cam.delta_azimuth
            base_distance = signed_azimuth_distance(base_azimuth)
            camera_distances = []
            for cam in other_cameras:
                cam_azimuth = cam.delta_azimuth
                cam_distance = signed_azimuth_distance(cam_azimuth)
                diff_to_base = abs(cam_distance - base_distance)
                camera_distances.append((cam, diff_to_base))
            sorted_cameras = [cam for cam, _ in sorted(camera_distances, key=lambda x: x[1])]
            return sorted_cameras

        if iteration >= iteration_flag2:
            partial = 1
        else:
            partial = 0
        height = torch.max(gaussians.get_xyz[:, 2]) - torch.min(gaussians.get_xyz[:, 2])
        h = height.item()
        all_cameras = scene.getRandTrainCameras(height=h).copy()
        base_camera = all_cameras.pop(0)
        if partial == 1:
            selected_cameras = get_apart_cameras(base_camera, all_cameras, 3)
            viewpoint_stack = [base_camera] + selected_cameras
        else:
            pre_selected_cameras = [all_cameras.pop(randint(0, len(all_cameras) - 1)) for _ in range(3)]
            selected_cameras = rand_order_cameras(base_camera, pre_selected_cameras)
            viewpoint_stack = [base_camera] + selected_cameras
        black_video = False
        sh_deg_aug_ratio = dataset.sh_deg_aug_ratio
        test = False
        bg_aug_ratio = dataset.bg_aug_ratio
        if black_video:
            bg_color = torch.zeros_like(background)
        if random.random() < sh_deg_aug_ratio and not test:
            act_SH = 0
        else:
            act_SH = gaussians.active_sh_degree
        if random.random() < bg_aug_ratio and not test:
            if random.random() < 0.7:
                bg_color = torch.rand_like(background)
            else:
                bg_color = torch.zeros_like(background)
        else:
            bg_color = background
        scene.iteration = iteration
        _aslatent = False
        aslatent = 0
        opt.as_latent_ratio = 0.4
        if iteration < opt.geo_iter or random.random() < opt.as_latent_ratio:
            _aslatent = True
            aslatent = 1
        for i in range(C_batch_size):
            try:
                viewpoint_cam = viewpoint_stack.pop(0)
            except:
                all_cameras = scene.getRandTrainCameras().copy()
                base_camera = all_cameras.pop(0)
                if partial == 1:
                    selected_cameras = get_apart_cameras(base_camera, all_cameras, 3)
                    viewpoint_stack = [base_camera] + selected_cameras
                else:
                    pre_selected_cameras = [all_cameras.pop(randint(0, len(all_cameras) - 1)) for _ in range(3)]
                    selected_cameras = rand_order_cameras(base_camera, pre_selected_cameras)
                    viewpoint_stack = [base_camera] + selected_cameras
                viewpoint_cam = viewpoint_stack.pop(0)
            azimuth = viewpoint_cam.delta_azimuth
            file_path = os.path.join(dataset._model_path, "results.txt")
            with open(file_path, "a") as file:
                file.write(f"[{iteration}],    i={i},      viewpoint_cam.delta_azimuth={viewpoint_cam.delta_azimuth}\n")
            text_z = [embeddings['uncond']]
            if guidance_opt.perpneg:
                text_z_comp, weights = adjust_text_embeddings(embeddings, azimuth, guidance_opt, iteration=iteration)
                text_z.append(text_z_comp)
                weights_.append(weights)
            text_z = torch.cat(text_z, dim=0)
            text_z_.append(text_z)
            if (iteration - 1) == debug_from:
                pipe.debug = True
            render_pkg = render(viewpoint_cam, gaussians, pipe, bg_color,
                                sh_deg_aug_ratio=dataset.sh_deg_aug_ratio,
                                bg_aug_ratio=dataset.bg_aug_ratio,
                                shs_aug_ratio=dataset.shs_aug_ratio,
                                scale_aug_ratio=dataset.scale_aug_ratio,
                                act_SH=act_SH, iteration=iteration)
            current_scales = render_pkg["scales"]
            image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg[
                "viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
            depth, alpha = render_pkg["depth"], render_pkg["alpha"]
            scales.append(current_scales)
            images.append(image)
            depths.append(depth)
            alphas.append(alpha)
            viewpoint_cams.append(viewpoint_cam)
        images = torch.stack(images, dim=0)
        depths = torch.stack(depths, dim=0)
        alphas = torch.stack(alphas, dim=0)
        # Loss
        warm_up_rate = 1. - min(iteration / opt.warmup_iter, 1.)
        guidance_scale = guidance_opt.guidance_scale
        if guidance_opt.perpneg:
            loss = guidance.train_step_perpneg(torch.stack(text_z_, dim=1), images,
                                               pred_depth=depths, pred_alpha=alphas,
                                               grad_scale=guidance_opt.lambda_guidance,
                                               save_folder=save_folder,
                                               iteration=iteration,
                                               warm_up_rate=warm_up_rate,
                                               weights=torch.stack(weights_, dim=1),
                                               resolution=(gcams.image_h, gcams.image_w),
                                               guidance_opt=guidance_opt,
                                               as_latent=_aslatent,
                                               embedding_inverse=text_z_inverse,
                                               iteration_flag1=iteration_flag1,
                                               iteration_flag2=iteration_flag2,
                                               gaussians=gaussians)
        else:
            loss = guidance.train_step(torch.stack(text_z_, dim=1), images,
                                       pred_depth=depths, pred_alpha=alphas,
                                       grad_scale=guidance_opt.lambda_guidance, save_folder=save_folder,
                                       iteration=iteration,
                                       warm_up_rate=warm_up_rate,
                                       resolution=(gcams.image_h, gcams.image_w),
                                       guidance_opt=guidance_opt, as_latent=_aslatent, embedding_inverse=text_z_inverse)
        scales = torch.stack(scales, dim=0)
        loss_scale = torch.mean(scales, dim=-1).mean()
        loss_tv = tv_loss(images) + tv_loss(depths)
        loss = loss + opt.lambda_tv * loss_tv + opt.lambda_scale * loss_scale
        print(
            f"[ITER{iteration}]-ALL_Loss({loss.item()})=loss({loss.item()})+lambda_tv*loss_tv({opt.lambda_tv * loss_tv})+lambda_scale * loss_scale({opt.lambda_scale * loss_scale})")
        with open(file_path, "a") as file:
            file.write(
                f"ALL_Loss({loss.item()})=loss({loss.item()})+lambda_tv*loss_tv({opt.lambda_tv * loss_tv})+lambda_scale * loss_scale({opt.lambda_scale * loss_scale})\n")
        loss.backward()
        iter_end.record()
        with torch.no_grad():
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if opt.save_process:
                if iteration % save_process_iter == 0 and len(process_view_points) > 0:
                    viewpoint_cam_p = process_view_points.pop(0)
                    render_p = render(viewpoint_cam_p, gaussians, pipe, background, test=True)
                    img_p = torch.clamp(render_p["render"], 0.0, 1.0)
                    img_p = img_p.detach().cpu().permute(1, 2, 0).numpy()
                    img_p = (img_p * 255).round().astype('uint8')
                    pro_img_frames.append(img_p)
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()
            training_report(tb_writer, iteration, iter_start.elapsed_time(iter_end), testing_iterations, scene, render,
                            (pipe, background))
            if (iteration in testing_iterations):
                if save_video:
                    video_inference(iteration, scene, render, (pipe, background))
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
            if iteration < opt.densify_until_iter:
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter],
                                                                     radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                if iteration % opt.opacity_reset_interval == 0 and iteration != 1000:
                    gaussians.reset_opacity()
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)
            if iteration in args.save_iter:
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
    if opt.save_process:
        imageio.mimwrite(os.path.join(save_folder_proc, "video_rgb.mp4"), pro_img_frames, fps=30, quality=8)


if __name__ == "__main__":

    parser = ArgumentParser(description="Training parameters")
    parser.add_argument('--opt', type=str, default=None)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_ratio", type=int, default=5)
    parser.add_argument("--save_ratio", type=int, default=2)
    parser.add_argument("--save_video", type=bool, default=False)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int,
                        default=[1000, 1200, 1400, 1500, 1600, 1800, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 6000,
                                 7000])
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--iteration_flag1", type=int, default=2000)
    parser.add_argument("--iteration_flag2", type=int, default=4000)
    parser.add_argument("--device", type=str, default='cuda')
    parser.add_argument("--test_iter", nargs="+", type=int, default=[500, 1000, 1500, 2000, 3000, 3500, 4000, 4500, 5000, 5500, 5800, 6000, 6500, 7000])
    parser.add_argument("--save_iter", nargs="+", type=int, default=[300, 600, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 5800, 6000, 6500, 7000])


    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    gcp = GenerateCamParams(parser)
    gp = GuidanceParams(parser)

    args = parser.parse_args(sys.argv[1:])

    if args.opt is not None:
        with open(args.opt) as f:
            opts = yaml.load(f, Loader=yaml.FullLoader)
        lp.load_yaml(opts.get('ModelParams', None))
        op.load_yaml(opts.get('OptimizationParams', None))
        pp.load_yaml(opts.get('PipelineParams', None))
        gcp.load_yaml(opts.get('GenerateCamParams', None))
        gp.load_yaml(opts.get('GuidanceParams', None))
        lp.opt_path = args.opt
        args.port = opts['port']
        args.save_video = opts.get('save_video', True)
        args.seed = opts.get('seed', 0)
        args.device = opts.get('device', 'cuda')
        gp.g_device = args.device
        lp.data_device = args.device
        gcp.device = args.device

    args.test_iterations = args.test_iter
    args.save_iterations = args.save_iter

    print('Test iter:', args.test_iterations)
    print('Save iter:', args.save_iterations)
    print("Optimizing " + lp._model_path)

    prompts = [gp.text]
    LOW_RESOURCE = False
    guidance, embeddings = guidance_setup(gp, opts['vdm_prompts'])
    tokenizer = guidance.tokenizer

    import lpips

    lpips_alex = lpips.LPIPS(net='alex').to("cuda")

    CLIP_ENCODER = 'clip_encoder'
    deltas_view = []
    vdm_prompts = opts['vdm_prompts']
    target_token_embs = {}

    for direction in ['prompt_front', 'prompt_side', 'prompt_back', 'prompt_target']:
        token_spans = Token_spans(vdm_prompts[direction])
        emb = PromptEmbedding(vdm_prompts[direction],
                              tokenwise_embeddings={CLIP_ENCODER: embeddings[direction][0][..., :1024]},
                              tokenwise_embedding_spans=token_spans, )
        tokenwise_masks = emb.get_tokenwise_mask(get_mask_regex(emb.prompt, vdm_prompts['prompt_target']))
        delta_dict = emb.tokenwise_embeddings

        if direction == "prompt_front":
            tensor_1 = delta_dict['clip_encoder']

        if direction == "prompt_side":
            tensor_2 = delta_dict['clip_encoder']

        target_token_embs[direction] = {
            encoder: embedding[tokenwise_masks[encoder].index(True)]
            for encoder, embedding in emb.tokenwise_embeddings.items()
        }

    for prompt_name in ['prompt_front', 'prompt_side', 'prompt_back']:
        deltas_view.append({
            "prompt_name": prompt_name,
            **{
                encoder: target_token_embs[prompt_name][encoder] - (torch.dot(target_token_embs[prompt_name][encoder],
                                                                              target_token_embs['prompt_target'][
                                                                                  encoder]) / torch.norm(
                    target_token_embs['prompt_target'][encoder]) ** 2) * target_token_embs['prompt_target'][encoder]
                for encoder in emb.tokenwise_embeddings
            }
        })
    dims = {'clip_encoder': 1024}
    delta = EmbeddingDelta(dims)

    for i, d in enumerate(deltas_view):
        prompt_name = d["prompt_name"]
        for encoder in delta.tokenwise_delta_front:
            if prompt_name == "prompt_front":
                delta.tokenwise_delta_front[encoder] = d[encoder].clone()
            elif prompt_name == "prompt_side":
                delta.tokenwise_delta_side[encoder] = d[encoder].clone()
            elif prompt_name == "prompt_back":
                delta.tokenwise_delta_back[encoder] = d[encoder].clone()

    keyword = vdm_prompts['prompt_target']
    prompts_back = vdm_prompts['prompt_back']
    prompts_side = vdm_prompts['prompt_side']
    characterwise_mask_back = get_mask_regex(prompts_back, keyword)
    characterwise_mask_side = get_mask_regex(prompts_back, keyword)
    token_spans = Token_spans(prompts_back)

    emb_back = PromptEmbedding(
        prompt=prompts_back,
        tokenwise_embeddings={CLIP_ENCODER: embeddings["back"][0][..., :1024]},
        tokenwise_embedding_spans=token_spans,
    )
    emb_side = PromptEmbedding(
        prompt=prompts_side,
        tokenwise_embeddings={CLIP_ENCODER: embeddings["side"][0][..., :1024]},
        tokenwise_embedding_spans=token_spans,
    )

    clip_utils.load_vit(root=os.path.expanduser("~/.cache/clip"))
    embed = lambda ims: clip_utils.clip_model_vit(images_or_text=clip_utils.CLIP_NORMALIZE(ims), num_layers=-1)

    training(guidance, embeddings, tokenizer, lp, op, pp, gcp, gp, args.test_iterations, args.save_iterations,
             args.checkpoint_iterations, args.checkpoint, args.debug_from, args.save_video, args.iteration_flag1, args.iteration_flag2)
