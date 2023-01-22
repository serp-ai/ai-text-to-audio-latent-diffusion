#!/usr/bin/env python3
import sys
import os
import itertools
from tqdm.auto import tqdm
from typing import List

from prefigure.prefigure import get_all_args
from copy import deepcopy
import math

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils import data

from diffusers.optimization import get_scheduler

from dataset.dataset import SampleDataset

from audio_diffusion.models import UNetModel
from audio_diffusion.utils import ema_update

from encodec import EncodecModel
from encodec.utils import save_audio

from transformers import T5Tokenizer, T5EncoderModel

from diffusers.schedulers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
    DDPMScheduler
)


logger = get_logger(__name__)


class FrozenT5Embedder(nn.Module):
    """Uses the T5 transformer encoder for text
    
    Code from: https://github.com/justinpinkney/stable-diffusion/blob/main/ldm/modules/encoders/modules.py"""
    def __init__(self, version="google/t5-v1_1-large", device="cuda", max_length=77):  # others are google/t5-v1_1-xl and google/t5-v1_1-xxl
        super().__init__()
        self.tokenizer = T5Tokenizer.from_pretrained(version)
        self.transformer = T5EncoderModel.from_pretrained(version)
        self.device = device
        self.max_length = max_length

    def freeze(self):
        self.transformer = self.transformer.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text):
        batch_encoding = self.tokenizer(text, truncation=True, max_length=self.max_length, return_length=True,
                                        return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
        tokens = batch_encoding["input_ids"].to(self.device)
        outputs = self.transformer(input_ids=tokens)

        z = outputs.last_hidden_state
        return z

    def encode(self, text):
        return self(text)


# Denoising loop
def sample(unet, codes, embedder, scheduler, device, num_inference_steps=50, batch_size=1, prompt='kawaii, future bass, edm', negative_prompt=None, do_classifier_free_guidance=True, guidance_scale=7, eta=0.0):
    """Code adapted from: https://github.com/huggingface/diffusers/blob/debc74f442dc74210528eb6d8a4d1f7f27fa18c3/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py"""
    # prepare timesteps
    scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = scheduler.timesteps
    
    text_embeddings = embedder(prompt)
    # get unconditional embeddings for classifier free guidance
    if do_classifier_free_guidance:
        uncond_tokens: List[str]
        if negative_prompt is None:
            uncond_tokens = [""] * batch_size
        elif type(prompt) is not type(negative_prompt):
            raise TypeError(
                f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                f" {type(prompt)}."
            )
        elif isinstance(negative_prompt, str):
            uncond_tokens = [negative_prompt]
        elif batch_size != len(negative_prompt):
            raise ValueError(
                f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                " the batch size of `prompt`."
            )
        else:
            uncond_tokens = negative_prompt

        uncond_embeddings = embedder(uncond_tokens)
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

    ts = codes.new_ones([codes.shape[0]])

    with tqdm(total=num_inference_steps):
        for i, t in enumerate(timesteps):
            # expand the codes if we are doing classifier free guidance
            code_model_input = torch.cat([codes] * 2) if do_classifier_free_guidance else codes
            code_model_input = scheduler.scale_model_input(code_model_input, t)

            # predict the noise residual
            noise_pred = unet(code_model_input, timesteps=t*ts, context=text_embeddings)

            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            codes = scheduler.step(noise_pred, t, codes).prev_sample
    return codes


def target_bandwidth_to_channels(target_bandwidth):
    """Maps target bandwidth to number of channels"""
    if target_bandwidth == 1.5:
        return 2
    elif target_bandwidth == 3.0:
        return 4
    elif target_bandwidth == 6.0:
        return 8
    elif target_bandwidth == 12.0:
        return 16
    elif target_bandwidth == 24.0:
        return 32
    else:
        raise ValueError(f"Invalid target bandwidth: {target_bandwidth}")


class DiffusionUncond(nn.Module):
    def __init__(self, global_args):
        super().__init__()
        self.unet = UNetModel(
            in_channels=global_args.channels,    # depends on target bandwidth
            out_channels=global_args.channels,   # depends on target bandwidth
            sample_size=4096,                    # size of latent codes (change depending on length of audio)
            model_channels=320,                  # hyperparameter to tune
            attention_resolutions=[4, 2, 1],     # hyperparameter to tune
            num_res_blocks=6,                    # hyperparameter to tune
            channel_mult=[ 1, 2, 4, 4 ],         # hyperparameter to tune
            use_audio_transformer=True,          # hyperparameter to tune
            use_linear_in_transformer=True,      # hyperparameter to tune
            transformer_depth=1,                 # hyperparameter to tune
            num_head_channels=64,                # hyperparameter to tune
            dropout=0.0,                         # hyperparameter to tune
            use_checkpoint=True if global_args.gradient_checkpointing not in [None, False, 'false', '', 'False'] else False,
            context_dim=1024,
            legacy=False,
            dims=1
        )
        self.unet_ema = deepcopy(self.unet)
        self.encodec = EncodecModel.encodec_model_24khz()
        self.encodec.set_target_bandwidth(global_args.target_bandwidth)


def main():
    """Code adapted from: https://github.com/Harmonai-org/sample-generator/blob/main/train_uncond.py"""
    args = get_all_args()

    save_path = None if args.save_path == "" else args.save_path

    args.channels = target_bandwidth_to_channels(args.target_bandwidth)

    print(f'Using {args.channels} channels for target bandwidth {args.target_bandwidth}')

    accelerator = Accelerator(
        gradient_accumulation_steps=args.accum_batches,
        mixed_precision=args.precision,
        log_with="tensorboard",
        logging_dir=save_path,
    )

    if args.scale_lr:
        args.lr = (
            args.lr * args.accum_batches * args.batch_size * accelerator.num_gpus
        )

    # taken from stable diffusion v-prediction model defaults, not sure if the most optimal, tune as needed
    scheduler_config = {
            "beta_end": 0.012,
            "beta_schedule": "scaled_linear",
            "beta_start": 0.00085,
            "clip_sample": False,
            "num_train_timesteps": 1000,
            "prediction_type": "v_prediction",
            "trained_betas": None
        }

    model = DiffusionUncond(args)
    if args.ckpt_path.lower() not in [None, '', 'none', 'false']:
        print(f'Loading checkpoint from {args.ckpt_path}')
        model.unet_ema.load_state_dict(torch.load(args.ckpt_path))
        model.unet.load_state_dict(torch.load(args.ckpt_path.replace('-ema', '')))
        print('Loaded checkpoint')

    if args.use_embedder:
        # Load the tokenizer
        if args.embedder_path not in ['none', 'false', '']:
            embedder = FrozenT5Embedder(version=args.embedder_path, device="cuda", max_length=77)
        else:
            embedder = FrozenT5Embedder(version="google/t5-v1_1-large", device="cuda", max_length=77)
        if args.train_text_encoder in ['none', 'false', '', None, False]:
            embedder.freeze()
        else:
            print('Training text encoder')
    else:
        embedder = None

    train_set = SampleDataset([args.training_dir], args)
    train_dl = data.DataLoader(train_set, args.batch_size, shuffle=True,
                               num_workers=args.num_workers, persistent_workers=True, pin_memory=True, drop_last=True)
    
    # Load models
    unet = model.unet
    unet_ema = model.unet_ema
    unet_ema.requires_grad_(False)
    encodec = model.encodec
    encodec.to(accelerator.device)
    encodec.requires_grad_(False)

    if args.gradient_checkpointing not in [None, '', 'none', 'false', False]:
        print('Enabling gradient checkpointing')
        if args.train_text_encoder not in ['none', 'false', '', None, False]:
            embedder.gradient_checkpointing_enable()

    # Use 8-bit Adam for lower memory usage
    if args.use_8bit_optim:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

        optimizer_class = bnb.optim.AdamW8bit
        print('using 8-bit Adam')
    else:
        optimizer_class = torch.optim.AdamW

    params_to_optimize = (
        unet.parameters()
    )
    optimizer = optimizer_class(
        params_to_optimize,
        lr=args.lr,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    if args.max_train_steps == 0 or args.max_train_steps == '':
        args.max_train_steps = None
    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dl) / args.accum_batches)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.accum_batches,
        num_training_steps=args.max_train_steps * args.accum_batches,
    )

    if not args.use_embedder or (args.train_text_encoder in ['none', 'false', ''] and args.use_embedder):
        unet, optimizer, train_dl, lr_scheduler = accelerator.prepare(
                unet, optimizer, train_dl, lr_scheduler
            )
    else:
        unet, embedder, optimizer, train_dl, lr_scheduler = accelerator.prepare(
                unet, embedder, optimizer, train_dl, lr_scheduler
            )
    accelerator.register_for_checkpointing(lr_scheduler)

    weight_dtype = torch.float32
    if args.precision == "fp16":
        weight_dtype = torch.float16
    elif args.precision == "bf16":
        weight_dtype = torch.bfloat16

    encodec.to(accelerator.device, dtype=torch.float32)
    if args.train_text_encoder in ['none', 'false', ''] and args.use_embedder:
        embedder.to(accelerator.device, dtype=weight_dtype)

    set_seed(args.seed)
    noise_scheduler = DDPMScheduler(**scheduler_config)

    scheduler_config["set_alpha_to_one"] = False
    scheduler_config["steps_offset"] = 1

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dl) / args.accum_batches)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("latentdancebooth", config=vars(args))

    # Train!
    total_batch_size = args.batch_size * accelerator.num_processes * args.accum_batches

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_set)}")
    logger.info(f"  Num batches each epoch = {len(train_dl)}")
    logger.info(f"  Num Epochs = {args.num_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.accum_batches}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0
    last_demo_step = -1

    if args.resume_from_checkpoint.lower() not in ['', 'none', 'false']:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the mos recent checkpoint
            dirs = os.listdir(args.save_path)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1]
        accelerator.print(f"Resuming from checkpoint {path}")
        accelerator.load_state(os.path.join(args.save_path, path))
        global_step = int(path.split("-")[1])

        resume_global_step = global_step * args.accum_batches
        first_epoch = resume_global_step // num_update_steps_per_epoch
        resume_step = resume_global_step % num_update_steps_per_epoch

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(global_step, args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    for epoch in range(first_epoch, args.num_train_epochs):
        unet.train()
        if args.use_embedder and args.train_text_encoder not in ['none', 'false', '']:
            embedder.train()
        for step, batch in enumerate(train_dl):
            # Skip steps until we reach the resumed step
            if args.resume_from_checkpoint.lower() not in ['', 'none', 'false'] and epoch == first_epoch and step < resume_step:
                if step % args.accum_batches == 0:
                    progress_bar.update(1)
                continue

            with accelerator.accumulate(unet):
                # move audio to discrete codes
                encoded_frames = encodec.encode(batch[0])
        
                codes = torch.cat([encoded[0] for encoded in encoded_frames], dim=-1).to(accelerator.device, dtype=weight_dtype)

                # scale from 0 - 1023 to -1 to 1
                codes = (codes / 511.5) - 1
                codes = torch.clamp(codes, -1., 1.)

                # Sample noise that we'll add to the codes
                noise = torch.randn_like(codes)
                bsz = codes.shape[0]

                # Sample a random timestep for each audio sample
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=codes.device)
                timesteps = timesteps.long()

                # Add noise to the codes according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_codes = noise_scheduler.add_noise(codes, noise, timesteps)

                if args.use_embedder:
                    # Get the text conditioning
                    input_ids = embedder(batch[1])

                    noise_pred = unet(noisy_codes, timesteps=timesteps, context=input_ids)
                else:
                    noise_pred = unet(noisy_codes, timesteps=timesteps)

                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(codes, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                loss = F.mse_loss(noise_pred.float(), target.float(), reduction='mean')

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = (
                        itertools.chain(unet.parameters(), embedder.parameters())
                        if args.use_embedder and args.train_text_encoder not in ['none', 'false', ''] else
                        unet.parameters()
                    )
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer.step()
                ema_update(unet, unet_ema, args.ema_decay)
                lr_scheduler.step()
                optimizer.zero_grad()
            
            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if global_step % args.checkpoint_every == 0:
                    if accelerator.is_main_process:
                        save_path = os.path.join(args.save_path, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")
                        torch.save(unet_ema.state_dict(), os.path.join(args.save_path, f"_checkpoint-ema-{global_step}.pkl"))
                        torch.save(unet.state_dict(), os.path.join(args.save_path, f"_checkpoint-{global_step}.pkl"))
                        if args.train_text_encoder not in ['none', 'false', '']:
                            torch.save(embedder.state_dict(), os.path.join(args.save_path, f"_checkpoint-t5-{global_step}.pkl"))

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

            if (global_step - 1) % args.demo_every != 0 or last_demo_step == global_step:
                continue
            
            if accelerator.is_main_process:
                with torch.no_grad():
                    last_demo_step = global_step
            
                    noise = torch.randn([args.num_demos, args.channels, 4096]).to(accelerator.device, dtype=weight_dtype)

                    try:
                        if args.use_embedder:
                            fakes = sample(unet, noise, embedder, DDIMScheduler(**scheduler_config), accelerator.device, num_inference_steps=args.demo_steps)
                        else:
                            fakes = sample(unet, noise, None, DDIMScheduler(**scheduler_config), accelerator.device, num_inference_steps=args.demo_steps)

                        # scale from -1 to 1 to 0 - 1023 and discretize
                        fakes = ((fakes + 1) * 511.5).to(torch.long)
                        fakes = fakes.clamp(0, 1023)

                        # decode
                        decoded_frames = encodec.decode([(fakes, None)])

                        # save demos
                        filename = f'demo_{global_step:08}.wav'
                        for i, audio in enumerate(decoded_frames):
                            if i > 0:
                                save_audio(audio.cpu(), f"{filename[:-4]}_{i}.wav", encodec.sample_rate)
                            else:
                                save_audio(audio.cpu(), filename, encodec.sample_rate)
                    except Exception as e:
                        print(f'{type(e).__name__}: {e}', file=sys.stderr)

        accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        accelerator.save(unet.state_dict(), f"checkpoint-{global_step}.pkl")

if __name__ == '__main__':
    main()
