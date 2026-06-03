# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
import composer
import torch
import torch.nn.functional as F
from composer.utils import dist, reproducibility
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import CLIPTextModel

try:
    import xformers
    is_xformers_installed = True
except:
    is_xformers_installed = False

class StableDiffusion(composer.models.ComposerModel):

    def __init__(self, model_name: str = 'stabilityai/stable-diffusion-2-base'):
        super().__init__()
        self.unet = UNet2DConditionModel.from_pretrained(model_name, subfolder='unet')
        if is_xformers_installed:
            self.unet.enable_xformers_memory_efficient_attention()
        self.vae = AutoencoderKL.from_pretrained(model_name, subfolder='vae')
        self.text_encoder = CLIPTextModel.from_pretrained(model_name, subfolder='text_encoder')
        self.noise_scheduler = DDPMScheduler.from_pretrained(model_name, subfolder='scheduler')

        # Freeze vae and text_encoder when training
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)

    def forward(self, batch):
        images, captions = batch['image'], batch['caption']

        # Encode the images to the latent space.
        latents = self.vae.encode(images)['latent_dist'].sample().data
        # Magical scaling number (See https://github.com/huggingface/diffusers/issues/437#issuecomment-1241827515)
        latents *= 0.18215

        # Encode the text. Assumes that the text is already tokenized
        conditioning = self.text_encoder(captions)[0]  # Should be (batch_size, 77, 768)

        # Sample the diffusion timesteps
        timesteps = torch.randint(1, len(self.noise_scheduler), (latents.shape[0], ), device=latents.device)
        # Add noise to the inputs (forward diffusion)
        noise = torch.randn_like(latents)
        noised_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
        # Forward through the model
        return self.unet(noised_latents, timesteps, conditioning)['sample'], noise

    def loss(self, outputs, batch):
        return F.mse_loss(outputs[0], outputs[1])

    def get_metrics(self, is_train: bool):
        return None



model = StableDiffusion(model_name='stabilityai/stable-diffusion-2-base')

total_params = sum(
	param.numel() for param in model.parameters()
)

trainable_params = sum(
	p.numel() for p in model.parameters() if p.requires_grad
)

print(f'Model has {total_params/1e6} M parameters and {trainable_params/1e6} M trainable_params')


