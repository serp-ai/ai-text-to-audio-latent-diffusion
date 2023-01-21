# README 🎁

# About

This repo is for text-to-audio diffusion utilizing a denoising unet and Meta's Encodec. The unet is trained to denoise Encodec's encoded codebooks while taking in t5 text embeddings as conditioning. Encodec's decoder can then take the denoised codebooks, and decode it to the uncompressed .wav file.

The architecture is by no means perfect as it is being actively tested/worked on. If you have any suggestions for improvements to try please don't hesistate to let us know!

# Instructions

- Clone the repo
- Set up your environment
- Launch the `train_latent_cond.py` file with accelerate (`example_launch_command.txt` in root directory for an example)
- `training_args.md` in root directory for argument explanations
- Inferencing scripts/notebooks/trained models coming soon

# Shout Outs

- Thanks to [Hugging Face](https://huggingface.co/) for diffusers/transformers and being a huge contribution to the open source community
- Thanks to [HarmonAI](https://www.harmonai.org/) for their audio diffusion research and contributions to the open source community
- Thanks to [Stable Diffusion](https://stability.ai/) and OpenAI for the unet/cross-attention base code and for their open source contributions
- Thanks to [Meta](https://github.com/facebookresearch/encodec) for open sourcing Encodec and all of their other open source contributions
- Thanks to [Google](https://github.com/google-research/text-to-text-transfer-transformer) for open sourcing the t5 large language model.
- Shoutout to [EveryDream](https://github.com/victorchall/EveryDream2trainer) for windows venv setup and bnb patch
