<div align="center">
<img align="center" width="30%" alt="image" src="https://www.sandiego.edu/assets/global/images/logos/logo-usd.png">
</div>

# Image Restorator

![](https://img.shields.io/badge/license-MIT-green?style=for-the-badge)
![](https://img.shields.io/badge/MSAAI-CV-blue?style=for-the-badge)
![](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python)

AAI-521 Final Team Project

## Table of Contents
- [Description](#Overview)
- [Superres](#Superres)
- [Inpainting](#Inpainting)
- [Denoising](#Denoising)
- [Colorization](#Colorization)
- [Contributing](#Contributors)
- [License](#license)

#### Github Project Structure

```
├── README.md                             # Project documentation (this file)
├────── image-restoration-ai                # Project folder
├────────── configs                         # yaml file for runtime configuration
├────────── lora_inpainting                 # inpainting tensor weights
├────────── src                             # Models loading and inferencing modules
├───────────── pipelines                             # individual model pipelines (superres, colorization, inpainting, denoising)
├────────── swin2sr_div2k_finetuned_x4_1000steps # superres tensor weights
├────────── ui                              # streamlit ui (executable)
├────────── Improved_inpainting.ipynb       # fine-tuning inpainting jupyter notebook
├── requirements.txt                        # python libs used for the project
└── .gitignore                            # Ignored files for Git
```

# Overview
## About the Project

This project developed a unified computer vision restoration system that supports four core tasks: image denoising, super-resolution, colorization, and inpainting. The goal was to deliver a practical, user-facing toolkit that combines strong pretrained baselines with lightweight fine-tuning where feasible, then evaluate each task using appropriate metrics and controlled testing conditions. Rather than treating each task as an isolated demonstration, the project frames restoration as a realistic multi-step workflow. In practice, an image may require denoising first, localized repair next, and resolution enhancement last. This end-to-end perspective shaped both the backend architecture and the Streamlit-based interface, which provides consistent preprocessing, reproducible inference settings, and straightforward task switching across the four pipelines.

The Streamlit interface serves as applied validation of the complete system. Users can upload images, select tasks, adjust parameters, and preview results in real time. The inpainting UI provides interactive mask creation with consistent preview sizing, brush controls, and task-aware layout. This deployment bridges offline experimentation with a realistic end-user workflow and demonstrates that the system extends beyond notebooks into a functional restoration tool.

# Superres

The super-resolution pipeline uses the 4× real-world Swin2SR checkpoint caidas/swin2SR-realworld-sr-x4-64-bsrgan-psnr. This model is a ~12M parameter Swin Transformer v2–based architecture designed to handle practical real-world degradations. Swin2SR follows an encoder-decoder structure with shallow convolutional feature extraction, multiple SwinV2 transformer blocks (local window self-attention), and an upsampling head that outputs a 4× higher-resolution RGB image. 

Starting from the pretrained checkpoint, we performed compact fine-tuning on random DIV2K patches using L1 reconstruction loss between predicted and ground-truth HR patches. The intent was to reduce overly smooth outputs on very blurry or out-of-distribution images.

<img width="30%" alt="image" src="https://github.com/IanRebmann/Computer_Vision_Project/blob/main/superres.png">

# Inpainting

The inpainting component aimed to restore missing or damaged regions in a visually natural way. Practically, the model should fill masks without visible seams, texture discontinuities, or obvious hallucinations such as random text or watermark-like artifacts. In addition to perceptual quality, we evaluated similarity to the original images using objective metrics. The system uses Stable Diffusion v1.5 Inpainting via Hugging Face Diffusers.

To test parameter-efficient improvement, LoRA adapters were inserted into key attention projections (to_q, to_k, to_v, to_out.0). The base UNet, VAE, and text encoder remained frozen. Core training settings were: 512×512 images, batch size 2, ~2000 steps, learning rate 1e-4, rank 8 with alpha 8, gradient clipping 1.0, and fp16 mixed precision.

<img width="30%" alt="image" src="https://github.com/IanRebmann/Computer_Vision_Project/blob/main/inpainting.png">

# Denoising

The denoising component was implemented using a Stable Diffusion Img2Img baseline with runwayml/stable-diffusion-v1-5. This approach treats denoising as controlled generative refinement rather than direct pixel regression. The main inference controls—strength, guidance scale, inference steps, and optional seed—enable systematic experimentation and help balance noise removal against the risk of over-generation.

# Colorization

The colorization pipeline uses a Hugging Face pretrained model with a ResNet encoder and UNet decoder. Images are converted from RGB/BGR into grayscale inputs with pixel values scaled to 0–1. The model performs colorization in the Lab color space, using the L (lightness) channel as input and predicting the a (green–red) and b (yellow–blue) channels. This framework reduces prediction complexity by requiring two chrominance channels instead of three RGB channels. The predicted a and b channels are returned in a normalized range (0–1), requiring reconstruction into valid Lab channel values before conversion back to RGB/BGR. We used the author’s normalization utility while implementing the remaining preprocessing and Lab→RGB conversion using OpenCV rather than Kornia.

<img width="30%" alt="image" src="https://github.com/IanRebmann/Computer_Vision_Project/blob/main/colorized.png">

## Contributors
<table>
  <tr>
    <td>
        <a href="https://github.com/IanRebmann">
          <img src="https://github.com/IanRebmann.png" width="100" height="100" alt="Ian Rebmann"/><br />
          <sub><b>Ian Rebmann</b></sub>
        </a>
    </td>
    <td>
        <a href="https://github.com/carlosOrtizM">
          <img src="https://github.com/carlosOrtizM.png" width="100" height="100" alt="Carlos Ortiz"/><br />
          <sub><b>Carlos Ortiz</b></sub>
        </a>
    </td>
    <td>
        <a href="https://github.com/SyedMSirajuddin">
          <img src="https://github.com/SyedMSirajuddin.png" width="100" height="100" alt="Syed Sirajuddin"/><br />
          <sub><b>Syed Sirajuddin</b></sub>
        </a>
    </td>
    <td>
        <a href="https://github.com/gHOSTSINGHAH">
          <img src="https://github.com/gHOSTSINGHAH.png" width="100" height="100" alt="Tadhbir Singh"/><br />
          <sub><b>Tadhbir Singh</b></sub>
        </a>
    </td>
  </tr>
</table>

## License

MIT License

Copyright (c) [2025]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
