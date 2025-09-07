+++
title = 'About'
date = '2025-09-07T00:00:00Z'
draft = false
+++

# Hi, I’m George

Currently exploring: pre-training LLM training efficiency, sample efficiency, building small coding LLMs. In addition, I work with finance LLM agents.

## Experience

I specialize in large-scale fine-tuning and evaluation of open-source LLMs (1B–405B parameters), with a focus on pushing the limits of training efficiency and model quality.

- Built distributed training platform for 60+ LLMs and with 600+ GPUs
- Built LLM-as-a-judge evaluation pipeline from scratch, orchestrated via Flyte
- Created custom sequence parallelism implementation using `flash_attn_varlen` for:
  - 131k context fine-tuning of LLaMA 3.1 70B (8×H100)
  - 16k context on LLaMA 3.1 405B (2‑node setup)
- Achieved +10% quality boost with document‑aware token packing

### Snap Inc.

Senior Machine Learning Engineer (Aug 2023 – Mar 2024, On‑site, Greater London)

- MyAI team (chatbot usage). Ran distributed training of multimodal LLMs (LLaVA) on 32+ GPUs and optimized LLM inference with vLLM.
- Led prompt iterations for intent classification that enabled multimodality features (image understanding and replies) and improved image captioning from baseline to production‑ready quality.

Machine Learning Engineer (May 2022 – Aug 2023, Hybrid)

- Led and developed “My New Twin” lens (on‑device image transformation), reaching very high engagement (350M+ DAU).
  - https://lens.snapchat.com/2897e0ee4e064728a6d259dca1a78f6c
- Ran distributed pre‑training of diffusion models on 128 GPUs; built a new AI Lenses pipeline focused on identity preservation in stylized images.
- Introduced a new lens experience based on diffusion models while preserving user attributes (gender, skin color, etc.); Anime AI lens showed strong virality and streamlined team production.

Keywords: StyleGAN2, StyleCLIP, fine‑tuning Stable Diffusion, image inversion, custom ControlNets, IP‑Adapter training, distillation to small models

## Publications

- Together AI — Together Evaluations: Benchmark Models for Your Tasks (Jul 28, 2025)  
  https://www.together.ai/blog/introducing-together-evaluations
- Together AI — Long Context Fine‑Tuning: A Technical Deep Dive (Nov 25, 2024)  
  https://www.together.ai/blog/long-context-fine-tuning-a-technical-deep-dive

### Patents (Snap)

- 2023‑11‑09 — Techniques for generating a stylized media content item with a generative neural network  
  https://patents.google.com/patent/US20240412433A1/
- 2023‑02‑03 — Face identity preservation for image‑to‑image models using stable diffusion generative model  
  https://patents.google.com/patent/US20240265498A1/

