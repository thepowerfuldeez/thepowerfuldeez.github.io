+++
title = 'Lessons from Learning a Tokenizer from Scratch, and SuperBPE'
date = '2025-09-07T00:00:00Z'
draft = false
tags = ['tokenization', 'bpe', 'llm']
summary = 'Building a BPE tokenizer from scratch, measuring quality and sample efficiency, and exploring a more capable SuperBPE variant.'
+++

In this blogpost I will share my experience with building a BPE tokenizer from scratch, analyzing its quality, sample efficiency, and efficient implementation. This work was started as part of the excellent CS336 “LLMs from Scratch” course, but then quickly developed on its own due to my curiosity to explore things further.

## 1. Introduction

- Why tokenization still matters for LLMs
- Trade‑offs: vocabulary size, sequence length, and compute
- Datasets, cleaning, and special tokens

## 2. Training

- From unigram counts to merges: BPE refresher
- Efficient statistics collection and merge updates
- Sample efficiency experiments and ablations
- Practical considerations: Unicode handling, normalization, and stability

## 3. Inference

- Fast encode/decode paths and caching
- Handling unknowns and fallback strategies
- Compatibility and deterministic behavior

## 4. SuperBPE

- Motivation for going beyond vanilla BPE
- Richer merge rules and multi‑segment merges
- Preliminary results and future directions

