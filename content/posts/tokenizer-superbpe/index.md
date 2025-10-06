+++
title = 'Lessons from Building a Tokenizer from Scratch — and a Peek at SuperBPE'
date = '2025-10-05T00:00:00Z'
draft = true
tags = ['tokenization', 'bpe', 'llm']
summary = 'Byte-level BPE from first principles: what matters for speed and quality, how to implement it cleanly, and why a SuperBPE variant can lift sample efficiency.'
+++

In this blogpost I will share my experience with building a BPE tokenizer from scratch, analyzing its quality, sample efficiency, and efficient implementation. 

This work was started as part of the excellent CS336 “LLMs from Scratch” course, but then quickly developed into its own mini-project due to my curiosity to explore things further.

I now have shared all code in my [repo](https://github.com/thepowerfuldeez/sample_efficient_gpt)


## Intro

Tokenization is needed to convert your text into some numeric representation that is useful for training LLMs (and to essentially compress your data too, reducing sequence size)

Let's discuss general algorithm of BPE tokenization:

{{< figure src="/posts/tokenizer-superbpe/bpe_scheme.svg" caption="General scheme of BPE training" >}}

Each string character in UTF-8 encoding is represented as bytes (number from 0 to 255). We convert text to bytes. This is our initial set of tokens.

At each step:
	1.	Count adjacent token-pair frequencies.
	2.	Merge the most frequent pair into a new token id (sequence becomes shorter).
	3.	Repeat until you hit your target vocab size.

Larger token id -> later it was created from merge.

Byte Level BPE algorithm accepts bytes and returns sequence of ints (this can be saved as *uint16* if your vocab size is less than 65536), and no input will be left un-tokenized (in contrast to earlier tokenizers which used `<UNK>` token for unknown tokens that don't fit into vocab)

{{< figure src="/posts/tokenizer-superbpe/bpe_train_step.svg" caption="Single training step of BPE algorithm" >}}

Here we make a frequency count for each pair of numbers, select max pair by this count and replace it with a new token id.

There are few nuances in the implementation.

#### Not all bytes are created equal

Since byte is just a number from 0 to 255, but we have much more characters than 256, we represent it with variable amount of bytes. An average English text would be 1.25 bytes / character.

- In UTF-8, each code point can take 1–4 bytes, depending on the character.
    - ASCII (U+0000–U+007F) → 1 byte
    - Latin, Cyrillic, etc. → 2 bytes
    - Most CJK (Chinese/Japanese/Korean) → 3 bytes
    - Rare supplementary (emoji, historic scripts) → 4 bytes

### Pre-tokenization

It's easy to show that naive algorithm works to the order of $O(N^2)$

For the large files just a single iteration becomes very computationally expensive (for 4e9 symbols and 5GB file, $N$ would be 5e9 and single step would take 100+ days on modern CPUs!)

One optimization that we could use is to split whole training corpora into words (split by token id 32 = " " in the example, or more generally - by any whitespace character) and group words together via hashmap. 

Then, knowing a count of pre-token, true count for a pair of numbers becomes `frequency of pair inside a word * pre token count`

It's even better to split not just by delimiter, but by a more complicated regex. For example, this is the regex used when training GPT-2:

```python
'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
```

Nowadays, people use better regex to handle code, long numbers or delimiters more smarter.

#### What's actually SOTA in regex patterns

•	Digits: triad chunking \p{N}{1,3} is standard in modern GPT tokenizers; plus support for thousands separators (, and _), decimals, scientific, percent—these help BPE learn reusable numeric chunks without exploding vocab.
•	Contractions: handle both ASCII ' and curly ’, case-insensitive, so "It’s", "IT’S", "it’s" → consistent splits.
•	Newlines: explicit rules for \r\n/\n reduce ambiguity and improve text layout modeling.
•	Punctuation runs: keep [^\s\p{L}\p{N}]+ to group operators/emoji/punct as learned units. (If you later want emoji ZWJ sequences as single tokens, you’ll need a non-regex grapheme/emoji segmenter; regex alone won’t cleanly capture all emoji families.)

In addition, when it comes to data preprocessing, there's also a notion of special tokens, that we do not split into subwords, or into bytes, ie they are not involved in the training. One of the most popular special token examples - end of text. It is important to keep it as it is as we use it to split documents with each other, and tell the LLM where the sequence ends, so that at inference we would have an LLM that can actually stop it's generation :)


Generally, now whole training pipeline looks like Pretokenization + N iterations of merges. Pre-tokenization can be parallelized and re-written to rust for max speedup. I used pyo3, simdutf, ahash for efficient splitting and counting.


### Nuances

There are still a couple of questions related to merging process:
1. How to detect most frequent pair efficiently? Do we need some specific data structure?
2. How to most efficiently merge 2 tokens together and update the list in-place?
3. Is there a difference between encoding during training and during inference?

Let me answer all of those questions simultaneously. I will share data structure, some hacks, and inference implementation. That might not be the best algorithm overall, but this is what I found worked for me and hopefully this can give you some food for thought.

First of all, for training our main goal is to be able to find the most frequent pair really fast, then merging each key would be easy. 

Let's cache pre_token_counts, updated_pre_tokens, all_pair_counts, pair_to_pre_tokens.

all_pair_counts is the most important map. We take top1 count each step to get max pair.

Two observations:

. So:
	•	Subtract counts for pairs destroyed by the merge (e.g., (2,3) → zero).
	•	Adjust counts for pairs that touch the merged span (e.g., (1,2)→(1,99), (3,4)→(99,4)).
	•	Leave everything else alone.

1. After each update, we don't need to re-compute all_pair_counts. We know that most pairs are untouched, so we keep only pre-tokens that have been updated (updated_pre_tokens). Moreover, we know that frequencies of pairs change in determenistic order. So, if we had a sequence (1, 2, 3, 4) and merged (2, 3) -> 99:
	- Subtract counts for pairs destroyed by the merge.
	- Adjust counts for pairs that touch the merged span (e.g., (1,2)→(1,99), (3,4)→(99,4)).
	- Leave everything else alone. 
Including this modifications let us cache all_pair_counts and update it after each iteration instead of re-computing.

This change reduced my total training time 7x.

For example, if our max pair is (a, b) and some pre-token was (c, a, b, d), then (c, a), (a, b), (b, d) are affected, where (a, b) becomes 0.

2. We know there are situations where topk pairs from all_pair_counts remain the same (so top2 becomes top1 in the next step, and if we know that in advance, we could retrieve next max pair in O(1)). Intuitively, this can be if our updated frequences in all_pair_counts do not touch any of the topk keys, or at least do not drive them too high. This is a pretty weak assumption but it helps to reduce search space and not re-sort keys of all_pair_counts every iteration.


Natural data structure for all_pair_counts is MaxHeap. In O(logN) we keep it sorted and take max element in a constant time. When I tried it, I noticed that we add / remove keys too frequently, so that the overhead of MaxHeap becomes too high. I keep it as a dict and try to sort elements when needed.

#### Inference

Now we don't need to find the most frequent token anymore, and all of our merges are fixed.

I ended up using Doubly Linked List for the sequence of tokens for each word.

After we found the most frequent pair, our goal is to create a chain of tokens, each knowing which token comes next and behind. We find required pair, replace first element to updated token id, unlink second element, re-link updated token to the next.next. Finally Leetcode is useful!!! 


## 4. SuperBPE

SuperBPE is a farly recent work, and in my opinion underexplored. This was my initial idea when I implemented tokenizer from scratch. Doing pre-tokens seems like a useful optimization hack, but doesn't make tokenizers more generalized. I could easily imagine cases where frequences of several words together will be higher than any frequency of two tokens next to each other. Language is redundant and contains phrases like "such as", "to be considered", "for example" which is logical to represent as a single token.

In order to implement SuperBPE, the only thing we need to change is our "pre_token_counts" representation. Instead of words going to each key, we could put whole document there. It will be huge dict, but algorithm would work.

I tested it and always got OOM, or my training will never finish (actually it's the same problem as we started this blogpost with). So I restricted pre-tokens to have max length of 10 words, split by any punctuation and apply few other heuristics + reduced dataset size. Algorithm is still like 10x slower than BPE, but hopefully we don't need to start from the beginning, and we could enable SuperBPE only at last 20% of training (this works better in practice, since we apply some bias at first to lear inter-word n-gram dependencies, and then lift this restriction)


## Experiments

The most interesting part.

I have used my baseline implementation of LLM training with handwritten triton flash attention 2 and some tweaks.

I used improved regex template and prepared "basic BPE tokenizer" and "SuperBPE" tokenizer both trained on 32700 vocab size (and I am using model with vocab size 32768 to be utilized on GPU better)

SuperBPE is trained by resuming basic BPE from 26000 iteration (roughly 80% of training).

Dataset used: DCLM (edu subset, filtered with score > 2.75)
Vocab size: 32700

I then trained the 100M parameter LLM on ~1.2B tokens (24k iterations, 512 context length, 96 batch size, single GPU)

After SuperBPE, dataset has 20% less tokens, so training is 20% more sample efficient!

{{< figure src="/posts/tokenizer-superbpe/superbpe_first_run.jpeg" caption="First run" >}}

After running first experiment, I noticed significantly higher loss. Then I checked outputs, it looked great, subjectively even better than baseline. What's suspicious is that val loss is exactly 25% higher (3.19 -> 3.98). A-ha! Each token now covers ~25% more text (because 1/(1-0.2)=1.25). If the model’s per-byte compression didn’t change, the per-token NLL should rise by that same 1.25x factor.

So I pre-computed pet roken byte lengths tensor and using that to normalize loss value for logging.

```python
id2byte_len = torch.tensor([
    len(tokenizer.decode([i]).encode("utf-8"))
    for i in range(tokenizer.vocab_size)
], device="cuda")
```

The total sum of bytes is 220KB vs 212KB, slight increase, but of course it translates to sample efficiency differently.
