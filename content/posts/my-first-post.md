+++
date = '2025-09-07T10:28:32+01:00'
draft = true
title = 'Formatting Demo: Code, Sidenotes, and Plots'
summary = 'Examples of code blocks, right-side sidenotes with [n] markers, and pretty plot embeds using reusable shortcodes.'
+++

This draft demonstrates code formatting, right-side comments with inline markers, and plot embedding.

Here is a sentence with an inline marker<span class="sidenote-ref">[1]</span> that also has a right-side note using our shortcode: {{< sidenote id="1" >}}This is a sidenote shown to the right on wide screens, and inline on mobile.{{< /sidenote >}}

## Code Blocks

```python
def pack_tokens(documents, max_len=4096):
    batch, cur = [], []
    for doc in documents:
        if len(doc) > max_len:
            continue  # skip or split long docs
        if sum(len(x) for x in cur) + len(doc) <= max_len:
            cur.append(doc)
        else:
            batch.append(cur)
            cur = [doc]
    if cur:
        batch.append(cur)
    return batch
```

Inline code looks like `vLLM` or `flash_attn_varlen`.

## Plot Embeds

{{< plot src="/images/plot-sample.png" caption="Placeholder plot caption" >}}

You can also full-bleed a wide chart:

{{< plot src="/images/plot-sample.png" caption="A wide figure" wide="true" >}}
