# Pool CLI

CLI that sorts your screenshots into **Pools** and suggests what to do with them.

## Quick start

```bash
pip install -e .
export OPENAI_API_KEY="sk-..."
pool ~/Pictures/Screenshots
```

First run will download the SigLIP2 model (~1.5 GB). After that it's cached locally.

## How it works

```
Screenshots
    │
    ▼
  Dedup (SHA1, keep canonical per hash)
    │
    ▼
  Embed (SigLIP2, each image → 768-d vector)
    │
    ▼
  Classify (image embeddings @ pool text embeddings = similarity matrix)
    │
    ├── top score & margin to 2nd pool above thresholds?
    │       yes → assign to Pool (confidence = f(score, margin))
    │       no  ↓
    │
  "Other" residuals
    │
    ▼
  Cluster (HDBSCAN on embeddings)
    │── filter: min cluster size
    │── filter: cohesion (mean pairwise similarity)
    │
    ▼
  Name each cluster (GPT-4.1-mini, from sample screenshots)
    │
    ▼
  Seed (assign cluster members to new pool)
    │── filter: similarity to cluster centroid
    │
    ▼
  Expand (SigLIP2 re-scores remaining "Other" against new descriptions)
    │── filter: prefilter similarity to centroid
    │── filter: score & margin thresholds
    │
    ▼
  User Pools + Predefined Pools
    │
    ▼
  Suggest actions (GPT-4.1-mini, per pool, from top matches)
    │
    ▼
  Report
```

**Embed & Classify** run locally via [SigLIP2](https://huggingface.co/google/siglip2-base-patch16-naflex). Classification is a matrix multiply of image embeddings against pool text embeddings, so it's fast even on 9k+ images.

**User pool discovery** (when `OPENAI_API_KEY` is set) takes the "Other" residuals, clusters them with HDBSCAN, and uses GPT-4.1-mini to name each cluster. New pools are then expanded by re-scoring "Other" images against the generated descriptions. If the API key is missing, this stage is skipped and residuals stay in `Other`.

**Action suggestions.** For each pool, a sample of top matches is sent to GPT-4.1-mini, which returns a concrete action ("Send to Spotify playlist", "Map places by city", etc.).

OpenAI calls scale with the number of pools, not images. 900 or 9,000 screenshots, same API cost. Everything is cached in SQLite (`~/.pool/cache/`), re-runs are instant.

## Evaluation

### Test set construction: automated screenshotting + public datasets

The test set contains **886 screenshots** (801 unique after SHA1 deduplication) across 4 predefined categories and a user-pool category. Data was collected in two ways: automated screenshotting of real websites via Playwright (mobile viewport, emulating how a user would see it on a phone), and filtering existing public datasets.

| Category | Count | Source | Method |
|---|---|---|---|
| Music | 117 | [AMEX-8k](https://huggingface.co/datasets/zonghanHZH/AMEX-8k) | Filtered Spotify/YouTube Music screenshots by app name + content keywords |
| Places | 162 | Google Maps, Apple Maps | Automated screenshots of place cards for 90+ real locations across 10 cities |
| Products | 237 | Farfetch, [AMEX-8k](https://huggingface.co/datasets/zonghanHZH/AMEX-8k) | Automated product page screenshots + filtered shopping app screenshots (SHEIN, IKEA, eBay, etc.) |
| Recipes | 300 | [VDR Cooking Recipes](https://huggingface.co/datasets/racineai/VDR_Cooking_Recipes) | English recipe screenshots with metadata |
| Other | 6 | Manual | Intentional non-matches (settings screens, generic UI) |
| User pools | ~64 | ESPN, BBC Sport, etc. | Automated screenshots of sports scoreboards with varied viewports and scroll offsets |

The full test set is published at [poteminr/pool](https://huggingface.co/datasets/poteminr/pool) on Hugging Face.

### Key metric: precision over recall

A wrong image in a pool hurts more than a missing one. Garbage breaks trust, a small gap doesn't. So we optimize for **precision** first, and use **F0.5** (precision weighted 2x over recall) as the aggregate score.

### Results (822 scored images)

| Pool | Size | Precision | Recall | Recall (fraction) | F0.5 |
|---|---|---|---|---|---|
| Music | 117 | 1.00 | 1.00 | 117 / 117 | 1.00 |
| Places | 151 | 1.00 | 0.93 | 151 / 162 | 0.99 |
| Products | 229 | 1.00 | 0.97 | 229 / 237 | 0.99 |
| Recipes | 299 | 1.00 | 1.00 | 299 / 300 | 1.00 |

**Accuracy: 97.6%** | **Macro F0.5: 0.80** | **Coverage: 98.3%**

All four predefined pools hit **100% precision**: no garbage at all. Recall is 93-100%, so very little is missed.

## What I'd improve with more time

I think the most promising direction is pushing the local model further. It's faster, cheaper, and scales naturally to large libraries. Predefined pools already work well with zero-shot SigLIP2. The real bottleneck is user pool discovery, where the pipeline currently relies on the OpenAI API for naming and expansion. Finding ways to do more of that locally would make the whole system more self-contained.

**Per-pool adaptive thresholds.** Pools separate differently (Music is trivial, Places/Products overlap). Thresholds should be per-pool, calibrated by LLM-judging borderline samples.

**Prototype centroids.** Replace hand-written text prompts with mean embeddings of confirmed members. Metric learning without training.

**Active learning loop.** First pass with text prompts, LLM verifies borderlines, confirmed examples become centroids, second pass re-classifies. Local model handles scale, LLM only touches a small sample.

**Contrastive validation for user pools.** Reject discovered pools whose centroid is too close to an existing predefined pool (e.g. "Landmarks and Transit" is really just Places).

### Bigger picture

**Continuous personalization.** Pool as a background process that learns from user feedback. Centroids update with every confirmation or removal, new screenshots are assigned instantly.

**Screenshot-specific fine-tuning.** SigLIP2 is trained on general images. Screenshots are a distinct domain (UI, text overlays, cards). Fine-tuning on screenshot datasets (Rico, AMEX-8k) could improve embeddings significantly.

## Demo

[![Watch demo video on YouTube](https://img.youtube.com/vi/WbFFn4jEeQk/hqdefault.jpg)](https://youtu.be/WbFFn4jEeQk)
