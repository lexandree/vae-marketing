# Model Selection And Usage Guide

This note explains what the current models are for, what the validation
artifacts mean in practice, and how to use the environment on real data
without making invalid claims.

It is written as an engineering guide, not as a research paper.

## The Big Picture

The repository now has three different model roles:

1. `quasi-causal validation`
2. `general latent representation`
3. `campaign-specific latent interpretation`

These roles should not be mixed.

Use the pipeline in this order:

1. validate the campaign effect first
2. only then interpret the effect with latent models
3. only report final latent results from leakage-controlled splits

If step 1 fails, steps 2 and 3 may still be useful as descriptive analysis, but
they should not be written up as causal findings.

## What Each Model Is For

### `beta-VAE`

Purpose:

- best general latent baseline
- best overall held-out semantic quality
- safest default representation model

Current status:

- strongest held-out `MIG` and `SAP`
- weaker campaign bridge than `beta-TCVAE`
- more general-purpose than `Contrastive VAE`

Use it when:

- you need one default latent model for new data
- you want stable factor validation
- you want the least controversial latent baseline

Do not use it to claim:

- campaign causality
- campaign-specific signal isolation by itself

### `Contrastive VAE`

Purpose:

- isolate what is salient for target windows relative to background windows
- best fit for campaign-specific shift analysis

Current status:

- not the best global latent model
- useful when the question is "what distinguishes treated campaign behavior?"
- strongest when evaluated in `salient-only` mode

Use it when:

- you want to compare `treated campaign windows` vs `background windows`
- you want a model that naturally emphasizes campaign-salient variation
- you want to explain the profile of a specific campaign response

Do not use it to claim:

- stronger overall disentanglement than `beta-VAE` unless held-out metrics show it
- causal effect identification

### `beta-TCVAE`

Purpose:

- explicit disentanglement pressure through `total correlation`
- stronger mapping from campaign outcomes to specific latent dimensions

Current status:

- weaker global held-out `MIG/SAP` than `beta-VAE`
- stronger campaign bridge for `26` and `30`
- useful as an interpretation model, not as the main universal baseline

Use it when:

- you already have a validated campaign effect
- you want a sharper bridge from `campaign_delta` to latent factors
- you want to inspect factors tied to `total_spend` and `category_diversity`

Do not use it as the only model in the project narrative.

## What The Main Terms Mean In Practice

### `Held-out`

Meaning:

- data that the model never saw during training

Practical use:

- trust held-out metrics
- do not trust in-sample brilliance

Rule:

- if a result is not held-out, treat it as exploratory

### `Leakage`

Meaning:

- the model saw evaluation households or evaluation campaigns during training

Practical use:

- leakage makes latent mappings look cleaner than they really are
- leakage is one of the easiest ways to accidentally produce impressive but weak results

Rule:

- final campaign case studies must use leakage-controlled training and evaluation

### `Latent factor`

Meaning:

- one hidden coordinate in the compressed representation

Practical use:

- a latent factor is not automatically interpretable
- it becomes interpretable only after it repeatedly aligns with observed attributes

### `MIG`

Meaning:

- a disentanglement metric that checks whether one latent dimension dominates the explanation of an observed attribute

Practical use:

- higher `MIG` means cleaner separation between factors
- useful for ranking general latent quality

Bad use:

- do not read `MIG` as business impact

### `SAP`

Meaning:

- a metric that checks whether the best latent factor explains an attribute clearly better than the runner-up

Practical use:

- higher `SAP` means attributes are less ambiguously spread across many latent dimensions

Bad use:

- do not use it as a campaign-effect metric

### `Total Correlation`

Meaning:

- dependence between latent dimensions

Practical use:

- lower effective total correlation usually means factors are less redundant
- `beta-TCVAE` explicitly penalizes this

Bad use:

- low total correlation alone does not guarantee useful semantics

### `Bridge`

Meaning:

- a table that links campaign-visible attribute shifts to latent factors

Practical use:

- this is where campaign validation and latent interpretation meet
- it is the main artifact for explaining what kind of behavioral shift the model thinks happened

## Current Practical Winners

### Best overall latent baseline

- `noleak-beta-vae-32d`

Why:

- best held-out `MIG/SAP`
- safest default for general latent validation

### Best campaign-salient model

- `contrastive-vae-best-noleak`

Why:

- best tuned contrastive configuration
- best choice for target-vs-background interpretation

### Best bridge model

- `beta-tcvae-32d-noleak`

Why:

- strongest bridge associations for:
  - `total_spend`
  - `category_diversity`

## How To Work With Real Data

When new data arrives, use this workflow.

### Stage 1: Build Reliable Splits

Use:

```bash
python main.py build-household-splits \
  --campaign-table data/campaign_table.csv \
  --eval-campaign-ids 26 30 \
  --output data/splits/household_splits.json \
  --seed 42
```

Goal:

- isolate case-study campaigns
- prevent representation leakage

Use this stage whenever:

- you introduce new evaluation campaigns
- you rebuild representation models

### Stage 2: Prepare No-Leak Training And Eval Frames

Use `prepare.py` and `build-window-attributes` to build:

- no-leak train frames
- no-leak validation/eval frames
- aligned observable attributes

Goal:

- keep latent training separate from case-study interpretation

### Stage 3: Validate Campaign Effects First

Use:

- `build-validation-data`
- `validate-campaigns`
- `analyze-campaign-sensitivity`

Goal:

- decide whether the campaign has:
  - `supported`
  - `weak`
  - `unsupported`
  evidence

Practical rule:

- if `validate-campaigns` says `unsupported`, do not write causal-looking claims

### Stage 4: Choose The Latent Model By Task

If the task is:

- "I need a stable baseline latent model"
  - use `beta-VAE`
- "I want to isolate treated vs background behavior"
  - use `Contrastive VAE`
- "I want the sharpest campaign-to-latent bridge"
  - use `beta-TCVAE`

### Stage 5: Run Latent Validation

Use:

```bash
python main.py validate-latents ...
```

Goal:

- measure whether latent factors align with observed business attributes on held-out data

Practical rule:

- do not skip held-out latent validation before using any bridge output in the write-up

### Stage 6: Build The Campaign Bridge

Use:

```bash
python main.py build-campaign-latent-bridge ...
```

Goal:

- explain which validated factors correspond to campaign-visible shifts

This is the main artifact for stakeholder-facing interpretation.

## What To Say In Real Work

### Safe language

- "The restricted quasi-causal design supports an increase in `total_spend`."
- "The held-out latent validation associates one factor with `category_diversity`."
- "The bridge suggests the campaign shift aligns with a validated spend-related factor."

### Unsafe language

- "The VAE proved the campaign caused more spend."
- "Latent 24 is the true causal mechanism."
- "The model discovered the exact business driver."

## Recommended Default Strategy

If you do not want to think too much each time, use this default:

1. validate campaign effects with the restricted design
2. run `beta-VAE` as the default latent baseline
3. run `Contrastive VAE` if the case is explicitly target-vs-background
4. run `beta-TCVAE` if you need a stronger campaign interpretation bridge
5. only report final claims from no-leak, held-out artifacts

## Key Files And Artifacts

Quasi-causal:

- `data/restricted_26_w2/restricted/`
- `data/restricted_30_w4/restricted/`

General latent baseline:

- `experiments/noleak-beta-vae-32d`
- `experiments/latent_validation_noleak_eval`

Campaign-salient model:

- `experiments/contrastive-vae-best-noleak`

Bridge-oriented disentanglement model:

- `experiments/beta-tcvae-32d-noleak`
- `experiments/latent_validation_beta_tcvae_noleak_eval`

Interpretation artifacts:

- `data/restricted_26_w2/latent_bridge_noleak`
- `data/restricted_30_w4/latent_bridge_noleak`
- `data/restricted_26_w2/latent_bridge_beta_tcvae`
- `data/restricted_30_w4/latent_bridge_beta_tcvae`

## Final Rule

Do not choose the model that looks most sophisticated.

Choose the model that answers the actual question:

- causal credibility question -> quasi-causal validation layer
- general latent quality question -> `beta-VAE`
- target-vs-background contrast question -> `Contrastive VAE`
- campaign interpretation bridge question -> `beta-TCVAE`
