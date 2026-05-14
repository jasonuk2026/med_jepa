# Event-Level JEPA Pretraining for EHR Sequences

This note summarizes the motivation, design choices, experimental results, and
current interpretation of the event-level JEPA experiments on packed MIMIC EHR
sequences. The short version is that the idea was well motivated, but the
current objective did not improve downstream ICU mortality linear probing over a
strong AR-only baseline.

## 1. Motivation

EHR sequences are naturally organized as events rather than only as a flat token
stream. A single patient timeline contains admission events, lab panels, vital
sign measurements, medication administrations, procedures, ventilation events,
and discharge or death outcomes. Within each event, there may be several tokens
describing codes, values, time information, and natural-language code
translations. Across events, the clinically meaningful signal is often not just
the next token, but the evolution from one event to later events.

Standard autoregressive language modeling trains the model to predict the next
token. This is useful because it forces the model to model local token
statistics and sequence order. However, it may be a weak match for our desired
downstream representation. For ICU mortality, we care about whether the patient
trajectory contains risk patterns over time: worsening physiology, therapies
that indicate severity, and combinations of abnormal events. These are often
event-level or trajectory-level signals rather than purely token-level signals.

The original motivation for adding a JEPA-style objective was therefore:

1. Encourage each event representation to contain predictive information about
   future clinical state.
2. Move the representation learning target from local next-token prediction
   toward event-level temporal abstraction.
3. Avoid forcing the model to generate exact future token identities, because
   EHR events are noisy, sparse, and many future details are not deterministic.
4. Learn a representation that could improve frozen-backbone linear probing for
   downstream labels such as ICU mortality.

The intended intuition was:

```text
current event representation -> predictor -> future event representation
```

If this worked, event representations should become more useful summaries of
the patient state, and averaging or selecting those event representations should
give a better input to a downstream linear classifier.

## 2. Data Representation

We tested both EOT and no-EOT packed EHR data.

In the EOT setting, events are separated by an explicit event boundary token:

```text
event_0 tokens <EOT> event_1 tokens <EOT> event_2 tokens <EOT> ...
```

In the no-EOT setting, the sequence removes the boundary token but keeps
`event_ids` in the parquet data:

```text
tokens:    t0  t1  t2  t3  t4  t5  t6  ...
event_ids:  0   0   0   1   1   2   2  ...
```

The no-EOT version was introduced to test a cleaner AR baseline where the model
does not spend capacity predicting or attending to artificial event separators.
For JEPA, `event_ids` are still used during loss construction to identify event
boundaries, even though the model input itself does not contain explicit EOT
tokens.

## 3. Model and Training Setup

Most recent experiments used:

- Backbone: `Qwen/Qwen3-0.6B-Base`
- Data: `data/pretrain/train_no_eot.parquet`
- Context length: 2048
- Global batch size: 128
- Epochs: 1 full epoch
- Learning rate: `2e-4`
- Warmup ratio: `0.10`
- Precision: bf16
- GPUs: 4
- Downstream evaluation: frozen backbone + linear classifier

The AR-only baseline trains the model with standard causal LM loss over the
no-EOT packed sequence:

```text
token_i -> token_{i+1}
```

The JEPA models add a representation prediction objective. We used a student
backbone, an EMA teacher backbone, and a predictor MLP. The student hidden state
at a source event is passed through the predictor, then matched to a teacher
hidden-state target from a future event.

## 4. JEPA Objective Variants

### 4.1 Original EOT-based Objective

The first version used EOT positions as the event representation. With
`future_k=2`, the source was an EOT representation and the target was based on
future events.

Conceptually:

```text
h(<EOT_i>) -> predictor -> representation of future events
```

This was motivated by the idea that an event boundary token could summarize the
preceding event, similar to how special tokens are sometimes used as sequence
representations.

### 4.2 No-EOT Last-Token Source

For no-EOT data, there is no boundary token in the model input. We therefore
used the last non-padding, non-EOT token of each event as the source
representation:

```text
source_i = h(last token of event_i)
```

This is a natural proxy for an event representation in a causal transformer,
because the final token of an event can attend to all earlier tokens in the same
event and all earlier patient history.

### 4.3 Mean Future-Token Target

The first no-EOT JEPA target was the mean of all non-EOT tokens from the next
`future_k` events:

```text
target_i = mean(h(tokens in event_{i+1}), ..., h(tokens in event_{i+k}))
```

This was intended to represent the future local clinical state without requiring
the model to predict exact future token identities.

In retrospect, this target is probably too diffuse. A mean over all future
tokens mixes codes, values, time tokens, and other details into a single vector.
It may wash out exactly the sparse, high-risk clinical signals that matter for
ICU mortality.

### 4.4 Last-Token Future Target

We later added a stricter and more symmetric target:

```text
source_i = h(last token of event_i)
target_i = stopgrad h(last token of event_{i+k})
```

For `future_k=2`, this trains:

```text
event_i_last -> event_{i+2}_last
```

This is better aligned with the downstream `mean_event_last_token` pooling,
because both the source and target are event-last-token representations rather
than last-token-to-mean-token representations.

The current implementation supports:

```bash
--jepa_source last_token
--jepa_target last_token
```

If the exact `future_k`-th future event does not exist, the pair is skipped. In
contrast, the older `--jepa_target mean` mode uses whatever future events exist
up to `future_k`; if only one future event remains, it averages that one.

## 5. Downstream Evaluation

The downstream task is ICU mortality. The evaluation protocol freezes the
pretrained backbone and trains a linear classifier on top of pooled hidden
states.

We tested several pooling strategies:

- `mean_eot`: average all EOT token representations.
- `last_eot`: use the final EOT token representation.
- `last_non_eot`: use the last valid non-EOT token of the sequence.
- `mean_event_last_token`: average the final non-padding, non-EOT token
  representation from each event.
- `appended_token`: append a final `<|endoftext|>` token for Qwen3-Embedding and
  use that representation.

The most important comparison for the no-EOT experiments is:

```text
AR-only no-EOT + last_non_eot
vs.
JEPA no-EOT + last_non_eot
vs.
JEPA no-EOT + mean_event_last_token
```

If JEPA improves event-level representations, we would expect improvement in
either `last_non_eot` or especially `mean_event_last_token`.

## 6. Results

The result table and plots are available here:

- `analysis/eval_results_summary.csv`
- `analysis/eval_results_table.png`
- `analysis/eval_metrics_line.png`
- `analysis/eval_results_table.md`

The strongest result so far is the no-EOT AR-only baseline:

```text
base_ar_only_no_eot_last_non_eot_linear_4gpu
AUROC = 0.7570
AUPRC = 0.2005
```

The best older no-EOT JEPA variants were lower:

```text
base_jepa_no_eot_last_token_jepa0p5_ar1_last_non_eot_linear_4gpu
AUROC = 0.7355
AUPRC = 0.1876

base_jepa_no_eot_last_token_jepa1_ar0p5_last_non_eot_linear_4gpu
AUROC = 0.7331
AUPRC = 0.1804

base_jepa_no_eot_last_token_ar1_last_non_eot_linear_4gpu
AUROC = 0.7242
AUPRC = 0.1718
```

The event-last-token pooling did not rescue the JEPA variants:

```text
base_ar_only_no_eot_mean_event_last_token_linear_4gpu
AUROC = 0.7260
AUPRC = 0.1682

base_jepa_no_eot_last_token_jepa0p5_ar1_mean_event_last_token_linear_4gpu
AUROC = 0.7139
AUPRC = 0.1607

base_jepa_no_eot_last_token_ar1_mean_event_last_token_linear_4gpu
AUROC = 0.7138
AUPRC = 0.1602
```

The base EOT JEPA model was also below AR-only:

```text
base_jepa_mse_full_epoch_last_eot_linear_4gpu
AUROC = 0.7263
AUPRC = 0.1700

base_jepa_mse_full_epoch_mean_eot_linear_4gpu
AUROC = 0.7258
AUPRC = 0.1750
```

Overall, the current evidence says that AR-only no-EOT pretraining gives the
best frozen linear-probe representation among the tested settings.

## 7. Important Caveat About Early Weight Sweeps

There was an implementation issue in the early AR/JEPA weight sweeps. The
original AR loss applied `--ar_weight` inside a weighted mean:

```text
sum(ar_weight * token_ce) / sum(ar_weight)
```

For no-EOT data, where all valid AR targets have the same weight, the scalar
weight cancels out. Therefore `ar_weight=0.5`, `ar_weight=1.0`, and
`ar_weight=0.03` behaved nearly the same as long as the weight was positive.

This has now been fixed. The current logic is:

```text
ar_loss = mean cross entropy
weighted_ar_loss = ar_weight * ar_loss
total_loss = jepa_weight * jepa_loss + ar_weight * ar_loss + var_weight * var_loss
```

The training log now prints both:

```text
ar=<unscaled AR mean CE>
ar_weighted=<actual AR contribution to total loss>
```

Because of this, the early JEPA-vs-AR weight sweep should be interpreted
carefully. It still shows that adding the tested JEPA objective did not beat
AR-only, but it should not be over-interpreted as a clean sweep over true AR
loss coefficients.

## 8. Why The Current JEPA Objective May Not Work

The negative result is plausible. The main issue is probably not a single bug,
but a mismatch between the training dynamics of the JEPA objective and the
downstream signal.

### 8.1 The JEPA target may be too smooth

Predicting a future hidden representation, especially a mean over future event
tokens, encourages the model to predict an average future state. ICU mortality
often depends on sparse but important signals: a high lactate, vasopressor use,
ventilation, severe renal dysfunction, or a particular sequence of worsening
events.

MSE to an averaged hidden state can penalize deviations from the average future
state, which may suppress rare but clinically meaningful features. This is very
different from AR training, which must preserve token-level detail because it
predicts exact next tokens.

### 8.2 AR and JEPA may optimize incompatible geometry

AR training shapes hidden states to support token prediction. It rewards
representations that retain local syntax, code identity, values, and order.

The JEPA objective shapes hidden states to be predictable under a smooth
future-representation loss. That may encourage invariances that are useful for
future-state prediction but harmful for a frozen linear classifier. In other
words, AR and JEPA may pull the representation geometry in different
directions.

This is why adding JEPA can reduce linear-probe performance even if the JEPA
loss itself decreases cleanly during training.

### 8.3 The target is self-distilled from the same model

The teacher is an EMA copy of the same backbone. This stabilizes training, but
the target is still a hidden state produced by the model itself. There is no
external semantic target telling the model which aspects of the future event are
clinically important.

The model can become good at matching its own latent geometry without becoming
better for ICU mortality prediction.

### 8.4 No-EOT data removes explicit event boundary tokens

The no-EOT setting is strong for AR-only pretraining, but it also means the
model input does not explicitly see event boundary tokens. We use `event_ids`
outside the model to define JEPA pairs, but the transformer itself only sees a
flat token stream.

Therefore the last token of an event is not guaranteed to become a robust event
summary. It is a reasonable proxy, but it is not explicitly trained as a summary
token except through the JEPA objective.

### 8.5 Frozen linear probing may expose only linearly accessible signal

The evaluation freezes the backbone and trains only a linear head. This is a
strict test of representation quality. A JEPA model might contain some useful
information that is not linearly accessible, or that requires LoRA/full
fine-tuning to extract.

However, because the goal here is to learn a useful general representation,
underperformance in a frozen linear probe is still meaningful. If AR-only
produces more linearly separable ICU mortality features, then AR-only is the
better representation learning baseline for this setting.

### 8.6 The downstream task may favor exact evidence retention

ICU mortality is not just about predicting what the next event looks like. It is
about retaining evidence of severity over the whole trajectory. AR naturally
preserves many details because any token could be the next prediction target.
JEPA can learn to ignore details that are hard to predict or unnecessary for
matching an average future hidden state.

This may explain why AR-only no-EOT currently wins.

## 9. Current Interpretation

The motivation was to make event representations more predictive of future
clinical state. That is a sensible research direction. But the current
implementation of JEPA did not improve the representation for ICU mortality
linear probing.

The most likely explanation is that the current JEPA objective learns a
future-state smoothing task, while the downstream classifier needs retention of
sparse, discriminative, trajectory-level risk evidence. The two objectives have
different dynamics.

The result does not prove that event-level pretraining cannot work. It says that
this particular form:

```text
event representation -> future hidden representation
```

has not beaten:

```text
plain AR on no-EOT EHR tokens
```

under frozen linear-probe evaluation.

## 10. Next Reasonable Experiments

If we continue this direction, the experiments should be more targeted:

1. Run the fixed-weight, last-token-target JEPA sweep:

   ```text
   jepa=1.0, ar=0.3
   jepa=1.0, ar=0.6
   jepa=1.0, ar=1.0
   jepa=0.6, ar=1.0
   jepa=0.3, ar=1.0
   ```

   These are cleaner than the earlier sweep because AR weighting is now fixed
   and the JEPA target is event-last-token rather than future-token mean.

2. Evaluate each new checkpoint with both:

   ```text
   last_non_eot
   mean_event_last_token
   ```

3. If these do not beat AR-only, deprioritize this JEPA variant.

4. If we still want event-level objectives, consider targets that are more
   clinically grounded:

   - Predict future high-level code groups.
   - Predict event type or phenotype buckets.
   - Predict time-to-next-ICU-discharge/death style targets.
   - Add an explicit learnable event summary token.
   - Use contrastive or ranking objectives over future events rather than MSE to
     self-hidden states.

5. Add a LoRA downstream evaluation to test whether JEPA information is present
   but not linearly accessible.

## 11. Takeaway

The design was motivated by a real limitation of token-level AR modeling for
EHR: clinical meaning is often event-level and trajectory-level. The JEPA
objective was intended to make event representations predictive of future
clinical state.

The experiments so far do not support this objective as implemented. The best
model remains the no-EOT AR-only baseline. The likely reason is that AR and JEPA
shape representations differently: AR preserves detailed evidence needed for
clinical risk prediction, while the current JEPA objective encourages smooth
future representation prediction that may discard sparse but important risk
signals.

