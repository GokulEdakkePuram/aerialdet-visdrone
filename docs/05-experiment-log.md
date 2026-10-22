# Experiment log

A running journal. The point is to record what was expected *before* each run,
so that being wrong stays visible instead of getting quietly rewritten into a
tidy narrative afterwards.

Template for each entry:

---

## YYYY-MM-DD — <short title>

**Question.** What am I trying to find out?

**Expectation.** What I think will happen, and why. Written before running.

**Setup.** Config, commit SHA, hardware, wall-clock time.

**Result.** Numbers. Link the run directory.

**Reading.** Did the expectation hold? If not, what is the most likely
explanation, and what would distinguish between the candidates?

**Next.** The single most informative experiment to run after this one.

---

## 2026-08-28 — Pipeline smoke test

**Question.** Does the whole path — download, convert, train, validate, export
— actually run end to end before any real compute is committed to it?

**Expectation.** Yes, with meaningless metrics (1 epoch on 2% of the data).

**Setup.** `configs/smoke.yaml`, YOLO11n, 640 px, MPS on an M2 Pro (16 GB).

**Result.** _to fill in_

**Reading.** _to fill in_

**Next.** `baseline_640`, the control for the resolution ablation.

---

The entries below were written up after the runs, from the README and docs.
The expectations are the ones the docs committed to before each run.

## Resolution and capacity sweep

**Question.** Does raising `imgsz` help, and is the gain about object size?

**Expectation.** Monotonic gain 640 → 1280, concentrated in the smallest
classes ([doc 01](01-why-small-objects-are-hard.md)).

**Setup.** `baseline_640`, `finetune_960`, `finetune_1280`, `size_n_960`,
`size_m_960`. RTX 4090, `cuda24` profile, batch 8 throughout. 7.4 h of training.

**Result.** 0.2218 → 0.2873 → 0.3260 mAP50-95 across resolutions; n/s/m at
960 gives 0.2324 / 0.2873 / 0.3203. Absolute AP50 gain vs median object size:
r = −0.88.

**Reading.** Held. Unexpected part: `size_m_960` and `finetune_1280` are
nearly tied, and they swap rank on test-dev.

**Next.** Is the gain from grid density or from object scale? → P2 head.

## Tiled vs whole-frame inference

**Question.** Does tiling help a model that was already trained at 1280?

**Expectation.** Smaller gain than on a weak checkpoint (+45% there).

**Setup.** `aerialdet tiled-eval` on `finetune_1280` and `baseline_640`.

**Result.** −10.1% on `finetune_1280`, +12.3% on `baseline_640`.

**Reading.** Worse than expected: tiling only stands in for resolution, and
whole-frame 1280 beats tiled 640 on both accuracy and latency.

**Next.** —

## P2 head at 640

**Question.** Does a stride-4 grid at 640 match 1280, which has the same
finest grid?

**Expectation.** If grid density is the mechanism, `p2_640` ≈ `finetune_1280`.

**Setup.** `p2_640`, same settings as `baseline_640`.

**Result.** 0.2204, level with `baseline_640` and far from 0.3264. Per class,
small objects gain slightly and large ones lose (r = −0.74 with size).

**Reading.** Grid density is a minor part of the effect. Caveat: P2 only
transfers layers 0-16 from COCO.

**Next.** Same partial transfer on the stock head → `baseline_640_partial`.

---

## 2026-10-22 — Init control for P2

**Question.** Did `p2_640` tie with `baseline_640` because of its
architecture, or because it starts with layers 17+ random?

**Expectation.** A small drop, within ~3% of `baseline_640`. P2 had plateaued
by epoch 37, which suggests 50 epochs is enough to recover from a random
bottom-up path. If it drops much further, P2 was held back by its init and
the architecture is worth more than the tie suggests.

**Setup.** `baseline_640_partial`: `baseline_640` with `transfer_upto: 16`.
RTX 4090 (48 GB variant), `cuda24` profile, so batch 8 as before.

**Result.** 0.2062 mAP50-95 (0.3589 mAP50), 7.0% below `baseline_640`.
`p2_640` is +6.9% over this control, with the gain tracking object size
(r = −0.76). The init cost is flat across sizes (r = −0.19).

**Reading.** Wrong. The partial init cost more than expected, and it was
masking a real P2 gain. Plateauing early showed P2 had converged, not that it
converged to the same place. P2 helps small objects without hurting large
ones, but is still ~6x weaker than resolution.

**Next.** A second seed for `baseline_640_partial` and `p2_640` to put an
error bar on the 0.014 gap.
