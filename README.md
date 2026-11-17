# HarmonyRL — research record

English | [한국어](README.ko.md)

Melody harmonization with reinforcement learning. Given a melody, a model writes a chord
progression as a sequence of (chord, duration) events. A transformer policy is pretrained on
human accompaniments and then fine-tuned with KL-regularized reinforcement learning.

This repository holds the research code: data preparation, training, evaluation, and the
listening tools. The models it produced are released for use elsewhere:

- Library: [`HarmonyRL/HarmonyRL`](https://github.com/HarmonyRL/HarmonyRL), on PyPI as
  [`harmonyrl`](https://pypi.org/project/harmonyrl/)
- Models: [`CloudHolic/harmonyrl-chromatic`](https://huggingface.co/CloudHolic/harmonyrl-chromatic),
  [`-smooth`](https://huggingface.co/CloudHolic/harmonyrl-smooth),
  [`-base`](https://huggingface.co/CloudHolic/harmonyrl-base), in the
  [HarmonyRL collection](https://huggingface.co/collections/CloudHolic/harmonyrl)

**In short.** Rewards learned from human data or from listener preferences did not pick
outputs a listener preferred. The final reward is a product goal instead: as many chords
outside the key as a KL penalty to the pretrained policy allows. RL with it doubled the time
under chords outside the key (6.9% → 12.9% on TEST) while blind listening rated it as good as
the pretrained policy once each could be redrawn. Neither hypothesis below found a difference
by ear.

## Contents

1. [Task](#task)
2. [Hypotheses](#hypotheses)
3. [Data](#data)
4. [Method](#method)
5. [Rewards that did not work](#rewards-that-did-not-work)
6. [Evaluation](#evaluation)
7. [Results](#results)
8. [Released models](#released-models)
9. [Reproduction](#reproduction)
10. [Repository layout](#repository-layout)
11. [Limitations](#limitations)
12. [License and references](#license-and-references)

## Task

**Input.** A melody segment: notes on an 8th-note grid, its time signature (any `x/4` or `x/8`
that does not change within the segment), its local keys over time, and optionally its section
label and tempo.

**Output.** A sequence of chord events that tiles the segment. Each event is
(chord, duration), with the duration in 8th notes (1–128). An episode is one segment; its
length is the number of chords.

**Chord representation.** Chords are written relative to the local key, as in the Hooktheory
data, and decomposed into components that the model predicts one after another:

| Head | Values |
|---|---|
| root | scale degree 1–7, or no chord |
| applied | 0–7 (secondary function, e.g. V/V) |
| borrowed | mode the chord is borrowed from, or none (9 modes) |
| type | triad, 7th, 9th, 11th, 13th |
| suspensions | none, sus2, sus4, both |
| adds | combinations of add 4, 6, 9 |
| omits | none, no 3, no 5, both |
| alterations | combinations of ♭5, ♯5, ♭9, ♯9, ♯11, ♭13 |
| inversion | 0–3 |
| duration | 1–128 8th notes |

**No chord.** A generated harmonization may leave only the pickup before the 1st full bar
without a chord. Elsewhere "no chord" is masked out, in training and in generation.

## Hypotheses

The goal is a model that harmonizes a given melody well. 2 hypotheses were tested along the
way; the order and number of runs followed the goal, not the hypotheses.

| | Question | Answer |
|---|---|---|
| **H1** | Does a sequence-level reward give more natural progressions than a local per-note reward (as in RL-Chord)? | **Not supported.** The local-reward model was preferred in 12 of 20 songs in round 6, and the 2 split 4 : 6 in round 7. Neither is significant. |
| **H2** | For the same target distribution π_ref · exp(R/β), does forward-KL training (DPG) keep more diversity than reverse-KL training (RLOO)? | **No meaningful difference.** Sample disagreement on TEST 0.808 (forward) vs 0.800 (reverse); same out-of-key share (12.9%); listening 4 of 10. |

The sequence-level reward of H1 ended up as the product goal described under
[Method](#method), not a learned evaluator: no learned reward passed the listening check
([Rewards that did not work](#rewards-that-did-not-work)).

## Data

[Hooktheory](https://www.hooktheory.com) melodies and chords, as released with
[Sheet Sage](https://github.com/chrisdonahue/sheetsage)
([data repository](https://github.com/chrisdonahue/sheetsage-data), commit `06113c0`). Both
the simplified file (`Hooktheory.json.gz`) and the raw file (`Hooktheory_Raw.json.gz`) are
read: the raw file supplies the chords in full spelling, the section label, and the tempo.

| Step | Items |
|---|---|
| Items in the dataset | 26,175 |
| With both melody and chords | 23,612 |
| Time signature changes inside the item (excluded) | 269 |
| Chords that cannot be represented (excluded) | 2 |
| **Segments used** | **23,341** |
| — with a key change inside | 473 |

Every time signature is kept as long as it does not change within a segment: 4/4 21,195,
6/4 1,127, 3/4 843, 6/8 47, 2/4 44, 12/8 32, 5/4 25, others 30. Key changes are kept, and the
local key is a model input.

The splits are the dataset's own `split` field:

| TRAIN | VALID | TEST |
|---|---|---|
| 18,937 | 1,931 | 2,473 |

All choices were made on VALID. TEST was evaluated once, at the end.

## Method

### Pretrained policy π_ref

`ChordPolicy` is an encoder–decoder transformer (18.4M parameters).

- **Encoder** over the melody, 1 frame per 8th note. Frame inputs: the pitch classes sounding
  in the frame relative to the local tonic (weighted by overlap), note onset, octave, position
  in the bar, beat strength, local mode, and key change. Per segment: beats per bar, beat unit,
  section label, and tempo bucket.
- **Decoder** over the chord events so far. Each event is predicted head by head, each head
  conditioned on the heads already chosen, with action masks that keep the chord valid.
- 384 dimensions, 4 attention heads, 4 encoder and 4 decoder layers, dropout 0.3.
- Trained by maximum likelihood on TRAIN: AdamW, learning rate 3e-4 with 500 warmup steps,
  weight decay 0.1, up to 15 epochs with early stopping. Section label and tempo are dropped
  at random (20% each) so the model works without them.
- Best epoch 13: VALID negative log-likelihood 2.529 per event. Accuracy per head: root 0.719,
  type 0.899, inversion 0.909, duration 0.807.

### Baselines

- **HMM**, after Yeh et al. (2021): 1 chord per slot between strong beats, Viterbi decoding.
- **π_ref greedy** and **π_ref sampled at temperature 0.85**. Sampling at T 1 produced chords
  that wandered off for no reason, and the greedy decode removes most chords outside the key
  (2.9% of the time against 9.6% for humans). T 0.85 was chosen by listening as the decoding
  RL had to beat. Without RL, π_ref at T 1 was already indistinguishable from the human
  accompaniments in a blind check on short songs (10 of 20).

### Reinforcement learning

Both KL directions fine-tune toward the same target distribution

&nbsp;&nbsp;&nbsp;&nbsp;p(y) ∝ π_ref^(1/0.85)(y) · exp(R(y) / β)

where π_ref^(1/0.85) is the pretrained policy sharpened to temperature 0.85, the decoding
listeners heard as the safe baseline. The policy starts from the same sharpened weights, so
the KL starts at 0, and it is decoded at T 1 afterward.

For each melody k = 8 samples are drawn from the current policy, and each gets the log ratio
s = R/β + log π_ref(y) − log π(y). The 2 methods share the rollout, the optimizer, and the
logging, and differ only in how the samples are weighted in −Σ w · log π(y):

| | Divergence | Weight w |
|---|---|---|
| **RLOO** (Ahmadian et al., 2024) | reverse KL, KL(π ‖ p) | β · (s − mean s of the other samples of the melody) |
| **DPG** (Parshakova et al., 2019; Khalifa et al., 2021) | forward KL, KL(p ‖ π) | k · softmax(s) over the samples of the melody |

There is no value function and no PPO. Settings for every run: 200 updates of 32 melodies ×
8 samples from TRAIN, learning rate 1e-5, gradient clip 1.0, seed 0.

### Final reward

The reward is given once, at the end of the episode:

&nbsp;&nbsp;&nbsp;&nbsp;R = (share of time under chords outside the key) − 2 × (share of time after 1 chord has held for 2 bars)

- **Outside the key**: the chord has a pitch class outside the local scale. In minor keys the
  scale is natural, harmonic, and melodic minor together; other modes use their own scale.
  This states what the product wants (rich reharmonization), not whether a chord is right:
  how far the policy may go at each place is left to π_ref through the KL term, and the
  overall boldness to β, chosen by listening (β 0.2).
- **Held**: the same pitch classes, in any inversion, sounding beyond 2 bars. Listeners
  rejected long holds (when only 1 of 2 harmonizations held a chord for 3 bars or more, the
  other won 33 of 42), and π_ref holds as often as the data, so the KL term does not prevent
  them.

No other rule judging harmony enters the reward.

### Local reward baseline (H1)

A re-implementation of the core of the RL-Chord reward (Ji et al., 2024), averaged over the
notes and rests of the melody:

- agreement with the human chord at the same time (shared pitch classes, less mismatches of
  bass, pitch classes, and no chord);
- consonance of the chord with the melody notes around it and within itself;
- progression: −1 for a chord held across a bar line, +1 inside a bar, −1 when no tone is
  shared with the chord before.

It was trained with RLOO at β 0.8, chosen separately because the reward has another scale.

## Rewards that did not work

Most of the project went into finding a reward that picks outputs a listener prefers. Every
candidate was first checked without RL: if the outputs a reward picks out of N samples
(best-of-N) do not sound better, RL toward that reward has no reason to either. All checks
below are blind listening comparisons against π_ref at T 0.85, by 1 listener.

| Reward | How it was learned | Best-of-N vs T 0.85 | Agreement with held-out judgments |
|---|---|---|---|
| Discriminator from scratch | frame transformer, human vs π_ref samples | best-of-8 11 : 10, best-of-32 10 : 12 | chance |
| Discriminator, MusicBERT fine-tuned | AUC 0.766 human vs sample | best-of-32 8 : 14 | chance |
| Preference reward, MusicBERT features | Bradley–Terry, linear heads on frozen features | — | 0.44–0.56 |
| Preference reward, π_ref states | Bradley–Terry on decoder states | best-of-32 16 : 16 | 0.64 in cross-validation → 0.49 on new songs |
| Preference reward, log π_ref | Bradley–Terry on mean and minimum event log-likelihood | best-of-32 14 : 18 | 0.59–0.60 → 0.54 on new songs |

The preference rewards were fit to and checked against 348 pairwise and ranking judgments by
the owner. Also tried and at chance: a symbolic LLM judge, a sum of rule penalties, and
P(melody | chords).

What these showed:

- **The judgments themselves were consistent** (17 of 18 repeated comparisons agreed). The
  signal is in the ear; the features could not carry it.
- **Cross-validation on a few hundred judgments was optimistic.** Accuracies of 0.69 and
  0.64 vanished on new songs. A reward was trusted only when listening on unseen songs agreed.
- **High-dimensional features overfit** a few hundred judgments, and smaller ones carried no
  more than the log-likelihood under π_ref, which saturates.
- **Preference is not monotone in likelihood.** Between close candidates the more likely one
  wins (0.59), but the greedy decode does not beat sampling. π_ref cannot tell "rare because
  bold" from "rare because wrong".
- **"Human-like" is the wrong target.** The data are user transcriptions of songs of every
  genre, with intentional oddities and transcription noise; a discriminator learns those too.
- **The model's greedy direction is the opposite of the listener's.** The likely chords are
  the common ones, which removes the chords outside the key the listener valued when they fit.

This led to the final reward: rather than predicting the judgment, state the goal (as many
chords outside the key as stay safe), and let π_ref through the KL term decide what is safe
at each place.

## Evaluation

### Metrics

Every system was scored on the same segments against the human accompaniments by
`scripts/eval/evaluate.py` (8 samples per segment for the policies):

- **Yeh et al. (2021)**: chord histogram entropy (CHE), chord coverage (CC), chord tonal
  distance (CTD), chord-tone to non-chord-tone ratio (CTnCTR), pitch consistency score (PCS),
  melody–chord tonal distance (MCTD).
- **Structure**: agreement of chords on repeated phrases, consistency with the key, chord
  distributions at and inside phrase ends (Jensen–Shannon distance to human).
- **Idiom**: non-chord tones on strong beats and their resolution, share of tensions,
  borrowed chords, and applied chords.
- **Diversity**: chord 2- and 3-gram distance to human, disagreement between samples of the
  same melody, length of passages copied from TRAIN.
- **Reward terms**: out-of-key share and held share.

No metric is used as a reward, except the 2 reward terms of the final reward.

### Listening

The decisions were made by blind listening by 1 listener, the project owner. Sets were small
(10–20 VALID songs each), repeated over several rounds, with order randomized and 2 groups
repeated to check self-consistency. Later rounds ranked K candidates per song, marked each as
acceptable or not, and marked the bars with problems. For listening only, each set was
transposed so the 1st key lies on the white keys (C major, A minor); the models never see
this. The judgments are not published.

## Results

### TEST (2,473 segments)

| | Human | HMM | π_ref greedy | π_ref T 0.85 | RL reverse | RL forward | RL local |
|---|---|---|---|---|---|---|---|
| Out-of-key share | 9.6% | 5.8% | 2.9% | 6.9% | **12.9%** | **12.9%** | 5.5% |
| Held share (> 2 bars) | 3.2% | 16.0% | 7.4% | 4.1% | 1.4% | 2.1% | 5.3% |
| Borrowed chords | 11.0% | 6.6% | 4.7% | 8.4% | 13.1% | 13.4% | 7.2% |
| Applied chords | 4.2% | 0.5% | 0.6% | 3.0% | 4.5% | 4.5% | 2.2% |
| Chords per bar | 1.34 | 0.88 | 1.13 | 1.29 | 1.32 | 1.31 | 1.20 |
| Sample disagreement | – | – | – | 0.767 | 0.800 | 0.808 | 0.716 |
| Chord 3-gram JS to human | 0 | 0.829 | 0.727 | 0.677 | 0.694 | 0.699 | 0.672 |
| CHE | 1.49 | 0.85 | 1.13 | 1.39 | 1.48 | 1.47 | 1.32 |
| CC | 6.04 | 3.25 | 3.55 | 5.23 | 5.65 | 5.67 | 4.81 |
| CTnCTR | 0.911 | 0.929 | 0.883 | 0.880 | 0.875 | 0.877 | 0.888 |
| PCS | 0.505 | 0.628 | 0.530 | 0.482 | 0.472 | 0.470 | 0.506 |
| MCTD | 1.338 | 1.262 | 1.353 | 1.372 | 1.378 | 1.378 | 1.355 |
| Key consistency | 0.855 | 0.922 | 0.948 | 0.894 | 0.830 | 0.829 | 0.914 |
| Repeated-phrase agreement | 0.856 | 0.977 | 0.984 | 0.879 | 0.871 | 0.869 | 0.896 |
| KL to π_ref^(1/0.85), per song | – | – | – | 0 | 0.257 | 0.244 | 0.215 |

RL reverse and forward are β 0.2 toward the final reward; RL local is β 0.8 toward the local
reward. The VALID results showed the same pattern, so the choices did not overfit VALID.

- The final reward did what it asked: twice the out-of-key time of π_ref at T 0.85, above the
  human level, with the fewest long holds, and chord variety (CHE, CC) close to the human level.
- The local reward moved the other way: more consonant (PCS), fewer chords outside the key,
  less diverse, and longer holds.
- The 2 KL directions are nearly identical on every metric. With β 0.2 and a sharpened
  reference, both stayed at a KL of about 0.25 per song.

### Listening

| Round | Songs | Systems | Result |
|---|---|---|---|
| 5 | 20 | T 0.85, reverse, forward | Acceptable 11 / 14 / 14. Reverse over T 0.85 in 13 of 20, forward over reverse in 8 of 19; none significant. |
| 6 | 20 | T 0.85, reverse, local | Acceptable 15 / 10 / 16. Local over reverse in 12 of 20 (H1). Rounds 5 and 6 together: reverse over T 0.85 in 21 of 40. |
| 7, stage 1 | 10 × 3 draws | T 0.85, forward, reverse, local | Acceptable draws 20 / 16 / 16 / 19 of 30. Bars marked as problems 5.7 / 18.8 / 19.5 / 6.9%. |
| 7, stage 2 | 10 | best draw of each system | Mean rank 2.6 / 2.6 / 2.4 / 2.4; every pair 4–6 of 10. |

Once a listener can redraw and keep the best of 3, the 4 systems could not be told apart, and
the best draw of the reverse model kept twice the out-of-key time of T 0.85 (10.4% vs 6.0%).
A single draw is less reliable: in most rounds about 40% of the bars the out-of-key reward
adds outside the key were marked as problems (41–47% in round 7, 7% in round 5), and many
problems were 1 bar long. RL taught the policy to go outside the key more often, not to
choose better where to do it; π_ref cannot say where it is safe in a given genre, and genre
is not an input.

## Released models

| Model | Source run | Default temperature | Style |
|---|---|---|---|
| [`harmonyrl-chromatic`](https://huggingface.co/CloudHolic/harmonyrl-chromatic) | RL reverse, β 0.2, final reward | 1.0 | rich reharmonization, many chords outside the key |
| [`harmonyrl-smooth`](https://huggingface.co/CloudHolic/harmonyrl-smooth) | RL local, β 0.8 | 1.0 | consonant, resolving progressions |
| [`harmonyrl-base`](https://huggingface.co/CloudHolic/harmonyrl-base) | π_ref, pretraining only | 0.85 | close to the training data |

The forward-KL run is kept only as the H2 result. All 3 models are meant to be used with
redrawing: the library returns several distinct candidates and redraws only the bars chosen.

## Reproduction

### Environment

- Python 3.14 managed by [uv](https://docs.astral.sh/uv/): `uv sync` installs everything,
  `uv run` runs a script. The `notebook` dependency group holds what the notebooks need.
- Developed on an NVIDIA GB10 (aarch64, CUDA 13.0, driver 580). PyTorch comes from PyPI.
- On aarch64, `uv run` may reinstall `nvidia-cusparselt-cu13` on every call, which breaks
  another `uv run` already in progress. Start long jobs with `uv run --no-sync`.
- Rendering and listening tools need [FluidSynth](https://www.fluidsynth.org) with
  `FluidR3_GM.sf2`, ffmpeg, and [MuseScore 4](https://musescore.org) as `mscore`. Their paths
  are in `configs/rendering.yaml` and `configs/build_round.yaml`. Training and evaluation do
  not need them.

### Data

```sh
git clone https://github.com/chrisdonahue/sheetsage-data
git -C sheetsage-data checkout 06113c0
mkdir -p data/raw/hooktheory
cp sheetsage-data/hooktheory/Hooktheory.json.gz sheetsage-data/hooktheory/Hooktheory_Raw.json.gz data/raw/hooktheory/
```

SHA-256:

```
917b7cd58f5f4e07d6c36acf7bfad958c99ee05472dab3555399141094698e0c  Hooktheory.json.gz
716af2979f060400c302ab098dd45d9f8c5fe4d4b3b1fe61c478dd9bdf041634  Hooktheory_Raw.json.gz
```

### Runs

Every script is configured with [Hydra](https://hydra.cc): the defaults are in
`configs/<script>.yaml`, overrides are `key=value`, `--cfg job` prints the composed
configuration, and `-m` runs a sweep. Each run writes into `output_dir` (by default under
`runs/`, named by `name=` or the start time), refuses a directory that is not empty, and saves
the composed configuration, the git hash, and its metrics.

```sh
# Segments for the 3 splits, into data/processed
uv run scripts/data/prepare_data.py

# π_ref; the best epoch is saved in runs/pretrain/final/best
uv run --no-sync scripts/train/pretrain.py name=final

# HMM baseline
uv run scripts/train/train_hmm.py name=base

# RL
uv run --no-sync scripts/train/train_rl.py name=reverse-b0.2
uv run --no-sync scripts/train/train_rl.py name=forward-b0.2 train.divergence=forward
uv run --no-sync scripts/train/train_rl.py name=local-b0.8 train.reward=local train.beta=0.8
```

Evaluation scores 1 system per run. `rewards` lists preference reward models to score with;
drop it when there is none.

```sh
uv run --no-sync scripts/eval/evaluate.py split=test name=human system=human '~rewards.reward'
uv run --no-sync scripts/eval/evaluate.py split=test name=hmm system=hmm '~rewards.reward'
uv run --no-sync scripts/eval/evaluate.py split=test name=pi_ref_greedy system=policy system.greedy=true '~rewards.reward'
uv run --no-sync scripts/eval/evaluate.py split=test name=pi_ref_t0.85 system=policy '~rewards.reward'
uv run --no-sync scripts/eval/evaluate.py split=test name=rl_reverse_b0.2 system=policy \
	system.model=runs/rl/reverse-b0.2/final system.temperature=1.0 '~rewards.reward'
```

The forward and local runs are evaluated the same way. `notebooks/results.ipynb` reads the
evaluation directories and draws the tables and figures.

A trained policy becomes a released model, with its default temperature, by:

```sh
uv run scripts/train/release.py --source runs/rl/reverse-b0.2/final --output runs/release/chromatic
uv run scripts/train/release.py --source runs/pretrain/final/best --output runs/release/base --temperature 0.85
```

The preference reward (`scripts/train/train_reward.py`) and the listening tools need the
owner's judgments, which are not published.

### Code checks

```sh
uv run ruff check
uv run ruff format --check
uv run ty check
```

## Repository layout

```
configs/                  Hydra configuration per script; system/ holds the systems to evaluate
scripts/
  data/                   dataset preparation and statistics, building and collecting listening rounds
  train/                  pretraining, HMM, preference reward, RL, release
  eval/                   evaluation of 1 system on 1 split
  io/                     rendering, checks of imported files
  tools/                  listening tools (pairwise and ranking judgments, browsing)
src/harmonyrl_research/
  schema/                 data contract: melodies, chord vocabulary, meters, keys, action masks
  data/                   joining, filtering, and converting the source files
  metrics/                Yeh et al., structure, idiom, diversity, reward terms
  models/                 shared features, batching, attention; policy/, hmm/, reward/
  rl/                     rollouts, rewards, the RLOO and DPG trainer
  ingest/                 MusicXML, MIDI, MuseScore, PDF, and monophonic audio into a melody
  render/                 MIDI, WAV, MP3, MusicXML, PDF, and MuseScore output
  tools/                  listening tools
  harmonizer.py           harmonizing a melody, or only chosen bars, with a released model
notebooks/results.ipynb   tables and figures of the results
```

`data/` and `runs/` are not tracked.

## Limitations

- **1 listener.** The choices reflect one person's ear; they show what that listener
  preferred, not a general preference.
- **Genre is not an input.** Whether a chord outside the key is welcome depends on genre, and
  the model cannot know it.
- **Where, not how often.** RL raised how often the policy leaves the key, not how well it
  chooses where; single draws of the RL models need redrawing more often than π_ref.
- **32-bar context.** One call covers up to 32 bars; longer pieces go in overlapping windows,
  so structure across a whole song, such as repeated verses, is not modeled.
- **Training data** are user transcriptions of songs of every genre, not a curated set of
  good accompaniments.

## License and references

The code is MIT licensed. The model weights and everything derived from the dataset are
CC BY-NC-SA 3.0, following the Hooktheory data. Data are not tracked in this repository.

- Donahue, Thickstun, Liang. "Melody Transcription via Generative Pre-Training." ISMIR 2022.
  [arXiv:2212.01884](https://arxiv.org/abs/2212.01884). Source of the data release.
- Yeh et al. "Automatic Melody Harmonization with Triad Chords: A Comparative Study."
  [arXiv:2001.02360](https://arxiv.org/abs/2001.02360). Metrics and the HMM baseline.
- Ji, Yang, Luo, Li. "RL-Chord: CLSTM-Based Melody Harmonization Using Deep Reinforcement
  Learning." IEEE Transactions on Neural Networks and Learning Systems, 2024.
  [doi:10.1109/TNNLS.2023.3248793](https://doi.org/10.1109/TNNLS.2023.3248793). Local reward.
- Ahmadian et al. "Back to Basics: Revisiting REINFORCE Style Optimization for Learning from
  Human Feedback in LLMs." [arXiv:2402.14740](https://arxiv.org/abs/2402.14740). RLOO.
- Parshakova, Andreoli, Dymetman. "Distributional Reinforcement Learning for Energy-Based
  Sequential Models." [arXiv:1912.08517](https://arxiv.org/abs/1912.08517). DPG.
- Khalifa, Elsahar, Dymetman. "A Distributional Approach to Controlled Text Generation."
  ICLR 2021. [arXiv:2012.11635](https://arxiv.org/abs/2012.11635).
- Korbak, Elsahar, Kruszewski, Dymetman. "Controlling Conditional Language Models without
  Catastrophic Forgetting." [arXiv:2112.00791](https://arxiv.org/abs/2112.00791).
  Conditional DPG.
- Go et al. "Aligning Language Models with Preferences through f-divergence Minimization."
  [arXiv:2302.08215](https://arxiv.org/abs/2302.08215). Comparing KL directions toward the
  same target.
- Zeng et al. "MusicBERT: Symbolic Music Understanding with Large-Scale Pre-Training."
  [arXiv:2106.05630](https://arxiv.org/abs/2106.05630). Features of the discarded rewards.
