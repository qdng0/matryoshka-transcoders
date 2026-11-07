# Matryoshka Transcoders and Hierarchy Misalignment

Code for the NeurIPS 2026 paper **"Matryoshka Transcoders and Hierarchy Misalignment: When SAE Absorption Protection Does Not Transfer"** by Quoc Dat Ngo and Hung Nguyen.

Matryoshka training reduces feature absorption in sparse autoencoders (SAEs). We ask whether that protection carries over to transcoders, which read the MLP input and reconstruct the MLP output. This repository contains the models, training launchers and two-hook SAEBench evaluation code used to train and compare Matryoshka, TopK, BatchTopK, ReLU and JumpReLU transcoders on Gemma-2-2B, together with the bridge controls, mechanism diagnostics and interventions reported in the paper.

## Layout

```text
config/config.json   Main grid and robustness-check settings used in the paper
config/runs/         Exact config.json of each main-grid, seed, width, prefix-ratio, Llama and joint-loss run
src/model/           Matryoshka and flat transcoders, Matryoshka SAE, activation store
src/training/        Training loops for transcoders and SAEs
src/scripts/         Launchers for every training run in the paper
src/utils/           Config defaults, results manager, logging helpers
eval/sae_bench/      SAEBench evaluations adapted to two hooks (absorption, sparse probing,
                     AutoInterp, RAVEL, SCR, FVU, loss recovered)
eval/sae_bench_vendor/  The parts of SAEBench that the adapted evaluations import
eval/diagnostics/    Mechanism diagnostics (source and target letter probes, group target
                     covariance, ridge skip fit, skip letter-signal decomposition)
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

The pinned versions match our environment (Python 3.12, PyTorch 2.6 with CUDA 12.4).

You also need

- Hugging Face access to `google/gemma-2-2b`, plus `meta-llama/Llama-3.1-8B` if you want to reproduce the cross-model check (`huggingface-cli login`),
- the training stream `monology/pile-uncopyrighted`, which is streamed from the Hub,
- an OpenAI API key in `OPENAI_API_KEY`, only for the AutoInterp evaluation.

Weights & Biases logging is on by default. Run `wandb login` first, or set `WANDB_MODE=offline` or `WANDB_MODE=disabled`.

Run every command from the repository root. Training outputs (configs, checkpoints, metrics) are written under `results/` by default, and `--results-base-dir` changes the location in the launchers that support it.

## Training

### Main grid

The main grid trains five architectures on Gemma-2-2B layers 0, 6, 12, 18 and 24 at width 16,384 for 500M tokens with seed 49 (see `config/config.json`). Each run takes 122,070 optimizer steps and fits on a single 24 GB GPU.

```bash
python -m src.scripts.train_matryoshka_all_layers \
  --layers 0,6,12,18,24 --dataset-path monology/pile-uncopyrighted \
  --num-tokens 500000000 --dict-size 16384 --aux-penalty 0.03125 \
  --dataset-shuffle-buffer 0

for layer in 0 6 12 18 24; do
  python -m src.scripts.train_flat_topk_default_setup --layer "$layer" --aux-penalty 0.03125
  python -m src.scripts.train_flat_batchtopk_default_setup --layer "$layer" --aux-penalty 0.03125 \
    --dataset-shuffle-buffer 0
done

python -m src.scripts.train_flat_relu_default_setup --layer 12 --act-l1-coef 9.5e-4
python -m src.scripts.train_flat_jumprelu_default_setup --layer 12 --act-l0-coef 3.9e-4 \
  --target-l0 111 --dict-size 16384 --seq-len 1024 --lr 7e-5 --model-batch-size 8
python -m src.scripts.train_flat_jumprelu_default_setup --layer 24 --act-l0-coef 5.6e-3 \
  --target-l0 96 --dict-size 16384 --seq-len 1024 --lr 7e-5 --model-batch-size 8
```

The per-layer sparsity targets (L0 of 115, 95, 111, 99 and 96) are filled in automatically from the layer index. The ReLU and JumpReLU coefficients are tuned per layer to hit the same targets. The exact settings for every cell, including those coefficients, are in `config/runs/default2/training/<architecture>/<layer>/config.json`.

**Note on data order.** The main grid was trained on the unshuffled Pile stream. The activation store now applies a seeded streaming shuffle with a buffer of 10,000 by default, so pass `--dataset-shuffle-buffer 0` to match the paper's data order. Launchers without that flag (TopK, ReLU, JumpReLU) take the default from `get_default_cfg()` in `src/utils/config.py`.

### Robustness checks

```bash
# Seeds 50 and 51 at layer 12 (same data order, new initialization)
python -m src.scripts.train_matryoshka_all_layers --layers 12 --seed 50 --dataset-shuffle-buffer 0
python -m src.scripts.train_flat_batchtopk_default_setup --layer 12 --seed 50 --dataset-shuffle-buffer 0

# Data-order control (seed 51 with a seeded stream shuffle)
python -m src.scripts.train_matryoshka_all_layers --layers 12 --seed 51 --dataset-shuffle-buffer 10000
python -m src.scripts.train_flat_batchtopk_default_setup --layer 12 --seed 51 --dataset-shuffle-buffer 10000

# Width sweep (8k, 16k, 32k at layers 6, 12 and 18). Matryoshka groups scale dyadically with --dict-size.
python -m src.scripts.train_matryoshka_all_layers --layers 6,12,18 --dict-size 32768 --dataset-shuffle-buffer 0
python -m src.scripts.train_flat_batchtopk_default_setup --layer 18 --dict-size 32768 --dataset-shuffle-buffer 0

# Prefix-ratio schedule of Bussmann et al. instead of the dyadic one
python -m src.scripts.train_matryoshka_all_layers --layers 12 \
  --group-sizes 512,1024,2048,4096,8704 --dataset-shuffle-buffer 0

# Llama-3.1-8B cross-model check (width 32k, L0 80, layers 8, 16 and 24, bf16)
python -m src.scripts.train_matryoshka_all_layers_llama32 --model meta-llama/Llama-3.1-8B \
  --layers 8,16,24 --dict-size 32768 --top-k 80 --model-dtype bfloat16 --dtype bfloat16
for layer in 8 16 24; do
  python -m src.scripts.train_flat_batchtopk_default_setup --model-name meta-llama/Llama-3.1-8B \
    --layer "$layer" --dict-size 32768 --top-k 80 --model-dtype bfloat16 --dtype bfloat16
done
```

### SAE bridge controls

`train_default2_sae_positive_control` trains Matryoshka and BatchTopK SAEs on the same recipe. Use `--site mlp_in` for the source-space x to x control and `--site mlp_out` for the target-space y to y control.

```bash
python -m src.scripts.train_default2_sae_positive_control --arch matryoshka --layers 12 --site mlp_in
python -m src.scripts.train_default2_sae_positive_control --arch batchtopk  --layers 12 --site mlp_out
```

### Layer-12 interventions

```bash
# Groupwise BatchTopK routing inside the Matryoshka transcoder
python -m src.scripts.train_matryoshka_all_layers --layers 12 --batchtopk-mode groupwise --dataset-shuffle-buffer 0

# Frozen output-SAE decoder bridge: freeze D_y and b_y from a trained y to y Matryoshka SAE
python -m src.scripts.train_frozen_output_sae_decoder_bridge --decoder-arch matryoshka --layer 12 \
  --decoder-config-path <y_sae_run>/config.json --decoder-checkpoint-path <y_sae_run>/checkpoints/final.pt

# Skip residual: fit a frozen ridge skip y_skip = xA + c, then train on y - y_skip
python -m eval.diagnostics.fit_ridge_skip --layer 12 --out <skip_dir>/skip_l12.pt
python -m src.scripts.train_default2_skip_transcoder --arch matryoshka --layer 12 \
  --skip-mode fixed --skip-path <skip_dir>/skip_l12.pt
```

### Joint x + y prefix loss

`--joint-source-weight w` adds a train-time source decoder head. Every Matryoshka prefix (or the single flat dictionary) then also reconstructs the MLP input x, scaled by Var(y)/Var(x) so that w = 1 gives both terms equal weight. The head is discarded at evaluation time, because `encode` and `decode` never use it.

```bash
# Matryoshka + joint loss (w = 1 and w = 0.5)
python -m src.scripts.train_matryoshka_all_layers --layers 6,12,18 --joint-source-weight 1.0 --dataset-shuffle-buffer 0
python -m src.scripts.train_matryoshka_all_layers --layers 12 --joint-source-weight 0.5 --dataset-shuffle-buffer 0

# BatchTopK + joint loss control
python -m src.scripts.train_flat_batchtopk_default_setup --layer 12 --joint-source-weight 1.0 --dataset-shuffle-buffer 0
```

## Evaluation

Every evaluation takes a finished run directory that holds `config.json` and `checkpoints/final.pt`.

```bash
# First-letter absorption and k-sparse probing (two-hook protocol)
python -m eval.sae_bench.absorption.main --model_name gemma-2-2b \
  --config_path <run>/config.json --output_folder results/absorption

# Reconstruction (y-FVU), eval L0 and dead latents
python -m eval.sae_bench.fvu.main --model_name gemma-2-2b \
  --config_path <run>/config.json --output_folder results/fvu

# Sparse probing, AutoInterp, loss recovered
python -m eval.sae_bench.sparse_probing.main_transcoder --help
python -m eval.sae_bench.autointerp.main --help
python -m eval.sae_bench.loss_recovered.main --help

# RAVEL and SCR for one run
python eval/sae_bench/ravel/main_transcoder_grid.py --model_name gemma-2-2b \
  --config_path <run>/config.json --output_folder results/ravel
python eval/sae_bench/scr_and_tpp/main_transcoder_grid.py --model_name gemma-2-2b --perform_scr \
  --config_path <run>/config.json --output_folder results/scr_and_tpp
```

`eval/rerun_default2_evals.py` runs RAVEL and SCR over all 25 main-grid cells (`--evals ravel scr`, with `--cuda_devices 0,1` to use two GPUs). It looks for each cell at `results/default2/training/<architecture>/<layer>/`, so copy or link the finished runs there first. The training launchers write to dated folders under `results/training/`.

The mechanism diagnostics live in `eval/diagnostics/`. `source_target_letter_probe.py` compares first-letter probes on the source and target activations. `matryoshka_group_target_covariance.py` measures how target covariance and clean letter features are spread across Matryoshka groups. `skip_letter_signal_decomposition.py` splits the first-letter signal between the linear skip and the sparse path. Each script documents its inputs with `--help`.

## Acknowledgements

The training code started from Bart Bussmann's [matryoshka_sae](https://github.com/bartbussmann/matryoshka_sae) and [BatchTopK](https://github.com/bartbussmann/BatchTopK) implementations. The evaluations adapt [SAEBench](https://github.com/adamkarvonen/SAEBench) (Karvonen et al.) to transcoders. The absorption metric follows Chanin et al. Models are loaded through [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens). See `THIRD_PARTY_NOTICES.md`.

## Citation

```bibtex
@inproceedings{ngo2026matryoshka,
  title     = {Matryoshka Transcoders and Hierarchy Misalignment: When {SAE} Absorption Protection Does Not Transfer},
  author    = {Ngo, Quoc Dat and Nguyen, Hung},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```

## License

Code written for this project is released under the MIT License (see `LICENSE`). Files derived from third-party projects keep their original terms, as listed in `THIRD_PARTY_NOTICES.md`.
