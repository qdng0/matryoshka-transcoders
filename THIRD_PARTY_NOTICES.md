# Third-party code

Parts of this repository are derived from other projects. Those files keep their original authorship and terms, and the MIT License in `LICENSE` covers only the code written for this project.

| Path | Derived from | Upstream license |
|---|---|---|
| `eval/sae_bench_vendor/sae_bench/` | Vendored subset of [SAEBench](https://github.com/adamkarvonen/SAEBench) v0.5.0 (Karvonen et al., upstream commit `3a2ec14`), with small patches to the AutoInterp, RAVEL and base eval-output modules. `sae_bench/common/` inside it is our own code | No license file in the upstream repository |
| `eval/sae_bench/` | Two-hook adaptations of the SAEBench evaluations (absorption, sparse probing, AutoInterp, RAVEL, SCR and TPP) | No license file in the upstream repository |
| `eval/sae_bench/ravel/transcoder_mdbm.py`, `eval/sae_bench/ravel/transcoder_intervention.py` | The transcoder-aware RAVEL intervention hook is ported from the [FF-KV](https://github.com/muyo8692/ff-kv-sae) SAEBench fork (arXiv:2510.22332), with a fix that keeps the real MLP output at every position that is not intervened on | No license file in the upstream repository |
| `src/model/`, `src/training/`, `src/utils/` | Started from [matryoshka_sae](https://github.com/bartbussmann/matryoshka_sae) by Bart Bussmann, which builds on [BatchTopK](https://github.com/bartbussmann/BatchTopK) and [SAELens](https://github.com/jbloomAus/SAELens) | matryoshka_sae: no license file. BatchTopK: MIT. SAELens: MIT |

This repository does not redistribute model weights, datasets or trained checkpoints. Gemma-2-2B is used under the Gemma Terms of Use, Llama-3.1-8B under the Llama 3.1 Community License, and the Pile (uncopyrighted subset) under its dataset terms.
