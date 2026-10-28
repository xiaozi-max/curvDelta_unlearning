# CurvDelta: Low-Curvature Delta Modeling For Surgical Learning and Unlearning

This repository contains the unlearning implementation of CurvDelta and CurvLoRA, based on OpenUnlearning ([project](https://github.com/locuslab/open-unlearning)).

![Overview of CurvDelta and CurvLoRA](assets/overview.png)

**Overview.** (a) Unconstrained optimization reaches low task loss but high preservation loss. (b) CurvDelta constructs a low-curvature delta space. The applied delta remains in this space throughout optimization, reaching a different solution with comparable task loss. (c) The base weights and projections remain frozen. The trainable delta is a dense matrix in CurvDelta or LoRA in CurvLoRA. (d) Performance comparison. The dashed line marks the base model's retention average. Our methods achieve the best trade-off between task performance and capability retention. [PDF](overview.pdf)

## Quick Start

```bash
pip install -r requirements.txt
pip install -e .
```

Set the data root and available GPUs before running an experiment. `HF_TOKEN` and `WANDB_API_KEY` are optional and must be supplied through the environment when needed.

```bash
export PROJECT_DATA_ROOT=/path/to/project-data
export GPU_IDS=0,1
export HF_TOKEN=your_huggingface_token
export WANDB_API_KEY=your_wandb_api_key
bash scripts/curvLoRA/01-tofu-3b-f10-simnpo-er-sweep.sh
```

The portable environment and download helpers are under `common_utils/`. Representative CurvLoRA, CurvDelta, OPLoRA, CrispEdit, FFT, LoRA, VILA, and Qwen3-4B fine-tuning scripts are under `scripts/`.

For simplicity, only representative scripts are shown in this repository. Readers can customize their script by modifying relevant variables in corresponding files, including LLM backbone, hyperparameter, fine-tuning method, and unlearning trainer.

## License

This project retains the OpenUnlearning MIT license. See [LICENSE](LICENSE).
