"""Release checks for representative public scripts and documentation."""

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]

EXPECTED_SCRIPTS = [
    *(f"scripts/curvLoRA/{name}" for name in (
        "_lib.sh",
        "01-tofu-3b-f10-simnpo-er-sweep.sh",
        "02-tofu-main-3b-8b.sh",
        "03-muse-books-main.sh",
        "04-tofu-3b-f10-projection-ablation.sh",
        "05-qwen3-4b-tofu-main.sh",
        "06-tofu-3b-f10-simnpo-retain-ratio.sh",
    )),
    *(f"scripts/oplora/{name}" for name in (
        "_lib.sh", "01-tofu-3b-8b.sh", "02-tofu-qwen3-4b.sh", "03-muse-books.sh"
    )),
    *(f"scripts/curvDelta/{name}" for name in (
        "tofu-3b.sh", "tofu-8b.sh", "tofu-qwen3-4b.sh", "muse-books.sh"
    )),
    "scripts/prepare_vila_init.py",
    "scripts/experiments/26-05-02-tofu-fft-lora-vila-baselines.sh",
    "scripts/experiments/26-06-08-qwen3-4b-tofu-finetune.sh",
    "scripts/experiments/26-06-08-qwen3-4b-tofu-npo-baselines.sh",
    "scripts/experiments/26-09-06-muse-books-crispedit-trainers-lr-sweep.sh",
]


class ReleaseValidation(unittest.TestCase):
    def test_representative_script_layout(self):
        self.assertEqual(
            [path for path in EXPECTED_SCRIPTS if not (ROOT / path).is_file()], []
        )
        for removed in (
            "scripts/comparison_lib.sh",
            "scripts/paper_grid.sh",
            "scripts/baselines",
            "scripts/finetune",
        ):
            self.assertFalse((ROOT / removed).exists(), removed)

    def test_qwen_finetuning_uses_5e_5(self):
        text = (ROOT / "scripts/experiments/26-06-08-qwen3-4b-tofu-finetune.sh").read_text(encoding="utf-8")
        self.assertRegex(text, r"(?:finetune_lr|learning_rate)=['\"]?5e-5")

    def test_qwen_npo_baselines_cover_fft_lora_vila(self):
        text = (ROOT / "scripts/experiments/26-06-08-qwen3-4b-tofu-npo-baselines.sh").read_text(encoding="utf-8")
        self.assertRegex(text, r"for adapt_tag in fft lora vila")
        self.assertIn("trainer=NPO", text)
        for value in (
            "lora_r=32",
            "lora_alpha=64",
            "lora_dropout=0.05",
            "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
            "num_train_epochs=10",
            "weight_decay=0.01",
            "warmup_epochs=1.0",
            "optim=adamw_8bit",
            "effective_batch_size=32",
        ):
            self.assertIn(value, text)
        self.assertIn(
            "gradient_accumulation_steps=$((effective_batch_size / (per_device_train_batch_size * num_processes)))",
            text,
        )

    def test_muse_crispedit_has_paper_trainer_settings(self):
        text = (ROOT / "scripts/experiments/26-09-06-muse-books-crispedit-trainers-lr-sweep.sh").read_text(encoding="utf-8")
        for value in (
            "data_split=Books",
            "num_train_epochs=5",
            "npo_beta=0.01",
            "simnpo_gamma=1.0",
            "graddiff_lrs=(1e-5 2e-5 5e-5)",
            "npo_lrs=(2e-5 5e-5 1e-4)",
            "simnpo_lrs=(5e-5 1e-4 2e-4)",
        ):
            self.assertIn(value, text)
        self.assertNotRegex(text.lower(), r"muse[-_ ]news")

    def test_download_cli_and_optional_muse_builder_are_books_only(self):
        download = (ROOT / "common_utils/download_cli.sh").read_text(encoding="utf-8")
        for repo_id in (
            "open-unlearning/tofu_Llama-3.1-8B-Instruct_full",
            "open-unlearning/tofu_Llama-3.1-8B-Instruct_retain90",
            "open-unlearning/tofu_Llama-3.1-8B-Instruct_retain95",
            "muse-bench/MUSE-books_target",
            "muse-bench/MUSE-books_retrain",
            "locuslab/TOFU",
            "muse-bench/MUSE-Books",
        ):
            self.assertIn(repo_id, download)
        self.assertNotRegex(download, r"HF_TOKEN=['\"]hf_")

        muse = (ROOT / "scripts/muse_unlearn.sh").read_text(encoding="utf-8")
        self.assertIn("data_split=Books", muse)
        self.assertNotRegex(muse.lower(), r"news|scalability|sustainability")

    def test_tofu_baseline_uses_unbalanced_vila(self):
        text = (ROOT / "scripts/experiments/26-05-02-tofu-fft-lora-vila-baselines.sh").read_text(encoding="utf-8")
        self.assertIn("lora_init.rebalance=false", text)
        self.assertNotIn("lora_init.rebalance=true", text)
        self.assertIn("GradDiff", text)
        self.assertNotIn("GradAscent", text)

        muse = (ROOT / "scripts/muse_unlearn.sh").read_text(encoding="utf-8")
        self.assertIn("trainers=(GradDiff NPO SimNPO)", muse)
        self.assertNotIn("GradAscent", muse)

    def test_curvlora_and_curvdelta_public_names_and_parameters(self):
        curvlora = "\n".join(
            path.read_text(encoding="utf-8")
            for path in (ROOT / "scripts/curvLoRA").glob("*.sh")
        )
        curvdelta = "\n".join(
            path.read_text(encoding="utf-8")
            for path in (ROOT / "scripts/curvDelta").glob("*.sh")
        )
        self.assertIn("ER_VALS=(0.5 0.6 0.7 0.8 0.9)", curvlora)
        self.assertIn("peft=curvLoRA", curvlora)
        self.assertIn("curvLoRA=default", curvlora)
        self.assertIn("peft=curvDelta", curvdelta)
        self.assertIn("curvDelta=default", curvdelta)

    def test_public_additions_are_english_and_private_data_free(self):
        paths = [ROOT / "README.md", ROOT / "common_utils"]
        paths.extend(ROOT / path for path in EXPECTED_SCRIPTS)
        for item in paths:
            files = item.rglob("*") if item.is_dir() else (item,)
            for path in files:
                if not path.is_file() or path.suffix not in {
                    ".sh", ".py", ".md", ".yaml", ".yml", ".json"
                }:
                    continue
                text = path.read_text(encoding="utf-8")
                with self.subTest(path=str(path.relative_to(ROOT))):
                    self.assertNotRegex(text, r"[\u4e00-\u9fff]")
                    self.assertNotRegex(text.lower(), r"feishu|sftp|webhook")
                    self.assertNotRegex(text, r"/mnt/|/home/|/root/|[a-zA-Z]:[\\/](?:users|project)")
                    self.assertNotRegex(text, r"hf_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9]{20,}")
                    self.assertNotRegex(text, r"curvature_lora|curvature_dense")

    def test_readme_is_minimal_and_contains_requested_figure_text(self):
        text = (ROOT / "README.md").read_text(encoding="utf-8")
        self.assertTrue(text.startswith(
            "# CurvDelta: Low-Curvature Delta Modeling For Surgical Learning and Unlearning"
        ))
        self.assertIn("based on OpenUnlearning", text)
        self.assertIn("assets/overview.png", text)
        self.assertIn("Unconstrained optimization reaches low task loss", text)
        self.assertIn("For simplicity, only representative scripts are shown", text)
        self.assertIn("## Abstract", text)
        self.assertIn("## Data and Checkpoints", text)
        self.assertIn("TOFU_EVAL.json", text)
        self.assertIn("bash common_utils/download_cli.sh", text)
        self.assertIn("## Main Unlearning Results", text)
        self.assertIn("https://ojs.aaai.org/index.php/AAAI/article/view/40703", text)
        self.assertIn("https://icml.cc/virtual/2026/poster/62453", text)
        self.assertIn("https://arxiv.org/abs/2508.21300", text)
        self.assertIn("bash scripts/", text)
        self.assertTrue((ROOT / "overview.pdf").is_file())
        self.assertTrue((ROOT / "assets/overview.png").is_file())


if __name__ == "__main__":
    unittest.main()
