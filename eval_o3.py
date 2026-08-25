"""
eval_o3.py — OOD-Guided Soft-Weighted Inference for PROD

Implements the formula at each linear layer:
    output = W_base · x + w(x) × (W_PROD - W_base) · x
           = (1 - w(x)) × W_base · x  +  w(x) × W_PROD · x

Where w(x) is computed by the OOD detector (RoBERTa + OCSVM + GMM).
Uses PyTorch forward hooks instead of hacked Transformers files.

Usage:
    python eval_o3.py \\
        --base_model codellama/CodeLlama-7b-hf \\
        --prod_model tummitum/PROD_epoch9_lr5e-06 \\
        --test_dataset ./data/codellama/D_test.json \\
        --ood_weights ./ood_checkpoints_codellama_0/ \\
        --ood_base_model microsoft/codebert-base
"""

import os
import re
import json
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import math
import sys
import pickle
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, RobertaTokenizer
from src.ood_model_selector import RobertaForSelector_inference
from scipy.stats import norm
from typing import Union

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:  # noqa: E722
    pass


# ============================================================
# Helper Functions
# ============================================================

def set_seed(seed: int):
    """Fix PRNG seed for reproducable experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def check_api_usage(generated_line, target_api, alias_dict):
    """Kiểm tra xem API sinh ra có khớp với target hay không bằng Regex."""
    for key, value in alias_dict.items():
        if value == target_api:
            pattern = r'\b' + re.escape(key) + r'\b'
            if re.search(pattern, generated_line):
                return True
    return False


# ============================================================
# OOD Weight Computation (from O3 paper)
# ============================================================

def gmm_cdf(x, gmm):
    """Cumulative probability function for GMM."""
    weights = gmm.weights_
    means = gmm.means_.flatten()
    stds = np.sqrt(gmm.covariances_.flatten())
    cdf_vals = [w * norm.cdf(x, mean, std) for w, mean, std in zip(weights, means, stds)]
    return np.sum(cdf_vals)


def cumulative_probability(x, gmm):
    """The cumulative probability for a point."""
    return gmm_cdf(x, gmm)


def symmetric_cumulative_probability(x, x0, gmm):
    """The cumulative probability for the symmetric point."""
    symmetric_x = 2 * x0 - x
    return gmm_cdf(symmetric_x, gmm)


# Global tracking for weight distribution analysis
all_w_res = []
all_w_res_dic = {}


def obtain_weights(input_x, gmm, x0):
    """
    Compute soft weight w(x) from OCSVM score.

    Returns:
        w(x) ∈ {0, (0.3, 0.4], 1.2}
        - 0:   Input is NOT in forget domain → base model only
        - 0.3~0.4: Partial activation
        - 1.2: Input IS in forget domain → strong unlearn
    """
    cp_x = cumulative_probability(input_x, gmm)
    cp_symmetric_x = symmetric_cumulative_probability(input_x, x0, gmm)

    cp_sum = 1 - max(cp_x, cp_symmetric_x) + min(cp_x, cp_symmetric_x)
    scaling_factor = 10
    cp_sum *= scaling_factor
    range_th = 2

    w_res = math.exp(cp_sum - range_th) / (1 + math.exp(cp_sum - range_th))

    if w_res > 0.9:
        w_res = 1.2
    elif w_res <= 0.4 and w_res > 0.3:
        w_res = w_res
    else:
        w_res = 0

    return w_res


# ============================================================
# Delta-W Forward Hook System
# ============================================================

class DeltaWeightManager:
    """
    Manages delta weights (W_PROD - W_base) and forward hooks.

    At each hooked linear layer, the hook computes:
        output = base_output + w(x) × F.linear(input, delta_W, delta_b)

    Which is equivalent to:
        output = (1 - w(x)) × W_base · x + w(x) × W_PROD · x
    """

    def __init__(self):
        self.ood_weight = 0  # scalar or tensor (batch_size,)
        self._hooks = []

    @staticmethod
    def compute_and_register(base_model, prod_model, target_module_names):
        """
        Compute delta_W = W_PROD - W_base for each target linear layer,
        register forward hooks on the base model, and return the manager.

        Args:
            base_model: The base pretrained model (on GPU)
            prod_model: The PROD fine-tuned model (can be on CPU)
            target_module_names: List of module name substrings to target
                                 e.g. ["q_proj", "k_proj", "v_proj", ...]

        Returns:
            DeltaWeightManager instance with hooks registered
        """
        manager = DeltaWeightManager()

        # Build lookup for prod model modules
        prod_modules = dict(prod_model.named_modules())

        hooked_count = 0
        for name, base_module in base_model.named_modules():
            # Check if this is a target linear layer
            if not isinstance(base_module, nn.Linear):
                continue
            if not any(t in name for t in target_module_names):
                continue

            # Get corresponding prod module
            if name not in prod_modules:
                print(f"  WARNING: Module '{name}' not found in PROD model, skipping")
                continue
            prod_module = prod_modules[name]

            # Compute delta weights (on CPU first to save VRAM)
            delta_w = (prod_module.weight.data.cpu() - base_module.weight.data.cpu()).clone()
            delta_b = None
            if base_module.bias is not None and prod_module.bias is not None:
                delta_b = (prod_module.bias.data.cpu() - base_module.bias.data.cpu()).clone()

            # Move to same device and dtype as base module
            base_device = base_module.weight.device
            base_dtype = base_module.weight.dtype
            delta_w = delta_w.to(device=base_device, dtype=base_dtype)
            if delta_b is not None:
                delta_b = delta_b.to(device=base_device, dtype=base_dtype)

            # Register hook
            hook = base_module.register_forward_hook(
                manager._make_hook(delta_w, delta_b)
            )
            manager._hooks.append(hook)
            hooked_count += 1

        print(f"  Registered {hooked_count} delta-W forward hooks")
        return manager

    def _make_hook(self, delta_weight, delta_bias):
        """Create a forward hook closure for a single linear layer."""
        manager_ref = self

        def hook(module, input, output):
            w = manager_ref.ood_weight

            # Fast path: no delta contribution
            if isinstance(w, (int, float)) and w == 0:
                return output

            x = input[0]
            delta_out = F.linear(x, delta_weight, delta_bias)

            # Handle per-sample weight tensor
            if isinstance(w, torch.Tensor):
                w = w.to(device=delta_out.device, dtype=delta_out.dtype)
                if w.dim() >= 1:
                    # Reshape (batch_size,) → (batch_size, 1, 1, ...) for broadcasting
                    w = w.view(-1, *([1] * (delta_out.dim() - 1)))

            return output + w * delta_out

        return hook

    def set_weight(self, w):
        """Set the OOD weight for the current batch."""
        self.ood_weight = w

    def remove_hooks(self):
        """Remove all registered hooks."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='PROD + OOD Soft-Weighted Inference')
    parser.add_argument('--test_dataset', type=str, default="./data/codellama/D_test.json",
                        help='Path to test dataset')
    parser.add_argument('--base_model', type=str, default="codellama/CodeLlama-7b-hf",
                        help='Base pretrained model (before PROD training)')
    parser.add_argument('--prod_model', type=str, default="tummitum/PROD_epoch9_lr5e-06",
                        help='PROD fine-tuned model path')
    parser.add_argument('--ood_base_model', type=str, default="microsoft/codebert-base",
                        help='OOD detector base model (RoBERTa)')
    parser.add_argument('--ood_weights', type=str, default="./ood_checkpoints_codellama_0/",
                        help='Path to OOD checkpoint directory')
    parser.add_argument('--ood_type', type=str, default="_all",
                        help='OOD type(s), underscore-separated')
    parser.add_argument('--ood_setting_name', type=str, default="codellama",
                        help='OOD setting name prefix for checkpoint files')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--load_8bit', action='store_true',
                        help='Load base model in 8-bit quantization to save VRAM')
    args = parser.parse_args()

    set_seed(args.seed)

    # --- Load test data ---
    data_a = json.load(open(args.test_dataset, encoding='utf-8'))
    print(f"Test dataset: {args.test_dataset} ({len(data_a)} examples)")

    # --- Parse OOD types ---
    types = args.ood_type.split("_")
    ood_types = [t for t in types if len(t) > 0]
    ood_type_name = "ocsvm"
    print(f"OOD types: {ood_types}")

    # --- Build OOD weight paths ---
    ood_weight_paths = []
    for t in ood_types:
        o_p = args.ood_weights + f"{args.ood_setting_name}_{t}_ood_{args.ood_setting_name}"
        ood_weight_paths.append(o_p)

    # --- Result file path ---
    prod_name = os.path.basename(args.prod_model.rstrip("/"))
    test_name = os.path.basename(args.test_dataset)
    result_file = f"results_prod_ood_seed{args.seed}_{prod_name}_{test_name}"
    print(f"Result file: {result_file}")

    # ============================================================
    # Stage 1: Load models, compute delta-W, register hooks
    # ============================================================
    print("\n" + "=" * 60)
    print("STAGE 1: Model Loading & Delta-W Computation")
    print("=" * 60)

    print("\n[1a] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, padding_side='left', clean_up_tokenization_spaces=True)
    # Use unk token as pad (standard for Llama generation)
    tokenizer.pad_token_id = 0

    print("[1b] Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        load_in_8bit=args.load_8bit,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    print("[1c] Loading PROD model (to CPU)...")
    prod_model = AutoModelForCausalLM.from_pretrained(
        args.prod_model,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
    )

    print("[1d] Computing delta weights and registering hooks...")
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    manager = DeltaWeightManager.compute_and_register(base_model, prod_model, target_modules)

    # Free PROD model from memory
    del prod_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("[1e] PROD model freed from memory\n")

    # Configure model for generation
    base_model.config.pad_token_id = 0
    base_model.config.bos_token_id = 1
    base_model.config.eos_token_id = 2
    base_model.eval()

    # ============================================================
    # Stage 2: Load OOD detector components
    # ============================================================
    print("=" * 60)
    print("STAGE 2: OOD Detector Loading")
    print("=" * 60)

    ood_tokenizer = RobertaTokenizer.from_pretrained(args.ood_base_model)
    ood_models = []
    ood_clrs = []
    ood_thresholds = []
    ood_x0 = []
    ood_mean_lists = []
    ood_precision_lists = []
    ood_fea_lists = []
    ood_gmm_w_cls = []

    for idx, wp in enumerate(ood_weight_paths):
        roberta_path = wp + f"_roberta_{ood_type_name}"
        ocsvm_path = wp + f"_{ood_type_name}.pkl"
        threshold_path = wp + f"_threshold_{ood_type_name}.json"
        mean_list_path = wp + f"_mean_list_{ood_type_name}.pt"
        precision_list_path = wp + f"_precision_list_{ood_type_name}.pt"
        fea_list_path = wp + f"_fea_list_{ood_type_name}.pt"
        gmm_w_path = wp + f"_gmm_w_{ood_type_name}.pkl"

        print(f"\n  [{idx}] Loading OOD detector: {roberta_path}")
        ood_models.append(
            RobertaForSelector_inference(
                args.ood_base_model, lora_path=roberta_path, projection_dim=100
            ).to(device)
        )

        with open(ocsvm_path, "rb") as f:
            ood_clrs.append(pickle.load(f))
        with open(gmm_w_path, "rb") as f:
            ood_gmm_w_cls.append(pickle.load(f))
        with open(threshold_path) as f:
            threshold = json.load(f)
        ood_thresholds.append(threshold[1])
        ood_x0.append(threshold[0])

        ood_mean_lists.append(torch.load(mean_list_path, map_location=torch.device(device)))
        ood_precision_lists.append(torch.load(precision_list_path, map_location=torch.device(device)))
        ood_fea_lists.append(torch.load(fea_list_path, map_location=torch.device(device)))

    print(f"\n  Loaded {len(ood_models)} OOD detector(s)")

    # ============================================================
    # Stage 3: Inference loop
    # ============================================================
    print("\n" + "=" * 60)
    print("STAGE 3: Inference")
    print("=" * 60 + "\n")

    max_new_tokens = 128
    max_batch_size = 64
    save_every = 200
    MAX_PROMPT_LENGTH = 512

    num_deprecated = 0
    num_replacement = 0
    num_mismatch = 0
    results = []
    outputs_list = []

    for start_idx in tqdm(range(0, len(data_a), max_batch_size)):
        end_idx = min(start_idx + max_batch_size, len(data_a))
        batch = data_a[start_idx:end_idx]

        # --- Build prompts for CodeLlama ---
        batch_prompts = []
        batch_meta = []
        for example in batch:
            code_context = example["probing input new"]
            encoded_context = tokenizer.encode(code_context, add_special_tokens=False)
            if len(encoded_context) > MAX_PROMPT_LENGTH - 200:
                encoded_context = encoded_context[-(MAX_PROMPT_LENGTH - 200):]
                code_context = tokenizer.decode(encoded_context)

            prompt_text = (
                f"Complete and output the next line for the following Python function:\n"
                f"```python\n"
                f"{code_context}"
            )
            batch_prompts.append(prompt_text)
            batch_meta.append({
                "code_context": code_context,
                "deprecated_api": example.get("deprecated api", []),
                "replacement_api": example.get("replacement api", ""),
                "alias_dict": example.get("alias dict", {})
            })

        # --- OOD Scoring ---
        ood_texts = [
            example.get('function', example.get('probing input new', ''))
            for example in batch
        ]
        ood_input = ood_tokenizer(
            ood_texts, padding='max_length', truncation=True,
            max_length=512, return_tensors="pt"
        )
        cur_batch_size = len(batch)
        max_ood_per_sample = np.zeros(cur_batch_size)

        for i in range(len(ood_weight_paths)):
            mah_score = ood_models[i].get_unsup_Mah_score_s(
                ood_input, ood_mean_lists[i], ood_precision_lists[i], ood_fea_lists[i]
            )[:, 1:]
            test_score = ood_clrs[i].score_samples(mah_score)
            w_ood = np.array([
                obtain_weights(s, ood_gmm_w_cls[i], ood_x0[i]) for s in test_score
            ])
            max_ood_per_sample = np.maximum(max_ood_per_sample, w_ood)

        # Log per-sample weights
        for w in max_ood_per_sample:
            all_w_res.append(w)
            dic_key = str(w)[:5]
            if dic_key in all_w_res_dic:
                all_w_res_dic[dic_key] += 1
            else:
                all_w_res_dic[dic_key] = 1

        print("ood_weight per sample: ", max_ood_per_sample)

        # --- Set OOD weight via delta-W manager ---
        ood_weight_tensor = torch.tensor(max_ood_per_sample, dtype=torch.bfloat16).to(device)
        manager.set_weight(ood_weight_tensor)

        # --- Generate ---
        inputs = tokenizer(
            batch_prompts, padding=True, return_tensors="pt",
            truncation=True, max_length=MAX_PROMPT_LENGTH
        )
        input_ids = inputs["input_ids"].to(device)

        with torch.no_grad():
            generation_output = base_model.generate(
                input_ids=input_ids,
                return_dict_in_generate=True,
                max_new_tokens=max_new_tokens,
                temperature=1.0,
                do_sample=False,
            )

        # --- Extract generated text ---
        input_length = input_ids.shape[1]
        generated_tokens = generation_output.sequences[:, input_length:]
        generated_texts = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

        # --- Free VRAM ---
        del generation_output, input_ids, inputs, ood_input
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # --- Evaluate generated outputs ---
        for idx, generated_text in enumerate(generated_texts):
            meta = batch_meta[idx]

            clean_text = generated_text.replace("```python", "").replace("```", "").strip()
            first_line_generated = clean_text.split('\n')[0]

            last_line_of_prompt = meta["code_context"].split('\n')[-1]
            full_line_to_check = last_line_of_prompt + first_line_generated

            is_replacement = check_api_usage(
                full_line_to_check, meta["replacement_api"], meta["alias_dict"]
            )
            is_deprecated = any(
                check_api_usage(full_line_to_check, api, meta["alias_dict"])
                for api in meta["deprecated_api"]
            )

            record = {
                "prompt": meta["code_context"],
                "deprecated_api": meta["deprecated_api"],
                "replacement_api": meta["replacement_api"],
                "generated_content": full_line_to_check
            }

            if is_replacement:
                num_replacement += 1
            elif is_deprecated:
                num_deprecated += 1
            else:
                num_mismatch += 1

            results.append(record)
            outputs_list.append(full_line_to_check)
            print(f"[dep={is_deprecated}, rep={is_replacement}] {full_line_to_check[:120]}")

        # --- Save periodically ---
        if end_idx % save_every == 0 or end_idx == len(data_a):
            total = len(results)
            print(f"\n{total}/{len(data_a)}, deprecated: {num_deprecated}, "
                  f"replacement: {num_replacement}, mismatch: {num_mismatch}, "
                  f"saving to {result_file}\n")
            data = {
                'total': total,
                'num_deprecated': num_deprecated,
                'num_replacement': num_replacement,
                'num_mismatch': num_mismatch,
            }
            # Add soft-weight summary at the final save
            if end_idx == len(data_a):
                n = len(all_w_res)
                n_zero = sum(1 for w in all_w_res if w == 0)
                n_mid = sum(1 for w in all_w_res if 0 < w < 1.0)
                n_high = sum(1 for w in all_w_res if w >= 1.0)
                data['soft_weight_summary'] = {
                    'total_samples': n,
                    'mean': float(np.mean(all_w_res)) if n > 0 else 0,
                    'min': float(min(all_w_res)) if n > 0 else 0,
                    'max': float(max(all_w_res)) if n > 0 else 0,
                    'delta_w_not_activated (w=0)': n_zero,
                    'delta_w_partial (0<w<1)': n_mid,
                    'delta_w_full (w>=1)': n_high,
                    'activation_rate': round((n - n_zero) / n * 100, 2) if n > 0 else 0,
                    'weight_distribution': dict(all_w_res_dic),
                }
            data['results'] = results
            with open(result_file, 'w') as f:
                json.dump(data, f, indent=2, separators=(',', ': '))

    # --- Print soft-weight summary ---
    n = len(all_w_res)
    if n > 0:
        n_zero = sum(1 for w in all_w_res if w == 0)
        n_mid = sum(1 for w in all_w_res if 0 < w < 1.0)
        n_high = sum(1 for w in all_w_res if w >= 1.0)
        print("\n" + "=" * 60)
        print("SOFT-WEIGHT SUMMARY (OOD -> Delta-W Activation)")
        print("=" * 60)
        print(f"  Total samples:               {n}")
        print(f"  Mean weight:                 {np.mean(all_w_res):.4f}")
        print(f"  Min / Max:                   {min(all_w_res):.4f} / {max(all_w_res):.4f}")
        print(f"  Delta-W NOT activated (w=0):  {n_zero} ({n_zero / n * 100:.1f}%)")
        print(f"  Delta-W partial (0<w<1):      {n_mid} ({n_mid / n * 100:.1f}%)")
        print(f"  Delta-W full (w>=1.0):        {n_high} ({n_high / n * 100:.1f}%)")
        print(f"  Activation rate:              {(n - n_zero) / n * 100:.1f}%")
        print(f"  Weight distribution:          {dict(all_w_res_dic)}")
        print("=" * 60)

    # Cleanup
    manager.remove_hooks()


if __name__ == "__main__":
    main()
