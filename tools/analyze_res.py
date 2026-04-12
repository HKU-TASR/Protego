import os

import yaml
import matplotlib.pyplot as plt
import numpy as np

from protego import BASE_PATH

RES_BASE_PATH = os.path.join(BASE_PATH, "experiments")
#RES_BASE_PATH = os.path.join(BASE_PATH, "results", "eval")

def _safe_load_yaml(file_path: str):
    if not os.path.exists(file_path):
        return None
    try:
        with open(file_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception:
        return None

def _discover_fr_names(user_dir: str, end2end_mode: bool) -> list:
    """Discover FR names from result filenames when cfgs.yaml is missing."""
    fr_names = []
    try:
        for fname in os.listdir(user_dir):
            if end2end_mode and fname.startswith("end2end_eval_res_") and fname.endswith(".yaml"):
                fr_names.append(fname.replace("end2end_eval_res_", "").replace(".yaml", ""))
            elif (not end2end_mode) and fname.startswith("eval_res_") and fname.endswith(".yaml"):
                fr_names.append(fname.replace("eval_res_", "").replace(".yaml", ""))
    except Exception:
        pass
    return sorted(list(set(fr_names)))

def ana_res(eval_name: str, compression: bool, use_lpips: bool, end2end_mode: bool = False):
    """
    Aggregate and visualize evaluation results.

    Args:
    - eval_name: name of the subfolder under results/eval/.
    - compression: whether to read compression robustness results from compression_res_{fr}.yaml.
    - use_lpips: whether to include LPIPS metric (only applicable in non end2end mode).
    - end2end_mode: if True, read end2end_eval_res_{fr}.yaml which only contains 1a/1b/2a/2b.
    """

    name_mapping = {
        '1a': 'Clean Query & Protected DB',
        '2b': 'Protected Query & Clean DB',
        '2a': 'Protected Query & Protected DB',
        '1b': 'FR Baseline Performance'
    }

    res_path = os.path.join(RES_BASE_PATH, eval_name)
    if not os.path.isdir(res_path):
        raise FileNotFoundError(f"Result path not found: {res_path}")

    user_names = [
        name for name in os.listdir(res_path)
        if not name.startswith(('.', '_')) and os.path.isdir(os.path.join(res_path, name))
    ]
    if not user_names:
        raise RuntimeError(f"No user folders found under {res_path}")

    cfg_path = os.path.join(res_path, user_names[0], 'cfgs.yaml')
    cfgs = _safe_load_yaml(cfg_path) or {}

    # Get FR list: prefer cfgs.yaml; otherwise infer from filenames
    test_frs = cfgs.get('eval_fr_names')
    if not test_frs:
        test_frs = _discover_fr_names(os.path.join(res_path, user_names[0]), end2end_mode=end2end_mode)
    if not test_frs:
        raise RuntimeError("Cannot determine FR names. Provide cfgs.yaml with eval_fr_names or ensure result files exist.")

    # Compression methods (might be absent in end2end or when compression=False)
    compression_methods = cfgs.get('eval_compression_methods', [])

    for fr_name in test_frs:
        scenario_keys = ['1a', '1b', '2a', '2b']
        # In end2end mode, no image quality metrics are present
        if end2end_mode:
            metrics_keys = []
            res_filename = f"end2end_eval_res_{fr_name}.yaml"
        else:
            metrics_keys = ['ssim', 'psnr', 'l0']
            if use_lpips:
                metrics_keys.append('lpips')
            res_filename = f"eval_res_{fr_name}.yaml"

        res_recorder = {k: [] for k in scenario_keys + metrics_keys}
        compression_recorders = None
        if compression and not end2end_mode:
            compression_recorders = {
                method: {k: [] for k in scenario_keys}
                for method in compression_methods
            }

        # Aggregate results across user folders
        for user_name in user_names:
            user_path = os.path.join(res_path, user_name)
            res_file = os.path.join(user_path, res_filename)
            if not os.path.exists(res_file):
                continue
            user_res = _safe_load_yaml(res_file) or {}

            for key in scenario_keys:
                if key in user_res:
                    res_recorder[key].append(user_res[key])
            for key in metrics_keys:
                if key in user_res:
                    res_recorder[key].append(user_res[key])

            if compression_recorders is not None:
                comp_file = os.path.join(user_path, f"compression_res_{fr_name}.yaml")
                if os.path.exists(comp_file):
                    compression_res = _safe_load_yaml(comp_file) or {}
                    for method, res_dict in compression_res.items():
                        if method not in compression_recorders:
                            continue
                        for k, v in res_dict.items():
                            if k in compression_recorders[method]:
                                compression_recorders[method][k].append(v)

        print(f"Results for {fr_name}:")
        for key, values in res_recorder.items():
            if key in metrics_keys:
                if values:
                    print(f"{key}: {np.mean(values):.4f}")
            else:
                if values:
                    print(f"{name_mapping.get(key, key)}: {np.mean(values):.4f}")

        if compression_recorders:
            print("Compression Results:")
            for method, recorder in compression_recorders.items():
                print(f"Method: {method}")
                for key, values in recorder.items():
                    if values:
                        print(f"{name_mapping.get(key, key)}: {np.mean(values):.4f}")

    # Bar chart (scenario metrics only)
        bar_keys = ['1a', '2b', '2a']
        bar_labels = [name_mapping[k] for k in bar_keys]
        bar_values = [np.mean(res_recorder[k]) if res_recorder[k] else 0 for k in bar_keys]
        bar_vars = [np.var(res_recorder[k]) if res_recorder[k] else 0 for k in bar_keys]
        bar_mins = [np.min(res_recorder[k]) if res_recorder[k] else 0 for k in bar_keys]
        bar_maxs = [np.max(res_recorder[k]) if res_recorder[k] else 0 for k in bar_keys]

        plt.figure(figsize=(10, 6))
        plt.bar(bar_labels, bar_values, yerr=bar_vars, capsize=6, alpha=0.9)
        for i, (value, var_val, min_val, max_val) in enumerate(zip(bar_values, bar_vars, bar_mins, bar_maxs)):
            plt.text(i, value + 0.01,
                     f"Mean: {value:.2f}\nVar: {var_val:.4f}\nMin: {min_val:.2f} Max: {max_val:.2f}",
                     ha='center', va='bottom', fontsize=9)
        title_prefix = "End2End " if end2end_mode else ""
        plt.title(f"{title_prefix}Evaluation Results for {fr_name.upper()}")
        plt.xlabel("Scenario")
        plt.ylabel("Average Value")
        plt.xticks(rotation=15, ha="right")
        plt.ylim(0, 1)
        baseline_mean = np.mean(res_recorder['1b']) if res_recorder['1b'] else 0
        baseline_var = np.var(res_recorder['1b']) if res_recorder['1b'] else 0
        baseline_min = np.min(res_recorder['1b']) if res_recorder['1b'] else 0
        baseline_max = np.max(res_recorder['1b']) if res_recorder['1b'] else 0
        plt.axhline(y=baseline_mean, color='r', linestyle='--', label='FR Baseline Performance')
        plt.text(len(bar_labels) - 0.1, baseline_mean + 0.01,
                 f"Baseline Mean: {baseline_mean:.2f}\nVar: {baseline_var:.4f}\nMin: {baseline_min:.2f} Max: {baseline_max:.2f}",
                 color='r', ha='right', va='bottom', fontsize=9)

        metric_stats_parts = []
        for mk in metrics_keys:
            if res_recorder[mk]:
                mean_v = np.mean(res_recorder[mk])
                var_v = np.var(res_recorder[mk])
                min_v = np.min(res_recorder[mk])
                max_v = np.max(res_recorder[mk])
                metric_stats_parts.append(
                    f"{mk.upper()}: {mean_v:.4f} (Var: {var_v:.6f}, Min: {min_v:.4f}, Max: {max_v:.4f})"
                )
        if metric_stats_parts:
            plt.figtext(0.5, 0.01, ", ".join(metric_stats_parts), ha='center', fontsize=10)
        plt.tight_layout(rect=[0, 0.05, 1, 1])
        plt.legend()
        fig_name = f"{fr_name}_evaluation_results.png" if not end2end_mode else f"{fr_name}_end2end_evaluation_results.png"
        plt.savefig(os.path.join(res_path, fig_name))
        plt.close()

        # Compression results subplots (only when not end2end and compression=True)
        if compression_recorders:
            method_num = len(compression_methods)
            if method_num > 0:
                plt.figure(figsize=(method_num * 5, 6))
                for idx, (method, recorder) in enumerate(compression_recorders.items()):
                    plt.subplot(1, method_num, idx + 1)
                    bar_values = [np.mean(recorder[k]) if recorder[k] else 0 for k in bar_keys]
                    bar_vars = [np.var(recorder[k]) if recorder[k] else 0 for k in bar_keys]
                    bar_mins = [np.min(recorder[k]) if recorder[k] else 0 for k in bar_keys]
                    bar_maxs = [np.max(recorder[k]) if recorder[k] else 0 for k in bar_keys]
                    plt.bar(bar_labels, bar_values, yerr=bar_vars, capsize=6, alpha=0.9)
                    for i, (value, var_val, min_val, max_val) in enumerate(zip(bar_values, bar_vars, bar_mins, bar_maxs)):
                        plt.text(i, value + 0.01,
                                 f"Mean: {value:.2f}\nVar: {var_val:.4f}\nMin: {min_val:.2f} Max: {max_val:.2f}",
                                 ha='center', va='bottom', fontsize=9)
                    plt.title(f"Compression Method: {method}")
                    plt.xlabel("Scenario")
                    plt.ylabel("Average Value")
                    plt.xticks(rotation=15, ha="right")
                    plt.ylim(0, 1)
                    baseline_mean = np.mean(recorder['1b']) if recorder['1b'] else 0
                    baseline_var = np.var(recorder['1b']) if recorder['1b'] else 0
                    baseline_min = np.min(recorder['1b']) if recorder['1b'] else 0
                    baseline_max = np.max(recorder['1b']) if recorder['1b'] else 0
                    plt.axhline(y=baseline_mean, color='r', linestyle='--', label='FR Baseline Performance')
                    plt.text(len(bar_labels) - 0.1, baseline_mean + 0.01,
                             f"Baseline Mean: {baseline_mean:.2f}\nVar: {baseline_var:.4f}\nMin: {baseline_min:.2f} Max: {baseline_max:.2f}",
                             color='r', ha='right', va='bottom', fontsize=9)
                    plt.legend()
                plt.tight_layout()
                plt.savefig(os.path.join(res_path, f"{fr_name}_compression_results.png"))
                plt.close()

if __name__ == "__main__":
    ######################### Configuration #########################
    # Under results/eval/<eval_name>/ there should be multiple user subfolders, each containing yaml results.
    eval_name = "demographics_race_middle_eastern"  # subfolder name with evaluation results
    compression = False  # set True if compression_res_{fr}.yaml exists
    need_lpips = False    # collect LPIPS only in non end2end mode
    end2end = False       # enable end2end mode: read end2end_eval_res_{fr}.yaml
    #################################################################
    ana_res(eval_name, compression, need_lpips, end2end_mode=end2end)


