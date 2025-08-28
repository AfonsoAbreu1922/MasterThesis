from datetime import datetime
from os.path import abspath, dirname, join
import os
import hydra
import sys
import torch
import matplotlib.pyplot as plt
from skimage.transform import resize
import json
from sklearn.metrics import roc_auc_score
import numpy as np
import torch.nn as nn

# -------------------------------------------------
# Pathing + setup
# -------------------------------------------------
sys.path.append(abspath(join(dirname('.'), "../../")))

from src.modules.experiment_execution import setup
setup.disable_warning_messages()
setup.enforce_deterministic_behavior()
setup.set_precision(level="high")

from src.modules.data.dataloader.preprocessed_dataloader import PreprocessedDataLoader
from src.modules.experiment_execution.config import experiment_execution_config
from src.modules.models_and_frameworks.fixmatch.fixmatch_components.model import FixMatchModel

# ======= CONFIG MANUAL (substitui se quiseres via YAML) =======
sup_ckpt = "/nas-ctm01/homes/ajpinheiro/MasterThesis/pytorch_lightning_dl_pipeline_template/experiment_results/experiment_34/version_1/datafold_5/models/mod=FxMc-exp=34-ver=1-epoch=53-var=val_auroc=0.836.ckpt"
semi_ckpt = "/nas-ctm01/homes/ajpinheiro/MasterThesis/pytorch_lightning_dl_pipeline_template/experiment_results/experiment_34/version_2/datafold_5/models/mod=FxMc-exp=34-ver=2-epoch=227-var=val_auroc=0.886.ckpt"
datafold_idx = [4]  # usa [] para todos os folds
# =============================================================

# =========================
# Helpers
# =========================
def disable_warnings_and_precision():
    setup.disable_warning_messages()
    setup.enforce_deterministic_behavior()
    setup.set_precision(level="high")


def resolve_dotted_attr(root, dotted: str):
    """Acede a um atributo por string com pontos, e índices numéricos."""
    obj = root
    for tok in dotted.split('.'):
        if tok.strip() == "":
            continue
        obj = obj[int(tok)] if tok.lstrip("-").isdigit() else getattr(obj, tok)
    return obj


def get_last_conv_module(root_module: nn.Module) -> nn.Module:
    """Tenta encontrar a última camada Conv2d para usar como alvo no Grad-CAM."""
    last = None
    for m in root_module.modules():
        if isinstance(m, nn.Conv2d):
            last = m
    if last is None:
        raise RuntimeError("Não foi possível encontrar uma camada nn.Conv2d no modelo.")
    return last


def tolerant_load_state_dict(model: nn.Module, state: dict, strict_guard: bool = True):
    """Carrega pesos tolerando prefixes e ignora se incompatibilidade for grande."""
    sd = state.get("state_dict", state)
    cleaned = {}
    for k, v in sd.items():
        k2 = k
        for pref in ("model.", "backbone.", "net."):
            if k2.startswith(pref):
                k2 = k2[len(pref):]
        cleaned[k2] = v

    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    n_missing, n_unexp = len(missing), len(unexpected)

    if n_missing or n_unexp:
        print(f"[warn] Missing keys: {n_missing}")
        if n_missing: print(f"       e.g.: {missing[:5]}")
        print(f"[warn] Unexpected keys: {n_unexp}")
        if n_unexp: print(f"       e.g.: {unexpected[:5]}")

    # Guard: se a incompatibilidade for muito grande, não usar estes pesos
    if strict_guard and (n_missing > 50 and n_unexp > 50):
        print("[warn] Checkpoint parece de outra arquitetura. Vou IGNORAR este load.")
        return False
    return True


# ---------- batch parsing robusto ----------
def _find_first_tensor(obj):
    if isinstance(obj, torch.Tensor):
        return obj
    if isinstance(obj, dict):
        for v in obj.values():
            t = _find_first_tensor(v)
            if t is not None:
                return t
    if isinstance(obj, (list, tuple)):
        for v in obj:
            t = _find_first_tensor(v)
            if t is not None:
                return t
    return None

def _find_first_tensor_4d_or_3d(obj):
    t = _find_first_tensor(obj)
    if t is None:
        return None
    if t.ndim == 4:
        return t
    if t.ndim == 3:
        return t.unsqueeze(0)
    if t.ndim == 2:
        t = t.unsqueeze(0).unsqueeze(1)  # [H,W] -> [1,1,H,W]
        return t
    return None

def _find_labels(obj, B):
    cand = []
    def collect(o):
        if isinstance(o, torch.Tensor):
            cand.append(o)
        elif isinstance(o, dict):
            for v in o.values(): collect(v)
        elif isinstance(o, (list, tuple)):
            for v in o: collect(v)
    collect(obj)

    for t in cand:
        if t.ndim == 1 and t.shape[0] == B:
            return t
        if t.ndim == 2 and t.shape[0] == B and t.shape[1] == 1:
            return t.squeeze(1)
    return None

def _find_filenames(obj, B):
    def collect_lists(o):
        out = []
        if isinstance(o, (list, tuple)):
            out.append(o)
            for v in o:
                out.extend(collect_lists(v))
        elif isinstance(o, dict):
            for v in o.values():
                out.extend(collect_lists(v))
        return out

    for seq in collect_lists(obj):
        if len(seq) == B and all(isinstance(s, str) for s in seq):
            return list(seq)
    return None

def ensure_tensor_input(batch0):
    t = _find_first_tensor_4d_or_3d(batch0)
    if t is None:
        raise KeyError("Não encontrei nenhum tensor no batch. Estrutura não reconhecida.")
    if t.ndim == 4:
        return t
    if t.ndim == 3:
        return t.unsqueeze(0)
    raise KeyError("Tensor encontrado, mas não consigo convertê-lo para [B,C,H,W].")

def parse_batch(batch):
    inputs = ensure_tensor_input(batch)
    B = inputs.shape[0]
    labels = _find_labels(batch, B)
    filenames = _find_filenames(batch, B) or [f"img_{i}" for i in range(B)]
    return inputs, labels, filenames

def save_cam_only(base_path, name_prefix, cam_array):
    os.makedirs(base_path, exist_ok=True)
    cam = np.asarray(cam_array, dtype=np.float32)
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-6)
    # percentual stretch para aumentar contraste quando é “quase plano”
    lo, hi = np.percentile(cam, [5, 95])
    if hi > lo:
        cam = np.clip((cam - lo) / (hi - lo), 0, 1)
    plt.imsave(join(base_path, f"{name_prefix}_heat.png"), cam, cmap='jet')


# ---------- guardar imagens com aspeto termal ----------
def save_images(base_path, name_prefix, image_tensor, cam_array):
    os.makedirs(base_path, exist_ok=True)
    original_path = join(base_path, f"{name_prefix}_original.png")
    zoom_path = join(base_path, f"{name_prefix}_zoom.png")
    cam_overlay_path = join(base_path, f"{name_prefix}_gradcam.png")

    original_np = image_tensor.squeeze().detach().cpu().numpy()
    plt.imsave(original_path, original_np, cmap='gray')

    zoomed_resized = resize(
        original_np, (512, 512), order=0, mode='reflect',
        anti_aliasing=False, preserve_range=True
    )
    plt.imsave(zoom_path, zoomed_resized.astype(original_np.dtype), cmap='gray')

    cam = np.asarray(cam_array, dtype=np.float32)
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-6)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(original_np, cmap='gray')
    ax.imshow(cam, cmap='jet', alpha=0.5, vmin=0.0, vmax=1.0)
    ax.axis('off')
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.savefig(cam_overlay_path, dpi=300)
    plt.close()


# =========================
# Grad-CAM
# =========================
class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model.eval()
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, _input, output):
            self.activations = output.detach()

        def backward_hook(module, _grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate_cam(self, image_tensor: torch.Tensor, class_idx: int | None = None):
        x = image_tensor.clone().detach().requires_grad_(True)
        self.model.eval()

        logits = self.model(x)
        if class_idx is None:
            class_idx = int(logits.argmax(dim=1).item())

        self.model.zero_grad(set_to_none=True)
        logits[:, class_idx].backward()

        pooled_gradients = self.gradients.mean(dim=(2, 3), keepdim=True)  # [B,C,1,1]
        cam = (pooled_gradients * self.activations).sum(dim=1, keepdim=True)  # [B,1,H',W']
        cam = torch.relu(cam)

        cam = torch.nn.functional.interpolate(
            cam, size=x.shape[2:], mode='bilinear', align_corners=False
        )
        cam = cam.squeeze().detach().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() + 1e-6)
        return cam


# =========================
# Main
# =========================
@hydra.main(version_base=None, config_path="../../config_files", config_name="main")
def run_explainability(config):
    print("Seed:", config.seed_value)
    disable_warnings_and_precision()
    setup.set_seed(config.seed_value)

    # Preparação de diretórios/ids (igual ao pipeline)
    experiment_execution_config.set_experiment_id(config)
    experiment_execution_config.delete_key(config, key='hyperparameter_grid_based_execution')
    experiment_execution_config.set_experiment_version_id(config)
    experiment_execution_config.set_paths(config)
    setup.create_experiment_dir(config.experiment_execution.paths.experiment_version_dir_path)

    # DataLoader
    dataloader = PreprocessedDataLoader(
        config=config.data,
        experiment_execution_paths=config.experiment_execution.paths
    )
    kfold_dataloaders = dataloader.get_dataloaders()

    # Config FixMatch
    fixmatch_hparams = (
        config.models_and_frameworks
              .model_and_framework_pipeline
              .pytorch_lightning_model_and_framework
              .hyperparameters
    )
    model_cfg = fixmatch_hparams.model

    # Instanciar modelos
    sup_model = FixMatchModel(model_cfg)
    semi_model = FixMatchModel(model_cfg)

    # Carregar pesos (serão ignorados se incompatíveis)
    sup_loaded  = tolerant_load_state_dict(sup_model,  torch.load(sup_ckpt,  map_location="cpu"))
    semi_loaded = tolerant_load_state_dict(semi_model, torch.load(semi_ckpt, map_location="cpu"))
    if not sup_loaded or not semi_loaded:
        print("[info] A correr com pesos INICIAIS (sem checkpoint compatível).")
    sup_model.eval()
    semi_model.eval()

    # Camada alvo (intermédia, ResNet18)
    target_layer_path = "model.layer3"   # <— instead of "model.layer4.1.conv1"
    sup_target  = resolve_dotted_attr(sup_model,  target_layer_path)
    semi_target = resolve_dotted_attr(semi_model, target_layer_path)


    sup_cam = GradCAM(sup_model, sup_target)
    semi_cam = GradCAM(semi_model, semi_target)

    # Iterar folds
    n_folds = len(kfold_dataloaders['test'])
    selected_folds = set(datafold_idx) if datafold_idx else set(range(1, n_folds + 1))

    for datafold_id in range(1, n_folds + 1):
        if datafold_id not in selected_folds:
            continue

        test_loader = kfold_dataloaders['test'][datafold_id - 1]

        # Agregadores por fold
        fold_labels, fold_sup_preds, fold_semi_preds = [], [], []

        # helper para processar um batch
        def process_batch(batch, b_idx: int):
            nonlocal fold_labels, fold_sup_preds, fold_semi_preds
            inputs, labels, filenames = parse_batch(batch)
            B = inputs.shape[0]
            for i in range(B):
                x = inputs[i].unsqueeze(0)  # [1,C,H,W]
                name = filenames[i] if isinstance(filenames, list) else str(filenames[i])

                cam_sup_arr  = sup_cam.generate_cam(x)
                cam_semi_arr = semi_cam.generate_cam(x)
                '''
                print(f"  CAM(supervised)  min={cam_sup_arr.min():.4f} max={cam_sup_arr.max():.4f}")
                print(f"  CAM(semisup)     min={cam_semi_arr.min():.4f} max={cam_semi_arr.max():.4f}")
                save_cam_only(img_dir, "supervised",     cam_sup_arr)
                save_cam_only(img_dir, "semisupervised", cam_semi_arr)
'''
                base_dir = join(config.experiment_execution.paths.experiment_version_dir_path, "explainability")
                img_dir  = join(base_dir, f"fold_{datafold_id}", name)
                os.makedirs(img_dir, exist_ok=True)

                # usar sempre a mesma função (garante aspeto termal)
                save_images(img_dir, "supervised",     x, cam_sup_arr)
                save_images(img_dir, "semisupervised", x, cam_semi_arr)

                with torch.no_grad():
                    sup_pred  = int(sup_model(x).argmax(dim=1).item())
                    semi_pred = int(semi_model(x).argmax(dim=1).item())
                label_val = int(labels[i].item()) if labels is not None else -1

                fold_labels.append(label_val)
                fold_sup_preds.append(sup_pred)
                fold_semi_preds.append(semi_pred)

                metadata = {
                    "file_base": name,
                    "fold_idx": datafold_id,
                    "batch_idx": b_idx,
                    "img_idx": i,
                    "supervised_prediction": sup_pred,
                    "semisupervised_prediction": semi_pred,
                    "label": label_val,
                    "timestamp": datetime.now().isoformat(),
                    "input_shape": list(x.shape),
                }
                with open(join(img_dir, "sample_metadata.json"), "w") as f:
                    json.dump(metadata, f, indent=2)

                print(f"[fold {datafold_id}] batch {b_idx} img {i} -> {name}")

        # Prime pass + processamento de todos os batches
        test_iter = iter(test_loader)
        try:
            first_batch = next(test_iter)
        except StopIteration:
            continue
        # estabilizar shapes
        img_prime, _, _ = parse_batch(first_batch)
        _ = sup_model(img_prime)
        _ = semi_model(img_prime)
        # processar todos
        process_batch(first_batch, b_idx=0)
        for b_idx, batch in enumerate(test_iter, start=1):
            process_batch(batch, b_idx=b_idx)

        # Métricas por fold
        labels_np    = np.array(fold_labels)
        sup_preds_np = np.array(fold_sup_preds)
        semi_preds_np= np.array(fold_semi_preds)

        def compute_metrics(preds, labels):
            mask = labels >= 0
            if not mask.any():
                return {"auc": float("nan"), "acc": float("nan"),
                        "precision": float("nan"), "recall": float("nan"),
                        "f1": float("nan"), "tp": 0, "tn": 0, "fp": 0, "fn": 0}
            L = labels[mask]; P = preds[mask]
            tp = int(((P == 1) & (L == 1)).sum())
            tn = int(((P == 0) & (L == 0)).sum())
            fp = int(((P == 1) & (L == 0)).sum())
            fn = int(((P == 0) & (L == 1)).sum())
            acc = (tp + tn) / max((tp + tn + fp + fn), 1)
            precision = tp / max((tp + fp), 1)
            recall = tp / max((tp + fn), 1)
            f1 = 2 * precision * recall / max((precision + recall), 1e-8)
            try:
                auc = roc_auc_score(L, P)
            except Exception:
                auc = float("nan")
            return {"auc": float(auc), "acc": float(acc), "precision": float(precision),
                    "recall": float(recall), "f1": float(f1),
                    "tp": tp, "tn": tn, "fp": fp, "fn": fn}

        fold_dir = join(config.experiment_execution.paths.experiment_version_dir_path, "explainability", f"fold_{datafold_id}")
        os.makedirs(fold_dir, exist_ok=True)

        metrics_sup  = compute_metrics(sup_preds_np,  labels_np)
        metrics_semi = compute_metrics(semi_preds_np, labels_np)
        summary = {
            "fold_idx": datafold_id,
            "timestamp": datetime.now().isoformat(),
            "supervised_ckpt": sup_ckpt,
            "semisupervised_ckpt": semi_ckpt,
            **{f"supervised_{k}": v for k, v in metrics_sup.items()},
            **{f"semisupervised_{k}": v for k, v in metrics_semi.items()},
        }
        with open(join(fold_dir, "overall_metadata.json"), "w") as f:
            json.dump(summary, f, indent=2)

    print("Grad-CAM concluído.")


if __name__ == "__main__":
    run_explainability()
