# src/data_loader.py
import os
import numpy as np

# ─── Configuração ────────────────────────────────────────────────────────────
DATASETS_ROOT = "/mnt/windows/data/rduarte/gdrive"  # ajuste ao subir no GitHub
UDLF_BIN      = None
# ─────────────────────────────────────────────────────────────────────────────

# Mapa dataset_name -> nome da pasta no disco (quando difere do nome lógico).
_FOLDER_OVERRIDES: dict[str, str] = {
    "corel": "corel5k",
}

_DATASET_CONFIGS: dict[str, dict[str, str | None]] = {
    "flowers": {
        "list_name":   "flowers_lists.txt",
        "class_name":  None,                       # labels geradas por i // 80
        "resnet_name": "cnn-last_linear-resnet152.npz",
        "senet_name":  "cnn-last_linear-senet154.npz",
        "vit_name":    "features_vit-b16_flowers.npy",
    },
    "cub200": {
        "list_name":   "cub200_lists.txt",
        "class_name":  "cub200_classes.txt",       # formato nome:classe
        "resnet_name": "cub200_resnet152.npy",
        "senet_name":  "cub200_senet.npy",
        "vit_name":    "cub200_vit-b16.npy",
    },
    "pets": {
        "list_name":   "pets_lists.txt",
        "class_name":  "pets_classes.txt",         # formato nome:classe
        "resnet_name": "pets_resnet.npy",
        "senet_name":  "pets_senet.npy",
        "vit_name":    "pets_vit16.npy",
    },
    "corel": {
        "list_name":   "corel5k_lists.txt",
        "class_name":  "corel5k_classes.txt",      # formato nome:classe
        "resnet_name": "cnn-last_linear-resnet152.npz",
        "senet_name":  None,
        "vit_name":    "features_VIT-B16_corel5k.npy",
    },
}


def get_dataset_paths(dataset_name: str) -> dict:
    """Retorna os caminhos absolutos dos arquivos de um dataset.

    Estrutura esperada em disco::

        <DATASETS_ROOT>/<folder>/
            features/
                <features de todos os extratores>
            <list_name>
            <class_name>

    ``folder`` normalmente é ``dataset_name``, exceto quando houver override
    em ``_FOLDER_OVERRIDES`` (ex.: ``corel`` -> ``corel5k``).

    Args:
        dataset_name: Um dos valores em ``_DATASET_CONFIGS``.

    Returns:
        Dict com ``data_dir``, ``feat_dir``, ``list_path``, ``class_path``,
        ``resnet_path``, ``senet_path``, ``vit_path``, ``bin_path``.
        Entradas sem arquivo associado são ``None``.

    Raises:
        ValueError: Se ``dataset_name`` não for suportado.
    """
    if dataset_name not in _DATASET_CONFIGS:
        raise ValueError(
            f"Dataset '{dataset_name}' não suportado. "
            f"Opções: {list(_DATASET_CONFIGS.keys())}"
        )

    folder_name = _FOLDER_OVERRIDES.get(dataset_name, dataset_name)
    data_dir = os.path.join(DATASETS_ROOT, folder_name)
    feat_dir = os.path.join(data_dir, "features")
    cfg = _DATASET_CONFIGS[dataset_name]

    def _feat_path(name: str | None) -> str | None:
        return os.path.join(feat_dir, name) if name else None

    def _meta_path(name: str | None) -> str | None:
        return os.path.join(data_dir, name) if name else None

    return {
        "data_dir":    data_dir,
        "feat_dir":    feat_dir,
        "list_path":   _meta_path(cfg.get("list_name")),
        "class_path":  _meta_path(cfg.get("class_name")),
        "resnet_path": _feat_path(cfg.get("resnet_name")),
        "senet_path":  _feat_path(cfg.get("senet_name")),
        "vit_path":    _feat_path(cfg.get("vit_name")),
        "bin_path":    UDLF_BIN,
    }


def load_data(
    dataset_name: str,
    feat_extractor: str,
    paths: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """Carrega features e labels do dataset selecionado.

    Args:
        dataset_name: Identificador do dataset.
        feat_extractor: Chave do extrator (``'resnet'``, ``'senet'`` ou
            ``'vit'``).
        paths: Dicionário retornado por :func:`get_dataset_paths`.

    Returns:
        Tupla ``(features, labels)`` como arrays NumPy.

    Raises:
        ValueError: Se extrator/dataset inválido ou formato não reconhecido.
    """
    # 1. Carregamento unificado das features.
    feat_path = paths.get(f"{feat_extractor}_path")
    if not feat_path:
        raise ValueError(
            f"Extrator '{feat_extractor}' não suportado para '{dataset_name}'."
        )

    if feat_path.endswith(".npz"):
        features = np.load(feat_path)["features"]
    elif feat_path.endswith(".npy"):
        features = np.load(feat_path)
    else:
        raise ValueError(f"Formato não reconhecido: {feat_path}")

    # 2. Carregamento das labels (varia por dataset).
    class_path = paths["class_path"]
    list_path  = paths["list_path"]

    if dataset_name == "flowers":
        # Flowers17: 17 classes x 80 imagens, agrupadas sequencialmente.
        with open(list_path, "r") as f:
            n = sum(1 for _ in f)
        labels = [i // 80 for i in range(n)]

    elif dataset_name in ("cub200", "pets", "corel"):
        # Formato comum: cada linha é "nome_imagem:classe".
        labels = []
        with open(class_path, "r") as f:
            for line in f:
                _, cls = line.strip().split(":")
                labels.append(int(cls))

    else:
        raise ValueError(f"Dataset '{dataset_name}' sem suporte a labels.")

    return np.array(features), np.array(labels)