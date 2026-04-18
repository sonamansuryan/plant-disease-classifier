from pathlib import Path
from collections import Counter, defaultdict

import cv2
import numpy as np
from torch.utils.data import Dataset, WeightedRandomSampler
import albumentations as A


def build_disease_mapping(root_dirs: list[Path]) -> dict[str, str]:
    all_folders: set[str] = set()
    for root in root_dirs:
        root = Path(root)
        if root.exists():
            for f in root.iterdir():
                if f.is_dir() and not f.name.startswith("__"):
                    all_folders.add(f.name.lower().strip())

    groups: dict[str, list[list[str]]] = defaultdict(list)
    for folder in all_folders:
        groups[folder.split()[0]].append(folder.split())

    folder_to_disease: dict[str, str] = {}
    for folder in all_folders:
        words     = folder.split()
        group     = groups[words[0]]

        plant_len = 0
        if len(group) > 1:
            for col in zip(*group):
                if len(set(col)) == 1:
                    plant_len += 1
                else:
                    break
        else:
            plant_len = 1

        disease = " ".join(words[plant_len:]).strip()
        folder_to_disease[folder] = disease if disease else "healthy"

    return folder_to_disease


def build_label_maps(
    folder_to_disease: dict[str, str],
) -> tuple[dict[str, int], dict[int, str]]:
    diseases = sorted(set(folder_to_disease.values()) - {""})
    disease2idx = {d: i for i, d in enumerate(diseases)}
    idx2disease = {i: d for d, i in disease2idx.items()}
    return disease2idx, idx2disease


class PlantDiseaseDataset(Dataset):
    def __init__(
        self,
        root: Path,
        transform: A.Compose,
        folder_to_disease: dict[str, str],
        disease2idx: dict[str, int],
    ):
        self.transform        = transform
        self.folder_to_disease = folder_to_disease
        self.disease2idx      = disease2idx
        self.samples: list[tuple[Path, int]] = []

        for folder in sorted(root.iterdir()):
            if not folder.is_dir() or folder.name.startswith("__"):
                continue
            disease = folder_to_disease.get(folder.name.lower().strip(), "")
            if not disease or disease not in disease2idx:
                continue
            label = disease2idx[disease]
            for img_path in list(folder.glob("*.jpg")) + list(folder.glob("*.png")):
                self.samples.append((img_path, label))

        print(f"  {root.name}: {len(self.samples)} images, {len(disease2idx)} classes")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        img = cv2.imread(str(path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tensor = self.transform(image=img)["image"]
        return tensor, label

def make_weighted_sampler(dataset: PlantDiseaseDataset) -> WeightedRandomSampler:
    counts = Counter(label for _, label in dataset.samples)
    weights = [1.0 / counts[label] for _, label in dataset.samples]
    return WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)
