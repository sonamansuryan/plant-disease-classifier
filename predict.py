import argparse
from typing import Optional

import cv2
import numpy as np
import torch
from torch.amp import autocast

from config import CFG
from model import DiseaseClassifier
from transforms import get_val_transforms


class DiseasePredictor:
    def __init__(self, ckpt_path: str, device: Optional[torch.device] = None):
        self.device = device or CFG.DEVICE

        ckpt = torch.load(ckpt_path, map_location=self.device)
        d2i: dict[str, int] = ckpt["disease2idx"]
        self.idx2disease = {v: k for k, v in d2i.items()}
        num_classes = len(d2i)

        self.model = DiseaseClassifier(ckpt["backbone"], num_classes, dropout=0.0)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.to(self.device).eval()

        self.transform = get_val_transforms(CFG.IMG_SIZE)

    def _load_image(self, path: str) -> np.ndarray:
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"Cannot read: {path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def _probs_to_result(self, probs: np.ndarray) -> dict:
        top3_idx = probs.argsort()[::-1][:3]
        top3 = [{"disease": self.idx2disease[int(i)], "confidence": float(probs[i])} for i in top3_idx]
        return {"disease": top3[0]["disease"], "confidence": top3[0]["confidence"], "top3": top3}

    @torch.no_grad()
    def predict(self, image_path: str) -> dict:
        img    = self._load_image(image_path)
        tensor = self.transform(image=img)["image"].unsqueeze(0).to(self.device)
        with autocast("cuda", enabled=CFG.AMP):
            logits = self.model(tensor)
        probs = logits.softmax(1).squeeze().cpu().float().numpy()
        return self._probs_to_result(probs)

    @torch.no_grad()
    def tta_predict(self, image_path: str) -> dict:
        img = self._load_image(image_path)
        variants = [
            self.transform(image=img)["image"],
            self.transform(image=cv2.flip(img, 1))["image"],
            self.transform(image=cv2.flip(img, 0))["image"],
            self.transform(image=cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE))["image"],
            self.transform(image=cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE))["image"],
        ]
        batch = torch.stack(variants).to(self.device)
        with autocast("cuda", enabled=CFG.AMP):
            logits = self.model(batch)
        probs = logits.softmax(1).mean(0).cpu().float().numpy()
        return self._probs_to_result(probs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--ckpt",  default="outputs/best_model.pth")
    parser.add_argument("--tta",   action="store_true")
    args = parser.parse_args()

    predictor = DiseasePredictor(args.ckpt)
    result    = predictor.tta_predict(args.image) if args.tta else predictor.predict(args.image)

    print(f"\nPrediction : {result['disease'].upper()}")
    print(f"Confidence : {result['confidence']:.2%}")
    print("\nTop-3:")
    for i, r in enumerate(result["top3"], 1):
        print(f"  {i}. {r['disease']:<30} {r['confidence']:.2%}")
