import argparse
import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from typing import List, Dict, Any, Optional, Tuple

from DataUtils import FakePartsV2DatasetBase, standardise_predictions, REQUIRED_COLS, collate_skip_none
from networks.resnet import resnet50


class FakePartsV2Dataset(FakePartsV2DatasetBase):
    def __init__(self, *args, **kwargs):
        if 'transform' not in kwargs:
            kwargs['transform'] = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
        super().__init__(*args, **kwargs)


class CNNDetector:
    """
    A wrapper for the CNN-based deepfake detector model.
    Handles model loading, device placement, and batch prediction.
    """

    def __init__(self, model_path: str, device: str = 'cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = self._load_model(model_path)
        self.model.to(self.device)
        self.model.eval()

    def _load_model(self, model_path: str) -> nn.Module:
        model = resnet50(num_classes=1)
        state_dict = torch.load(model_path, map_location='cpu')
        model.load_state_dict(state_dict['model'])
        print(f"Model loaded from {model_path}")
        return model

    @torch.no_grad()
    def predict_batch(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        images = images.to(self.device)
        scores_tensor = self.model(images).squeeze(1)
        scores = torch.sigmoid(scores_tensor)
        preds = (scores >= 0.5).long()
        return scores.cpu(), preds.cpu()


def run_inference(
        data_root: str,
        output_csv: str,
        model_path: str,
        model_name: str,
        batch_size: int = 64,
        num_workers: int = 4,
        device: str = 'cuda',
        done_csv_list: Optional[List[str]] = None
):
    """
    Runs inference on the dataset and saves results to a CSV file.
    """
    print(f"Initializing dataset from: {data_root}")
    dataset = FakePartsV2Dataset(
        data_root=data_root,
        mode='frame',
        model_name=model_name,
        done_csv_list=done_csv_list or [],
        on_corrupt='warn'
    )

    if not dataset:
        print("No samples to process.")
        return

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_skip_none,
        pin_memory=True
    )

    detector = CNNDetector(model_path, device=device)

    results: List[Dict[str, Any]] = []
    for batch in tqdm(data_loader, desc=f"Inference on {model_name}"):
        if batch is None:
            continue
        images, _, metas = batch
        scores, preds = detector.predict_batch(images)

        for i in range(len(images)):
            meta = {key: value[i] for key, value in metas.items()}
            meta['score'] = scores[i].item()
            meta['pred'] = preds[i].item()
            results.append(meta)
        df = standardise_predictions(results)
        results.clear()
        file_exists = os.path.isfile(output_csv)
        df.to_csv(output_csv, mode='a', header=not file_exists, index=False)

    if not results:
        print("No results generated.")
        return


def main():
    parser = argparse.ArgumentParser(description="Run CNN-based detector inference on FakePartsV2.")
    parser.add_argument('--data_root', type=str, required=True, help='Root directory of the dataset.')
    parser.add_argument('--output_csv', type=str, required=True, help='Path to save the output CSV file.')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the pretrained model weights.')
    parser.add_argument('--model_name', type=str, default='CNNDetector', help='Name of the model for reporting.')
    parser.add_argument('--batch_size', type=int, default=2048, help='Batch size for inference.')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading.')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use for inference (e.g., "cuda", "cpu").')
    parser.add_argument('--done_csv_list', nargs='*',
                        help='List of CSV files or directories with CSVs of already processed samples.')

    args = parser.parse_args()

    run_inference(
        data_root=args.data_root,
        output_csv=args.output_csv,
        model_path=args.model_path,
        model_name=args.model_name,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
        done_csv_list=args.done_csv_list
    )


if __name__ == '__main__':
    main()
