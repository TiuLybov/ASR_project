import argparse
import re
from pathlib import Path

from src.metrics.utils import calc_cer, calc_wer


def normalize_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z ]", "", text)
    return text


def main():
    parser = argparse.ArgumentParser(description="Calculate WER and CER")
    parser.add_argument(
        "--gt-dir",
        type=str,
        required=True,
        help="Path to directory with ground truth .txt files",
    )
    parser.add_argument(
        "--pred-dir",
        type=str,
        required=True,
        help="Path to directory with prediction .txt files",
    )
    args = parser.parse_args()

    gt_dir = Path(args.gt_dir)
    pred_dir = Path(args.pred_dir)

    cers = []
    wers = []
    missing = 0

    for gt_file in sorted(gt_dir.glob("*.txt")):
        pred_file = pred_dir / gt_file.name
        if not pred_file.exists():
            missing += 1
            continue

        gt_text = normalize_text(gt_file.read_text().strip())
        pred_text = normalize_text(pred_file.read_text().strip())

        cers.append(calc_cer(gt_text, pred_text))
        wers.append(calc_wer(gt_text, pred_text))

    if len(cers) == 0:
        print("No matching files found.")
        return

    avg_cer = sum(cers) / len(cers)
    avg_wer = sum(wers) / len(wers)

    print(f"Evaluated on {len(cers)} utterances")
    if missing > 0:
        print(f"Missing predictions: {missing}")
    print(f"CER: {avg_cer:.4f} ({avg_cer * 100:.2f}%)")
    print(f"WER: {avg_wer:.4f} ({avg_wer * 100:.2f}%)")


if __name__ == "__main__":
    main()
