"""Convert HFValid + TotalSegmentator coarse masks into TWO nnU-Net v2 raw datasets (SimpleITK).

Dataset A "CTOnly"  (model that segments the femur from the raw CT, with side):
    images{Tr,Ts}/PatXXX_0000.nii.gz   CT (copy of the original)
    labels{Tr,Ts}/PatXXX.nii.gz        0 = background, 1 = TARGETS[0] (femur_left),
                                       2 = TARGETS[1] (femur_right); the class of the modelled
                                       side comes from HFValid_Metadata.csv

Dataset B "Guided"  (boundary refinement model):
    images{Tr,Ts}/PatXXX_0000.nii.gz   CT (copy of the original)
    images{Tr,Ts}/PatXXX_0001.nii.gz   coarse TotalSegmentator mask of TARGETS[0] (0/1)
    images{Tr,Ts}/PatXXX_0002.nii.gz   coarse TotalSegmentator mask of TARGETS[1] (0/1)
    labels{Tr,Ts}/PatXXX.nii.gz        0 = background, 1 = femur (side-agnostic)

Train ids go to imagesTr/labelsTr (the only folders nnU-Net trains on); test ids go to
imagesTs/labelsTs, which nnU-Net ignores. Predict on them with
    nnUNetv2_predict -i <dataset>/imagesTs -o <out> -d <id> -c 3d_fullres -f <fold>
and compare against labelsTs.

Sources (nothing is modified or deleted, files are only read / copied):
    CT:          <HFValid>/Subjects/PatXXX/PatXXX.nii.gz
    STL:         <HFValid>/Subjects/PatXXX/PatXXX.stl   (-> utils.stl_to_ct_mask)
    metadata:    <HFValid>/HFValid_Metadata.csv          (or --metadata)
    coarse mask: <pred_path>/PatXXX/ml_segmentation.nii  (multilabel, same grid as the CT)

`HFValid`, `pred_path` and `TARGETS` are imported from config.py.
A case whose CT / coarse mask / label shapes or geometry do not match is skipped with a warning.

Subject ids are integers 1..101 (as in HFValid_Metadata.csv); "Pat012" / "012" are accepted too.
The nnU-Net case identifiers are always the folder names (Pat001, ...).

Usage (CLI):
    python convert_hfvalid_to_nnunet.py --train-ids 1 2 3 ... --test-ids 4 5 ...
    python convert_hfvalid_to_nnunet.py --split-file split.json   # {"train": [...], "test": [...]}
Usage (Python):
    convert_dataset(train_ids, test_ids, out_root)
Destination = $nnUNet_raw unless --out-root is given.
"""
import argparse
import csv
import json
import os
import re
import shutil
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
from totalsegmentator.map_to_binary import class_map

from config import HFValid, pred_path, TARGETS  # noqa: E402
from utils import stl_to_ct_mask  # noqa: E402

VALID_IDS = range(1, 102)  # Pat001 ... Pat101
SEG_NAMES = ("ml_segmentation.nii", "ml_segmentation.nii.gz")
SIDE_TO_TARGET = {"R": "femur_right", "L": "femur_left"}  # HFValid_Metadata.csv: R / L


def subject_name(i: int) -> str:
    return f"Pat{i:03d}"


def parse_id(x) -> int:
    """1, "1", "012" or "Pat012" -> int."""
    if isinstance(x, (int, np.integer)):
        return int(x)
    m = re.fullmatch(r"(?:pat)?0*(\d+)", str(x).strip(), flags=re.I)
    if not m:
        raise ValueError(f"cannot parse subject id {x!r}")
    return int(m.group(1))


def find_seg(pid: str) -> Path | None:
    for name in SEG_NAMES:
        p = Path(pred_path) / pid / name
        if p.exists():
            return p
    return None


def label_ids(task: str) -> dict[str, int]:
    """Map TARGETS names -> integer labels of the multilabel TotalSegmentator output."""
    name_to_id = {v: k for k, v in class_map[task].items()}
    missing = [t for t in TARGETS if t not in name_to_id]
    if missing:
        raise KeyError(f"{missing} not found in class_map['{task}']")
    return {t: name_to_id[t] for t in TARGETS}


def load_sides(csv_path, id_col=None, side_col=None) -> dict[int, str]:
    """subject id -> TARGETS name of the modelled femur (from HFValid_Metadata.csv)."""
    df = pd.read_csv(csv_path, sep=None, engine="python")
    df.columns = [str(c).strip() for c in df.columns]

    def pick(given, pattern, what):
        if given:
            return given
        cands = [c for c in df.columns if re.search(pattern, c.lower())]
        if len(cands) != 1:
            raise SystemExit(f"Cannot identify the {what} column among {list(df.columns)}; "
                             f"pass it explicitly.")
        return cands[0]

    id_col = pick(id_col, r"\bid\b", "subject ID")
    side_col = pick(side_col, r"\bside\b", "femur side")
    print(f"Metadata columns: id='{id_col}', side='{side_col}'")

    sides = {}
    for _, row in df.iterrows():
        s = str(row[side_col]).strip().upper()[:1]
        if s not in SIDE_TO_TARGET:
            raise SystemExit(f"Unexpected side value '{row[side_col]}' for subject {row[id_col]}")
        sides[int(str(row[id_col]).removeprefix("Pat"))] = SIDE_TO_TARGET[s]
    return sides


def read_info(path: Path) -> sitk.ImageFileReader:
    """Header only (size/spacing/origin/direction), does not load the voxels."""
    r = sitk.ImageFileReader()
    r.SetFileName(str(path))
    r.ReadImageInformation()
    return r


def geometry_mismatch(a, b) -> str | None:
    """a, b: sitk.Image or ImageFileReader. Returns a message if they are not on the same grid."""
    if tuple(a.GetSize()) != tuple(b.GetSize()):
        return f"shape mismatch {tuple(a.GetSize())} vs {tuple(b.GetSize())} (x, y, z)"
    for name in ("Spacing", "Origin", "Direction"):
        if not np.allclose(getattr(a, f"Get{name}")(), getattr(b, f"Get{name}")(), atol=1e-3):
            return f"{name.lower()} differs"
    return None


def save_mask(arr: np.ndarray, dst: Path, ref) -> None:
    """arr in sitk order (z, y, x); geometry copied from ref (sitk.Image)."""
    img = sitk.GetImageFromArray(arr.astype(np.uint8))
    img.CopyInformation(ref)
    sitk.WriteImage(img, str(dst), useCompression=True)


def convert_case(pid, ct_src, seg_src, stl_src, side_target, labels, A, B):
    """A / B = (images_dir, labels_dir) of the CT-only / guided dataset for this case's split.
    Returns voxel counts, or None if the case was skipped."""
    ct_info = read_info(ct_src)
    seg_img = sitk.ReadImage(str(seg_src))

    # --- geometry checks: warn and skip instead of writing an unusable case ---
    msg = geometry_mismatch(ct_info, seg_img)
    if msg:
        warnings.warn(f"{pid}: CT vs coarse mask: {msg} -> case skipped")
        return None

    # stl_to_ct_mask: sitk.Image or ndarray (assumed sitk order z, y, x)
    lab = stl_to_ct_mask(str(stl_src), ct_path=str(ct_src))
    if isinstance(lab, sitk.Image):
        msg = geometry_mismatch(ct_info, lab)
        label = sitk.GetArrayFromImage(lab)
    else:
        label = np.asarray(lab)
        ct_shape = tuple(ct_info.GetSize())[::-1]
        msg = None if label.shape == ct_shape else f"shape mismatch {label.shape} vs {ct_shape} (z, y, x)"
    if msg:
        warnings.warn(f"{pid}: CT vs STL label: {msg} -> case skipped")
        return None
    label = label.astype(np.uint8)
    if label.max() > 1:
        warnings.warn(f"{pid}: STL label is not binary (max={label.max()})")

    seg = sitk.GetArrayFromImage(seg_img)
    coarse = {t: (seg == labels[t]) for t in TARGETS}
    counts = {t: int(m.sum()) for t, m in coarse.items()}
    counts["label"] = int(label.sum())

    # sanity check: the TS mask of the modelled side should be the bigger one
    other = [t for t in TARGETS if t != side_target]
    if any(counts[side_target] < counts[o] for o in other):
        warnings.warn(f"{pid}: metadata says {side_target}, but the coarse mask is larger on "
                      f"the other side ({counts}) -> possible L/R swap, check this case")

    side_id = TARGETS.index(side_target) + 1  # class id in the CT-only dataset
    written = []
    try:
        # dataset A: CT-only
        save_mask(label * np.uint8(side_id), A[1] / f"{pid}.nii.gz", seg_img)
        written.append(A[1] / f"{pid}.nii.gz")
        # dataset B: guided
        for k, t in enumerate(TARGETS, 1):
            dst = B[0] / f"{pid}_{k:04d}.nii.gz"
            save_mask(coarse[t], dst, seg_img)
            written.append(dst)
        save_mask(label, B[1] / f"{pid}.nii.gz", seg_img)
        written.append(B[1] / f"{pid}.nii.gz")
        # CT copies (originals untouched)
        for images_dir in (A[0], B[0]):
            dst = images_dir / f"{pid}_0000.nii.gz"
            shutil.copy2(ct_src, dst)
            written.append(dst)
    except Exception:
        for p in written:  # remove only the partial outputs of this case
            p.unlink(missing_ok=True)
        raise
    return counts


def write_dataset_json(dataset_dir: Path, channel_names, labels, description):
    n_cases = len(list((dataset_dir / "imagesTr").glob("*_0000.nii.gz")))
    with open(dataset_dir / "dataset.json", "w") as f:
        json.dump({"channel_names": channel_names, "labels": labels,
                   "numTraining": n_cases, "file_ending": ".nii.gz",
                   "description": description}, f, indent=2)
    return n_cases


def convert_dataset(train_ids, test_ids, out_root, ct_only_id=501, guided_id=502,
                    name_prefix="HFValid", metadata=None, id_col="PatientID", side_col=None,
                    task="total", overwrite=False):
    train_ids = [parse_id(i) for i in train_ids]
    test_ids = [parse_id(i) for i in (test_ids if test_ids is not None else [])]
    if len(set(train_ids)) != len(train_ids) or len(set(test_ids)) != len(test_ids):
        raise ValueError("duplicate ids in train_ids or test_ids")
    if set(train_ids) & set(test_ids):
        raise ValueError(f"ids in both train and test: {sorted(set(train_ids) & set(test_ids))}")
    bad = [i for i in train_ids + test_ids if i not in VALID_IDS]
    if bad:
        raise ValueError(f"ids outside {VALID_IDS.start}..{VALID_IDS.stop - 1}: {bad}")
    if not out_root:
        raise ValueError("out_root is empty (set $nnUNet_raw or pass --out-root)")
    if not set(SIDE_TO_TARGET.values()) <= set(TARGETS):
        raise ValueError(f"TARGETS must contain {sorted(SIDE_TO_TARGET.values())}")

    labels = label_ids(task)
    print("Label ids:", labels)
    sides = load_sides(metadata or Path(HFValid) / "HFValid_Metadata.csv", id_col, side_col)

    dirs, ds = {}, {}  # ds[key][split] = (images_dir, labels_dir)
    for key, ds_id, suffix in (("A", ct_only_id, "CTOnly"), ("B", guided_id, "Guided")):
        d = Path(out_root) / f"Dataset{ds_id:03d}_{name_prefix}{suffix}"
        dirs[key] = d
        ds[key] = {"Tr": (d / "imagesTr", d / "labelsTr")}
        if test_ids:
            ds[key]["Ts"] = (d / "imagesTs", d / "labelsTs")
        for images_dir, labels_dir in ds[key].values():
            images_dir.mkdir(parents=True, exist_ok=True)
            labels_dir.mkdir(parents=True, exist_ok=True)

    report, problems = [], []
    jobs = [(i, "Tr") for i in train_ids] + [(i, "Ts") for i in test_ids]
    for i, split in jobs:
        pid = subject_name(i)
        A, B = ds["A"][split], ds["B"][split]
        subj_dir = Path(HFValid) / "Subjects" / pid
        ct_src = subj_dir / f"{pid}.nii.gz"
        stl_src = subj_dir / f"{pid}.stl"
        seg_src = find_seg(pid)

        # leakage guard: leftovers of an earlier run with a different split
        other = "Ts" if split == "Tr" else "Tr"
        if any((ds[k][other][0] / f"{pid}_0000.nii.gz").exists() for k in ds if other in ds[k]):
            warnings.warn(f"{pid} is now in '{split}' but files from an earlier run exist in "
                          f"'{other}' -> remove them manually to avoid train/test leakage")

        missing = [f"CT {ct_src}" if not ct_src.exists() else None,
                   f"STL {stl_src}" if not stl_src.exists() else None,
                   f"coarse mask in {Path(pred_path) / pid}" if seg_src is None else None,
                   f"side in metadata (id {i})" if i not in sides else None]
        missing = [m for m in missing if m]
        if missing:
            warnings.warn(f"{pid}: not found: {', '.join(missing)} -> case skipped")
            problems.append(pid)
            continue

        expected = [A[0] / f"{pid}_0000.nii.gz", A[1] / f"{pid}.nii.gz", B[1] / f"{pid}.nii.gz"]
        expected += [B[0] / f"{pid}_{k:04d}.nii.gz" for k in range(len(TARGETS) + 1)]
        if not overwrite and all(p.exists() for p in expected):
            print(f"{pid} [{split}]: already converted, skipping")
            continue

        try:
            counts = convert_case(pid, ct_src, seg_src, stl_src, sides[i], labels, A, B)
        except Exception as e:  # keep going, report at the end
            warnings.warn(f"{pid}: failed with {e!r} -> case skipped")
            problems.append(pid)
            continue
        if counts is None:
            problems.append(pid)
            continue

        report.append({"case": pid, "split": split, "side": sides[i],
                       **{f"{k}_voxels": v for k, v in counts.items()}})
        flag = "  <-- empty label!" if counts["label"] == 0 else ""
        print(f"{pid} [{split}] ({sides[i]}): "
              + ", ".join(f"{k}={v}" for k, v in counts.items()) + flag)

    # per-case voxel counts: handy for spotting TotalSegmentator / voxelisation failures
    if report:
        with open(dirs["B"] / "conversion_report.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=report[0].keys())
            w.writeheader()
            w.writerows(report)

    n_a = write_dataset_json(
        dirs["A"], {"0": "CT"},
        {"background": 0, **{t: k for k, t in enumerate(TARGETS, 1)}},
        "CT only; classes by side of the modelled femur")
    # "noNorm" keeps the binary coarse-mask channels at 0/1; meaning is in "description"
    n_b = write_dataset_json(
        dirs["B"], {"0": "CT", **{str(k): "noNorm" for k in range(1, len(TARGETS) + 1)}},
        {"background": 0, "femur": 1},
        "channel 0 = CT; " + "; ".join(f"channel {k} = coarse mask {t}"
                                       for k, t in enumerate(TARGETS, 1)))

    n_ts = len(list(ds["B"]["Ts"][0].glob("*_0000.nii.gz"))) if test_ids else 0
    print(f"\nCT-only: {n_a} train cases in {dirs['A']}\n"
          f"Guided:  {n_b} train cases in {dirs['B']}\n"
          f"Test cases (imagesTs/labelsTs in each dataset): {n_ts}")
    if problems:
        print(f"{len(problems)} skipped case(s): " + ", ".join(problems))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-ids", nargs="+", help="e.g. 1 2 3 or Pat001 Pat002")
    ap.add_argument("--test-ids", nargs="+", default=[])
    ap.add_argument("--split-file", help="JSON with keys 'train' and 'test' (lists of ids)")
    ap.add_argument("--ct-only-id", type=int, default=501)
    ap.add_argument("--guided-id", type=int, default=502)
    ap.add_argument("--name-prefix", default="HFValid")
    ap.add_argument("--out-root", default=os.environ.get("nnUNet_raw"),
                    help="defaults to $nnUNet_raw")
    ap.add_argument("--metadata", default=None,
                    help="defaults to <HFValid>/HFValid_Metadata.csv")
    ap.add_argument("--id-col", default=None)
    ap.add_argument("--side-col", default=None)
    ap.add_argument("--task", default="total", help="key of totalsegmentator class_map")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    if args.split_file:
        with open(args.split_file) as f:
            split = json.load(f)
        train_ids, test_ids = split["train"], split.get("test", [])
    elif args.train_ids:
        train_ids, test_ids = args.train_ids, args.test_ids
    else:
        raise SystemExit("Pass --train-ids (and --test-ids) or --split-file")

    convert_dataset(train_ids, test_ids, args.out_root, args.ct_only_id, args.guided_id,
                    args.name_prefix, args.metadata, args.id_col, args.side_col,
                    args.task, args.overwrite)


if __name__ == "__main__":
    main()