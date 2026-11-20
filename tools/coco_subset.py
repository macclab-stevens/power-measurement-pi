"""Build the deterministic COCO val2017 subsets used by the dataset-v2 tools and the Pi runs.

Subsets (all chosen from the val2017 file names sorted ascending, so reruns are identical):
  eval    first 500 images             -> tools/data/coco500/{images,labels} + tools/data/coco500.yaml
  calib   next 200 images (disjoint)   -> tools/data/calib200/images        (INT8 calibration)
  stream  50 of the remaining images, evenly spaced across them -> inputs/stream50/*.jpg
  empty   first remaining image with zero label lines           -> inputs/empty.jpg
  crowded remaining image with the most label lines             -> inputs/crowded.jpg
and inputs/MANIFEST.json (rule + {coco_file, sha256, n_labels} per file), the only committed output.
Downloads, subsets and the YAML live in tools/data/ (gitignored); inputs/*.jpg is gitignored too.

Usage (from tools/):
    uv run python coco_subset.py                       # download zips if missing, build everything
    uv run python coco_subset.py --zips-dir /some/dir  # reuse val2017.zip / coco2017labels.zip there

Images are read straight out of val2017.zip and the label counts out of coco2017labels.zip, so only
the selected files are ever extracted. A val2017 image without a label file has zero labels
(COCO has 48 such images; the labels zip leaves them out).
"""
import argparse
import hashlib
import json
import shutil
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent

IMAGES_URL = "http://images.cocodataset.org/zips/val2017.zip"
LABELS_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco2017labels.zip"
IMAGES_ZIP = "val2017.zip"
LABELS_ZIP = "coco2017labels.zip"
IMAGES_PREFIX = "val2017/"            # member prefix inside val2017.zip
LABELS_PREFIX = "coco/labels/val2017/"  # member prefix inside coco2017labels.zip

N_EVAL, N_CALIB, N_STREAM = 500, 200, 50

SELECTION_RULE = {
    "eval": "first 500 val2017 images by sorted file name",
    "calib": "next 200 images by sorted file name (disjoint from eval)",
    "stream": "50 of the remaining images (sorted), indices floor(i*N/50) for i in 0..49, N=remaining count",
    "empty": "first remaining image (sorted) with zero label lines; missing label file counts as zero",
    "crowded": "remaining image with the most label lines (ties: first in sorted order)",
    "remaining": "all images after the eval and calib subsets; stream/empty/crowded may overlap each other",
}

COCO_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
    "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush",
]


# --------------------------------------------------------------------------
# Pure selection logic
# --------------------------------------------------------------------------
def select_subsets(entries, n_eval=N_EVAL, n_calib=N_CALIB, n_stream=N_STREAM):
    """entries: iterable of (file_name, n_labels). Returns
    {"eval": [...], "calib": [...], "stream": [...], "empty": name, "crowded": name}."""
    items = sorted(entries)
    n_rest = len(items) - n_eval - n_calib
    if n_rest < n_stream:
        raise ValueError(f"need at least {n_eval + n_calib + n_stream} images, got {len(items)}")
    eval_ = [f for f, _ in items[:n_eval]]
    calib = [f for f, _ in items[n_eval:n_eval + n_calib]]
    rest = items[n_eval + n_calib:]
    stream = [rest[i * len(rest) // n_stream][0] for i in range(n_stream)]
    empties = [f for f, n in rest if n == 0]
    if not empties:
        raise ValueError("no empty (zero-label) image among the remaining images")
    crowded = rest[0]
    for e in rest:
        if e[1] > crowded[1]:          # strict: ties keep the first in sorted order
            crowded = e
    return {"eval": eval_, "calib": calib, "stream": stream, "empty": empties[0], "crowded": crowded[0]}


def count_label_lines(text):
    return sum(1 for line in text.splitlines() if line.strip())


def manifest_entry(coco_file, path, n_labels):
    return {"coco_file": coco_file, "sha256": sha256_file(path), "n_labels": n_labels}


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def dataset_yaml(root):
    """Ultralytics dataset YAML for the eval subset; root holds images/ and labels/."""
    names = "\n".join(f"  {i}: {n}" for i, n in enumerate(COCO_NAMES))
    return f"path: {root}\ntrain: images\nval: images\nnames:\n{names}\n"


# --------------------------------------------------------------------------
# Download (resumable) and integrity
# --------------------------------------------------------------------------
def download(url, dest, chunk=1 << 20):
    """Resumable HTTP download into dest (via dest.part, resumed with a Range request)."""
    dest = Path(dest)
    part = dest.with_name(dest.name + ".part")
    dest.parent.mkdir(parents=True, exist_ok=True)
    have = part.stat().st_size if part.exists() else 0
    req = urllib.request.Request(url, headers={"Range": f"bytes={have}-"} if have else {})
    try:
        resp = urllib.request.urlopen(req, timeout=60)
    except urllib.error.HTTPError as e:
        if e.code == 416 and have:       # part is already the whole file
            part.replace(dest)
            return
        raise
    with resp:
        mode = "ab" if have and resp.status == 206 else "wb"   # 200: server ignored Range
        with open(part, mode) as f:
            while block := resp.read(chunk):
                f.write(block)
    part.replace(dest)


def zip_ok(path):
    try:
        with zipfile.ZipFile(path) as z:
            return z.testzip() is None
    except (zipfile.BadZipFile, OSError):
        return False


def ensure_zip(path, url, download=download):
    """Make sure path is an intact zip: keep it if valid, else (re)download and verify."""
    path = Path(path)
    if path.exists() and zip_ok(path):
        return path
    path.unlink(missing_ok=True)
    download(url, path)
    if not zip_ok(path):
        raise RuntimeError(f"zip integrity check failed after download: {path}")
    return path


# --------------------------------------------------------------------------
# Build
# --------------------------------------------------------------------------
def read_label_counts(labels_zip):
    """{stem: n_label_lines} for every label file of val2017 in the labels zip."""
    counts = {}
    with zipfile.ZipFile(labels_zip) as z:
        for name in z.namelist():
            if name.startswith(LABELS_PREFIX) and name.endswith(".txt"):
                counts[Path(name).stem] = count_label_lines(z.read(name).decode())
    return counts


def list_images(images_zip):
    with zipfile.ZipFile(images_zip) as z:
        return sorted(Path(n).name for n in z.namelist()
                      if n.startswith(IMAGES_PREFIX) and n.endswith(".jpg"))


def _extract(zf, member, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    with zf.open(member) as src, open(dest, "wb") as out:
        shutil.copyfileobj(src, out)


def build(data_dir, inputs_dir, images_zip, labels_zip):
    data_dir, inputs_dir = Path(data_dir), Path(inputs_dir)
    images = list_images(images_zip)
    counts = read_label_counts(labels_zip)
    n_labels = {f: counts.get(Path(f).stem, 0) for f in images}
    sel = select_subsets(n_labels.items())

    # Rebuild every output from scratch so a rerun never leaves stale files behind.
    eval_dir, calib_dir = data_dir / "coco500", data_dir / "calib200"
    stream_dir = inputs_dir / "stream50"
    for d in (eval_dir, calib_dir, stream_dir):
        shutil.rmtree(d, ignore_errors=True)
    for f in (inputs_dir / "empty.jpg", inputs_dir / "crowded.jpg"):
        f.unlink(missing_ok=True)

    manifest = {"rule": SELECTION_RULE, "source": {"images": IMAGES_URL, "labels": LABELS_URL},
                "eval": {}, "calib": {}, "stream": {}, "empty": {}, "crowded": {}}

    def record(section, key, coco_file, path):
        manifest[section][key] = manifest_entry(coco_file, path, n_labels[coco_file])

    with zipfile.ZipFile(images_zip) as zi, zipfile.ZipFile(labels_zip) as zl:
        label_members = set(zl.namelist())
        for f in sel["eval"]:
            dest = eval_dir / "images" / f
            _extract(zi, IMAGES_PREFIX + f, dest)
            lab = LABELS_PREFIX + Path(f).stem + ".txt"
            lab_dest = eval_dir / "labels" / (Path(f).stem + ".txt")
            if lab in label_members:
                _extract(zl, lab, lab_dest)
            else:                      # background image: explicit empty label file
                lab_dest.parent.mkdir(parents=True, exist_ok=True)
                lab_dest.write_text("")
            record("eval", f"coco500/images/{f}", f, dest)
        for f in sel["calib"]:
            dest = calib_dir / "images" / f
            _extract(zi, IMAGES_PREFIX + f, dest)
            record("calib", f"calib200/images/{f}", f, dest)
        for f in sel["stream"]:
            dest = stream_dir / f
            _extract(zi, IMAGES_PREFIX + f, dest)
            record("stream", f"inputs/stream50/{f}", f, dest)
        for key in ("empty", "crowded"):
            dest = inputs_dir / f"{key}.jpg"
            _extract(zi, IMAGES_PREFIX + sel[key], dest)
            record(key, f"inputs/{key}.jpg", sel[key], dest)

    (data_dir / "coco500.yaml").write_text(dataset_yaml(eval_dir))
    inputs_dir.mkdir(parents=True, exist_ok=True)
    (inputs_dir / "MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return sel, manifest


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data-dir", type=Path, default=HERE / "data", help="gitignored work dir")
    ap.add_argument("--inputs-dir", type=Path, default=REPO / "inputs", help="Pi inputs dir")
    ap.add_argument("--zips-dir", type=Path, default=None,
                    help="dir holding val2017.zip and coco2017labels.zip "
                         "(default: <data-dir>/downloads; missing/corrupt zips are downloaded)")
    args = ap.parse_args(argv)
    zips = args.zips_dir or args.data_dir / "downloads"
    images_zip = ensure_zip(zips / IMAGES_ZIP, IMAGES_URL)
    labels_zip = ensure_zip(zips / LABELS_ZIP, LABELS_URL)
    sel, manifest = build(args.data_dir, args.inputs_dir, images_zip, labels_zip)
    print(f"eval {len(sel['eval'])}  calib {len(sel['calib'])}  stream {len(sel['stream'])}  "
          f"empty {sel['empty']}  crowded {sel['crowded']} "
          f"({manifest['crowded']['inputs/crowded.jpg']['n_labels']} labels)")
    print(f"wrote {args.inputs_dir / 'MANIFEST.json'}")


if __name__ == "__main__":
    main()
