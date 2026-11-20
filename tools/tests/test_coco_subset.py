"""Tests for the pure selection logic and helpers of tools/coco_subset.py."""
import json
import zipfile

import pytest

import coco_subset as cs


def synthetic(n=1000, empty=(720, 950), crowded=(800, 900), crowded_n=40):
    """n images 000000000000.jpg.. in reverse insertion order (selection must sort)."""
    entries = []
    for i in range(n):
        k = 3
        if i in empty:
            k = 0
        if i in crowded:
            k = crowded_n
        entries.append((f"{i:012d}.jpg", k))
    return list(reversed(entries))


def test_eval_and_calib_are_first_sorted_and_disjoint():
    sel = cs.select_subsets(synthetic())
    assert sel["eval"] == [f"{i:012d}.jpg" for i in range(500)]
    assert sel["calib"] == [f"{i:012d}.jpg" for i in range(500, 700)]
    assert not set(sel["eval"]) & set(sel["calib"])


def test_stream_is_50_sorted_spanning_remaining():
    sel = cs.select_subsets(synthetic())
    s = sel["stream"]
    assert len(s) == 50 and s == sorted(s) and len(set(s)) == 50
    assert not set(s) & (set(sel["eval"]) | set(sel["calib"]))
    assert s[0] == "000000000700.jpg"          # starts at the first remaining image
    assert int(s[-1][:12]) >= 700 + 300 * 49 // 50 - 1   # reaches the end of the set


def test_empty_is_first_remaining_zero_label_image():
    entries = synthetic(empty=(10, 720, 950))   # an early empty one is not "remaining"
    sel = cs.select_subsets(entries)
    assert sel["empty"] == "000000000720.jpg"


def test_crowded_is_max_labels_ties_to_first_sorted():
    sel = cs.select_subsets(synthetic(crowded=(800, 900, 100), crowded_n=40))
    assert sel["crowded"] == "000000000800.jpg"      # 100 is in eval, not remaining


def test_selection_is_deterministic_and_input_order_independent():
    a = cs.select_subsets(synthetic())
    b = cs.select_subsets(sorted(synthetic()))
    assert a == b


def test_too_few_images_or_no_empty_raises():
    with pytest.raises(ValueError, match="at least"):
        cs.select_subsets(synthetic(n=740))
    with pytest.raises(ValueError, match="empty"):
        cs.select_subsets(synthetic(empty=(10,)))


def test_count_label_lines_ignores_blank_lines():
    assert cs.count_label_lines("0 .1 .2 .3 .4\n\n1 .1 .2 .3 .4\n") == 2
    assert cs.count_label_lines("") == 0


def test_manifest_entries_and_rule(tmp_path):
    f = tmp_path / "a.jpg"
    f.write_bytes(b"abc")
    m = cs.manifest_entry("000000000001.jpg", f, 7)
    assert m == {"coco_file": "000000000001.jpg", "n_labels": 7,
                 "sha256": "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"}
    assert "sorted" in cs.SELECTION_RULE["eval"]


def test_ensure_zip_skips_valid_existing_and_redownloads_corrupt(tmp_path):
    good = tmp_path / "good.zip"
    with zipfile.ZipFile(good, "w") as z:
        z.writestr("x.txt", "hi")
    calls = []
    cs.ensure_zip(good, "http://example/x", download=lambda u, d: calls.append(u))
    assert calls == []                          # present and valid: no download

    bad = tmp_path / "bad.zip"
    bad.write_bytes(b"not a zip")

    def fake(url, dest):
        with zipfile.ZipFile(dest, "w") as z:
            z.writestr("y.txt", "ok")
    cs.ensure_zip(bad, "http://example/y", download=fake)
    assert zipfile.is_zipfile(bad)


def test_ensure_zip_raises_if_download_still_invalid(tmp_path):
    dest = tmp_path / "z.zip"
    with pytest.raises(RuntimeError, match="integrity"):
        cs.ensure_zip(dest, "http://example/z",
                      download=lambda u, d: d.write_bytes(b"junk"))


def test_dataset_yaml_has_80_names():
    y = cs.dataset_yaml("/abs/coco500")
    assert "path: /abs/coco500" in y and "val: images" in y
    assert "0: person" in y and "79: toothbrush" in y
    assert len(cs.COCO_NAMES) == 80


# ---- resumable download against a local server that honours Range -----------------
@pytest.fixture
def range_server():
    import http.server
    import threading

    payload = bytes(range(256)) * 4000
    seen = []

    class H(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            rng = self.headers.get("Range")
            seen.append(rng)
            start = int(rng.split("=")[1].split("-")[0]) if rng else 0
            if start >= len(payload):
                self.send_response(416)
                self.end_headers()
                return
            body = payload[start:]
            self.send_response(206 if rng else 200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    srv = http.server.HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}/f", payload, seen
    srv.shutdown()


def test_download_fresh_and_resume(tmp_path, range_server):
    url, payload, seen = range_server
    dest = tmp_path / "d" / "f.bin"
    cs.download(url, dest)
    assert dest.read_bytes() == payload and seen == [None]

    dest.unlink()
    (tmp_path / "d" / "f.bin.part").write_bytes(payload[:1000])   # interrupted earlier
    cs.download(url, dest)
    assert dest.read_bytes() == payload and seen[-1] == "bytes=1000-"
    assert not (tmp_path / "d" / "f.bin.part").exists()


def test_download_part_already_complete(tmp_path, range_server):
    url, payload, _ = range_server
    dest = tmp_path / "f.bin"
    (tmp_path / "f.bin.part").write_bytes(payload)
    cs.download(url, dest)
    assert dest.read_bytes() == payload


def test_build_end_to_end_on_tiny_zips(tmp_path):
    """Small synthetic zips with the real archive layout; checks outputs and manifest."""
    n = 760
    with zipfile.ZipFile(tmp_path / "img.zip", "w") as zi, zipfile.ZipFile(tmp_path / "lab.zip", "w") as zl:
        zi.writestr("val2017/", "")
        for i in range(n):
            zi.writestr(f"val2017/{i:012d}.jpg", f"img{i}".encode())
            if i != 710:                                   # 710: background image, no label file
                zl.writestr(f"coco/labels/val2017/{i:012d}.txt", "0 .5 .5 .1 .1\n" * (30 if i == 750 else 2))
    sel, man = cs.build(tmp_path / "data", tmp_path / "inputs", tmp_path / "img.zip", tmp_path / "lab.zip")
    assert sel["empty"] == "000000000710.jpg" and sel["crowded"] == "000000000750.jpg"
    assert len(list((tmp_path / "data/coco500/images").iterdir())) == 500
    assert len(list((tmp_path / "data/coco500/labels").iterdir())) == 500
    assert len(list((tmp_path / "data/calib200/images").iterdir())) == 200
    assert len(list((tmp_path / "inputs/stream50").iterdir())) == 50
    assert (tmp_path / "inputs/empty.jpg").read_bytes() == b"img710"
    assert "names:" in (tmp_path / "data/coco500.yaml").read_text()
    on_disk = json.loads((tmp_path / "inputs/MANIFEST.json").read_text())
    assert on_disk == man
    assert man["crowded"]["inputs/crowded.jpg"]["n_labels"] == 30
    assert man["empty"]["inputs/empty.jpg"]["n_labels"] == 0
    assert len(man["stream"]) == 50
