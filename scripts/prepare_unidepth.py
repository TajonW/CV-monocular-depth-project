"""Download official pinned UniDepth code and Small weights into this project."""

import hashlib
import json
from pathlib import Path
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def fetch(url):
    return urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "CV-Course-Project"}), timeout=45)


def main():
    from huggingface_hub import snapshot_download
    from scipy.io import loadmat
    split_url = "https://horatio.cs.nyu.edu/mit/silberman/indoor_seg_sup/splits.mat"
    split = ROOT / "data/nyu/splits.mat"
    split.parent.mkdir(parents=True, exist_ok=True)
    if not split.exists():
        with fetch(split_url) as stream:
            split.write_bytes(stream.read())
    indices = loadmat(split)
    if indices["trainNdxs"].size != 795 or indices["testNdxs"].size != 654:
        raise ValueError("Invalid NYU split")
    (split.parent / "splits_provenance.json").write_text(json.dumps(dict(
        source_url=split_url, sha256=hashlib.sha256(split.read_bytes()).hexdigest(), development_count=795, test_count=654), indent=2))
    source = ROOT / "external/UniDepth-source"
    tmp = ROOT / "tmp"
    tmp.mkdir(exist_ok=True)
    if not (source / "unidepth/models/unidepthv2/unidepthv2.py").exists():
        commit = "8d8cfe4c7ee15297099983607febf0d4f32eb3d6"
        url = f"https://codeload.github.com/lpiccinelli-eth/UniDepth/zip/{commit}"
        archive = tmp / f"UniDepth-{commit}.zip"
        with fetch(url) as stream, archive.open("wb") as out:
            while chunk := stream.read(1024 * 1024):
                out.write(chunk)
        source.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive) as bundle:
            prefix = f"UniDepth-{commit}/"
            for member in bundle.infolist():
                if not member.filename.startswith(prefix):
                    raise ValueError("Unexpected archive root")
                relative = member.filename[len(prefix):]
                target = (source / relative).resolve()
                if not target.is_relative_to(source.resolve()):
                    raise ValueError("Unsafe archive path")
                if member.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                elif relative:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(bundle.read(member))
        provenance = dict(repository="https://github.com/lpiccinelli-eth/UniDepth", commit=commit,
                          archive_url=url, archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest())
        (source / "source_provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print("Official source:", source, flush=True)
    model_id = "lpiccinelli/unidepth-v2-vits14"
    revision = "038c238f06c87b6c2f5b3749fd51fbf442b1f218"
    destination = ROOT / "checkpoints/unidepth-v2-vits14"
    snapshot_download(model_id, revision=revision, token=False,
                      allow_patterns=["config.json", "model.safetensors", "pytorch_model.bin"],
                      local_dir=destination, cache_dir=tmp / "huggingface")
    provenance = dict(model_id=model_id, revision=revision,
                      files={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in destination.iterdir()
                             if p.is_file() and p.name != "download_provenance.json"})
    (destination / "download_provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print(json.dumps(provenance, indent=2), flush=True)


if __name__ == "__main__":
    main()
