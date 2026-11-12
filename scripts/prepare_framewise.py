"""Prepare pinned official code, Small weights and the supplied example clip."""

import hashlib
import os
import subprocess
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_COMMIT = "a561b849ebae10a6f5ef49e26c83cbbcd36c71bf"
ARTIFACTS = (
    (
        "checkpoints/depth_anything_v2_vits.pth",
        "https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/03876f8651c73a60fe4c2c48294e09fcb6838fcf/depth_anything_v2_vits.pth",
        "715fade13be8f229f8a70cc02066f656f2423a59effd0579197bbf57860e1378",
    ),
    (
        "data/videos/davis_rollercoaster.mp4",
        "https://raw.githubusercontent.com/DepthAnything/Video-Depth-Anything/4f5ae23172ba60fd7bc11ef671cca678842c7072/assets/example_videos/davis_rollercoaster.mp4",
        "7268cbecd9806a1e90a416de50dc02e50b4ae01428d5971837cf679dd0c87cb8",
    ),
)


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare_artifact(relative_path, url, expected):
    target = ROOT / relative_path
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if sha256_file(target) != expected:
            raise ValueError(f"Existing artifact checksum mismatch: {target}")
    else:
        temporary = target.with_suffix(target.suffix + ".part")
        print(f"Downloading {url}", flush=True)
        request = urllib.request.Request(url, headers={"User-Agent": "AIAA3201-project"})
        with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as stream:
            while block := response.read(1024 * 1024):
                stream.write(block)
        if sha256_file(temporary) != expected:
            raise ValueError(f"Downloaded artifact checksum mismatch: {temporary}")
        os.replace(temporary, target)
    print(f"SHA-256 verified: {relative_path}", flush=True)


def main():
    upstream = ROOT / "external/Depth-Anything-V2"
    upstream.parent.mkdir(parents=True, exist_ok=True)
    if not upstream.exists():
        subprocess.run(["git", "clone", "--depth", "1",
                        "https://github.com/DepthAnything/Depth-Anything-V2.git",
                        str(upstream)], check=True)
    def git(*arguments):
        return subprocess.run(["git", "-C", str(upstream), *arguments],
                              check=True, capture_output=True, text=True).stdout.strip()
    if git("status", "--porcelain"):
        raise ValueError(f"Upstream code has local modifications: {upstream}")
    if git("rev-parse", "HEAD") != MODEL_COMMIT:
        git("fetch", "--depth", "1", "origin", MODEL_COMMIT)
        git("checkout", "--detach", MODEL_COMMIT)
    print(f"Model code verified: {git('rev-parse', 'HEAD')}", flush=True)
    for artifact in ARTIFACTS:
        prepare_artifact(*artifact)


if __name__ == "__main__":
    main()
