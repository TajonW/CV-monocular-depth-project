"""Extract and verify the 15 required Middlebury trainingQ stereo pairs."""

import argparse
from pathlib import Path
from zipfile import ZipFile


def prepare(data_root: Path) -> None:
    archive_root = data_root / "downloads"
    output_root = (data_root / "middlebury").resolve()
    archives = [
        archive_root / "MiddEval3-data-Q.zip",
        archive_root / "MiddEval3-GT0-Q.zip",
    ]
    # Check inputs before creating or extracting output files.
    for archive_path in archives:
        if not archive_path.is_file():
            raise FileNotFoundError(f"Required archive not found: {archive_path}")

    selected = []
    for archive_path in archives:
        with ZipFile(archive_path) as archive:
            members = [
                member
                for member in archive.infolist()
                if member.filename.startswith("MiddEval3/trainingQ/")
                and not member.is_dir()
            ]
            if not members:
                raise ValueError(f"No trainingQ files in {archive_path}")
            for member in members:
                target = (output_root / member.filename).resolve()
                if not target.is_relative_to(output_root):
                    raise ValueError(f"Unsafe archive path: {member.filename}")
                if target.exists() and target.stat().st_size != member.file_size:
                    raise ValueError(f"Existing file has unexpected size: {target}")
            selected.append((archive_path, members))

    output_root.mkdir(parents=True, exist_ok=True)
    for archive_path, members in selected:
        with ZipFile(archive_path) as archive:
            for member in members:
                target = output_root / member.filename
                if not target.exists():
                    archive.extract(member, output_root)

    scene_root = output_root / "MiddEval3" / "trainingQ"
    scenes = sorted(path for path in scene_root.iterdir() if path.is_dir())
    if len(scenes) != 15:
        raise ValueError(f"Expected 15 scenes, found {len(scenes)} in {scene_root}")
    for scene in scenes:
        for name in ("im0.png", "im1.png", "calib.txt", "disp0GT.pfm"):
            if not (scene / name).is_file():
                raise FileNotFoundError(f"Missing scene input: {scene / name}")
    print(f"Verified {len(scenes)} scenes at {scene_root}")
    print("Scenes: " + ", ".join(scene.name for scene in scenes))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data",
        help="Directory containing downloads/; default: this repository's data/",
    )
    args = parser.parse_args()
    prepare(args.data_root.resolve())


if __name__ == "__main__":
    main()
