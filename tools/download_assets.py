"""
Download the large datasets and checkpoints that are not shipped inside the
public repository (datasets and unpublished/redistribution-restricted weights are
hosted on Google Drive and fetched here).

USAGE
    python -m tools.download_assets            # download everything that is configured
    python -m tools.download_assets --only fr_checkpoints demo_ppt

NOTE FOR THE MAINTAINER
    Each asset below has a ``url`` field set to a unique, greppable placeholder token
    of the form ``DRIVE_LINK__<NAME>``. After uploading the corresponding file/zip to
    Google Drive, replace the matching token with the shareable link. Run

        grep -rn "DRIVE_LINK__" .

    to find every placeholder that still needs a link. Assets whose ``url`` is still a
    placeholder are skipped with a clear message instead of failing.

    Most facial-recognition checkpoints additionally download themselves on first use
    (see ``protego/FacialRecognition.py``), so ``fr_checkpoints`` is only a convenience
    bundle that mirrors the ``FR_DB/<model>/pretrained/`` layout.
"""
import argparse
import os
import zipfile

import gdown

from protego import BASE_PATH

# Each entry: key -> {url, dest, unzip, note}
#   url   : a Google Drive share link, OR a "DRIVE_LINK__<NAME>" placeholder to be filled in.
#   dest  : directory (relative to the repo root) the file is downloaded into / extracted to.
#   unzip : whether the downloaded file is a .zip that should be extracted into ``dest``.
ASSETS = {
    # Preprocessed FaceScrub eval subset: cropped ``face_scrub`` plus the ``_noise_db``
    # retrieval database the recall metric needs.
    "facescrub_eval": {
        "url": "DRIVE_LINK__FACESCRUB_EVAL",
        "dest": os.path.join(BASE_PATH, "face_db"),
        "unzip": True,
        "note": "Preprocessed FaceScrub eval subset (cropped face_scrub/ + _noise_db/).",
    },
    # SMIRK pretrained model. Alternatively fetched from the official SMIRK repo by
    # ``smirk/quick_install.sh`` during ``setup_quick.sh``.
    "smirk_weights": {
        "url": "DRIVE_LINK__SMIRK_WEIGHTS",
        "dest": os.path.join(BASE_PATH, "smirk", "pretrained_models"),
        "unzip": False,
        "note": "SMIRK_em1.pt -> smirk/pretrained_models/SMIRK_em1.pt",
    },
    # MediaPipe face landmarker task. Usually already provided by the SMIRK assets clone.
    "mediapipe_task": {
        "url": "DRIVE_LINK__MEDIAPIPE_TASK",
        "dest": os.path.join(BASE_PATH, "smirk", "assets"),
        "unzip": False,
        "note": "face_landmarker.task -> smirk/assets/face_landmarker.task",
    },
    # One zip mirroring the kept FR_DB/<model>/pretrained/ layout.
    "fr_checkpoints": {
        "url": "DRIVE_LINK__FR_CHECKPOINTS",
        "dest": BASE_PATH,
        "unzip": True,
        "note": "FR checkpoints zip mirroring FR_DB/<model>/pretrained/ (also auto-download on first use).",
    },
    # Pretrained Protego PPT(s) for the demo protectee so the inference notebook works out-of-the-box.
    "demo_ppt": {
        "url": "DRIVE_LINK__DEMO_PPT",
        "dest": os.path.join(BASE_PATH, "experiments"),
        "unzip": True,
        "note": "Pretrained PPT(s)/univ_mask.npy for the demo protectee -> experiments/default/<protectee>/univ_mask.npy",
    },
    # Sample image/video for the inference notebook/demo.
    "sample_media": {
        "url": "DRIVE_LINK__SAMPLE_MEDIA",
        "dest": os.path.join(BASE_PATH, "face_db"),
        "unzip": True,
        "note": "Sample image(s)/video for the inference notebook/demo.",
    },
}


def _download_one(key: str, spec: dict) -> bool:
    url = spec["url"]
    if url.startswith("DRIVE_LINK__"):
        print(f"[skip] '{key}': link not configured yet ({url}).")
        print(f"       Expected: {spec['note']}")
        print(f"       Fill in the link in tools/download_assets.py, then re-run.")
        return False
    dest = spec["dest"]
    os.makedirs(dest, exist_ok=True)
    print(f"[download] '{key}' -> {dest}")
    out = gdown.download(url=url, output=dest + os.sep, fuzzy=True, quiet=False)
    if out is None:
        print(f"[error] Failed to download '{key}'. Check the link/permissions.")
        return False
    if spec["unzip"] and str(out).endswith(".zip"):
        with zipfile.ZipFile(out, "r") as zf:
            zf.extractall(dest)
        os.remove(out)
        print(f"[done] Extracted '{key}' into {dest}.")
    else:
        print(f"[done] Saved '{key}' to {out}.")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Protego datasets and checkpoints.")
    parser.add_argument("--only", nargs="+", choices=list(ASSETS.keys()),
                        help="Only download the specified assets (default: all configured).")
    cli = parser.parse_args()

    keys = cli.only if cli.only else list(ASSETS.keys())
    configured = 0
    for key in keys:
        if _download_one(key, ASSETS[key]):
            configured += 1
    print(f"\nFinished. {configured}/{len(keys)} asset(s) downloaded.")
    if configured < len(keys):
        print("Some assets were skipped because their Google Drive links are not set.")
        print('Run  grep -rn "DRIVE_LINK__" .  to locate the placeholders to fill in.')


if __name__ == "__main__":
    main()
