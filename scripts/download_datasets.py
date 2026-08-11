
import argparse
import os
import shutil

DATASETS = {
    "isl_letters": {
        "kind": "kaggle",
        "id": "prathumarikeri/indian-sign-language-isl",
        "dest": "data/raw/isl/letters",
        "note": "Character-level ISL images. Reorganize into <LABEL>/*.jpg if not already.",
    },
    "bsl_letters": {
        "kind": "kaggle",
        "id": "alifsathar/bsl-fingerspelling-dataset",
        "dest": "data/raw/bsl/letters",
        "note": "Two-handed BSL alphabet images, already organized by letter folder.",
    },
    "isl_sentences": {
        "kind": "kaggle",
        "id": "drblack00/isl-csltr-indian-sign-language-dataset",
        "dest": "data/raw/isl/sentences",
        "note": "700 videos / 100 sentences / word+sentence frames -- see paper for frame->sentence mapping.",
    },
    "asl_words_wlasl": {
        "kind": "manual",
        "note": (
            "WLASL requires cloning https://github.com/dxli94/WLASL and running their "
            "video_downloader.py (needs yt-dlp, since videos are sourced from YouTube), "
            "then re-sorting into data/raw/asl/words/<GLOSS>/*.mp4 using WLASL_v0.3.json "
            "(see the reorg snippet in scripts/extract_landmarks_words.py's docstring). "
            "Requires agreeing to the C-UDA license."
        ),
    },
    "isl_words_include": {
        "kind": "manual",
        "note": (
            "INCLUDE dataset: https://zenodo.org/record/4010759 -- download and extract into "
            "data/raw/isl/words/<GLOSS>/*.mp4."
        ),
    },
    "bsl_words_fs23k": {
        "kind": "manual",
        "note": (
            "FS23K (BSL fingerspelling, derived from BOBSL): https://taeinkwon.com/projects/fs23k/ "
            "-- requires BOBSL access (BBC Terms of Use) to pull the underlying video clips, "
            "then extract into data/raw/bsl/words/<GLOSS>/*.mp4."
        ),
    },
    "bsl_sentences_bobsl": {
        "kind": "manual",
        "note": (
            "BOBSL: https://www.robots.ox.ac.uk/~vgg/data/bobsl/ -- 1,400+ hours, requires "
            "signing a BBC Terms of Use agreement before access is granted."
        ),
    },
}


def download_kaggle(dataset_id, dest):
    import kagglehub

    path = kagglehub.dataset_download(dataset_id)
    os.makedirs(dest, exist_ok=True)
    for item in os.listdir(path):
        src_path = os.path.join(path, item)
        dst_path = os.path.join(dest, item)
        if os.path.isdir(src_path):
            shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
        else:
            shutil.copy(src_path, dst_path)
    print(f"Downloaded '{dataset_id}' -> {dest}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=list(DATASETS.keys()))
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()

    if args.list or not args.dataset:
        for key, info in DATASETS.items():
            print(f"- {key} ({info['kind']}): {info.get('note', '')}")
        return

    info = DATASETS[args.dataset]
    if info["kind"] == "kaggle":
        download_kaggle(info["id"], info["dest"])
        print(f"Note: {info['note']}")
    else:
        print(f"'{args.dataset}' requires manual steps:\n\n{info['note']}")


if __name__ == "__main__":
    main()
