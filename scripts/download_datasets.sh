#!/bin/bash

# Download datasets used in the DIOR experiments.
#
# Some datasets (Stanford Cars, DeepFashion In-Shop) require authentication
# or manual acceptance of terms and cannot be fully automated. For those,
# this script prints the instructions you need to follow manually.
#
# Usage:
#   bash scripts/download_datasets.sh [DATASETS_DIR]
#
# DATASETS_DIR defaults to ./datasets (matches the --datasets_dir default).

set -e

DATASETS_DIR="${1:-./datasets}"
mkdir -p "${DATASETS_DIR}"

have_cmd() { command -v "$1" >/dev/null 2>&1; }

fetch() {
    local url="$1"
    local out="$2"
    if have_cmd wget; then
        wget -c "${url}" -O "${out}"
    elif have_cmd curl; then
        curl -L -C - "${url}" -o "${out}"
    else
        echo "ERROR: neither wget nor curl is available" >&2
        exit 1
    fi
}

# -----------------------------------------------------------------------------
# CUB-200-2011
# -----------------------------------------------------------------------------
download_cub200() {
    local target="${DATASETS_DIR}/cub200"
    if [ -d "${target}" ] && [ -n "$(ls -A "${target}" 2>/dev/null)" ]; then
        echo "[cub200] already present, skipping"
        return
    fi

    echo "[cub200] downloading CUB-200-2011 ..."
    mkdir -p "${target}"
    local tarball="${DATASETS_DIR}/CUB_200_2011.tgz"
    fetch "https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz" "${tarball}"

    echo "[cub200] extracting ..."
    tar -xzf "${tarball}" -C "${DATASETS_DIR}"

    # Follow the InDiReCT protocol: keep only test classes 101-200.
    # Class folders are named like "001.Black_footed_Albatross" — sorting them
    # alphabetically matches the ID order, so taking the last 100 gives the
    # metric-learning test split.
    echo "[cub200] selecting test classes 101-200 and flattening ..."
    local images_dir="${DATASETS_DIR}/CUB_200_2011/images"
    local test_classes
    test_classes=$(ls "${images_dir}" | sort | tail -n 100)
    while IFS= read -r cls; do
        find "${images_dir}/${cls}" -type f -name "*.jpg" -exec mv {} "${target}/" \;
    done <<< "${test_classes}"

    rm -rf "${DATASETS_DIR}/CUB_200_2011" "${DATASETS_DIR}/attributes.txt" "${tarball}"
    echo "[cub200] done"
}

# -----------------------------------------------------------------------------
# Movie Posters
# -----------------------------------------------------------------------------
download_movie_posters() {
    local target="${DATASETS_DIR}/movie_posters"
    if [ -d "${target}" ] && [ -n "$(ls -A "${target}" 2>/dev/null)" ]; then
        echo "[movie_posters] already present, skipping"
        return
    fi

    echo "[movie_posters] downloading Movie Posters ..."
    mkdir -p "${target}"
    local zipfile="${DATASETS_DIR}/Movie_Poster_Dataset.zip"
    fetch "https://www.cs.ccu.edu.tw/~wtchu/projects/MoviePoster/Movie_Poster_Dataset.zip" "${zipfile}"

    echo "[movie_posters] extracting ..."
    unzip -q -o "${zipfile}" -d "${target}"

    # If images landed in a nested directory, flatten them.
    find "${target}" -mindepth 2 -type f \( -name "*.jpg" -o -name "*.jpeg" -o -name "*.png" \) \
        -exec mv -n {} "${target}/" \;
    find "${target}" -mindepth 1 -type d -empty -delete

    rm -f "${zipfile}"
    echo "[movie_posters] done"
}

# -----------------------------------------------------------------------------
# Datasets that require manual steps
# -----------------------------------------------------------------------------
manual_synthetic_cars() {
    cat <<'EOF'

[synthetic_cars] The original download link is no longer available.
  Upstream: https://github.com/konstantinkobs/DML-analysis/tree/master/3D_cars
  You will need to obtain the images from another mirror (e.g. a cached copy)
  and place them (flat) under: datasets/synthetic_cars/
EOF
}

manual_cars196() {
    cat <<'EOF'

[cars196] Stanford Cars requires Kaggle authentication.
  1. Install the Kaggle CLI:  pip install kaggle
  2. Place your kaggle.json API token at ~/.kaggle/kaggle.json (chmod 600).
  3. Run:
       kaggle datasets download -d jessicali9530/stanford-cars-dataset -p datasets/cars196 --unzip
  4. Flatten so that all images sit directly under: datasets/cars196/
EOF
}

deepfashion() {
    local target="${DATASETS_DIR}/deepfashion"
    mkdir -p "${target}"

    # Already flattened? Nothing to do.
    if compgen -G "${target}/*.jpg" > /dev/null; then
        echo "[deepfashion] already present, skipping"
        return
    fi

    # If the user dropped img.zip in place, extract it.
    if [ -f "${target}/img.zip" ] && [ ! -d "${target}/img" ]; then
        echo "[deepfashion] extracting img.zip ..."
        unzip -q -o "${target}/img.zip" -d "${target}"
    fi

    # If we have the nested img/ tree, flatten {Category}/img_{id}.jpg
    # to {Category}-img_{id}.jpg so filenames match deepfashon_metadata.csv.
    if [ -d "${target}/img" ]; then
        echo "[deepfashion] flattening img/{Category}/img_*.jpg -> {Category}-img_*.jpg ..."
        find "${target}/img" -mindepth 2 -type f -name "*.jpg" | while IFS= read -r src; do
            local cat base
            cat=$(basename "$(dirname "${src}")")
            base=$(basename "${src}")
            mv -n "${src}" "${target}/${cat}-${base}"
        done
        rm -rf "${target}/img" "${target}/img.zip"
        echo "[deepfashion] done ($(ls "${target}" | wc -l) files)"
        return
    fi

    # Nothing to process — print manual download instructions.
    cat <<'EOF'

[deepfashion] DeepFashion In-Shop requires accepting the terms on the official site.
  1. Request access / review terms:
       https://mmlab.ie.cuhk.edu.hk/projects/DeepFashion/InShopRetrieval.html
  2. Download img.zip from the In-Shop Clothes Retrieval Benchmark folder:
       https://drive.google.com/drive/folders/0B7EVK8r0v71pekpRNUlMS3Z5cUk
     and place it at: datasets/deepfashion/img.zip
     (or extract it yourself so datasets/deepfashion/img/ exists)
  3. Re-run this script — it will extract and flatten the images into
     datasets/deepfashion/{Category}-img_{id}.jpg to match
     metadata/deepfashon_metadata.csv.
EOF
}

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
echo "Target directory: ${DATASETS_DIR}"
echo

download_cub200
download_movie_posters

manual_synthetic_cars
manual_cars196
deepfashion

echo
echo "Done. Datasets available automatically: cub200, movie_posters."
echo "See messages above for the remaining datasets."
