import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
import torch
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
from tqdm import tqdm

from utils.utils import hsv_to_color_name, reshape_score

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
METADATA_DIR = os.path.join(REPO_ROOT, "metadata")
INFO_FILES_DIR = os.path.join(REPO_ROOT, "info_files")


class Dataset:
    def __init__(self, setting_path):
        self.setting_path = setting_path
        self.setting_name = setting_path.split("/")[-1]
        self.embeddings = []
        self.labels = []
        self.accuracy_calculator = AccuracyCalculator(device=torch.device("cpu"))

    def load_metadata(self):
        raise NotImplementedError("Subclasses should implement this method.")

    def load_embeddings(self):
        raise NotImplementedError("Subclasses should implement this method.")

    def compute_scores(self):
        raise NotImplementedError("Subclasses should implement this method.")

    def compute_accuracies(self, image_features, labels):
        possible_labels = list(set(labels))
        label_mapping = {label: i for i, label in enumerate(possible_labels)}
        int_labels = torch.tensor([label_mapping[label] for label in labels])

        return self.accuracy_calculator.get_accuracy(
            image_features, int_labels, image_features, int_labels, ref_includes_query=True
        )

    def evaluate(self):
        self.load_metadata()
        self.load_embeddings()
        scores = self.compute_scores()
        result = {"setting_name": self.setting_name}
        result.update(scores)
        return result


class SyntheticCarsDataset(Dataset):
    def load_metadata(self):
        df = pd.read_csv(os.path.join(METADATA_DIR, "synthetic_metadata.csv"), index_col=0)[:1000]
        df["color_name"] = df.apply(
            hsv_to_color_name,
            axis=1,
            hue_column="color_hue",
            sat_column="color_sat",
            val_column="color_val",
        )
        df["bg_color_name"] = df.apply(
            hsv_to_color_name,
            axis=1,
            hue_column="bg_color_hue",
            sat_column="bg_color_sat",
            val_column="bg_color_val",
        )
        self.metadata = df

    def load_embeddings(self):
        embeddings = []
        paths = sorted(glob.glob(f"{self.setting_path}/*.json"))
        for path in paths:
            with open(path, "r", encoding="utf-8") as f:
                line = f.readline()
                data = json.loads(line)
                embedding = data.get("embedding")
                if isinstance(embedding[0], list):
                    embedding = embedding[0]
            embeddings.append(embedding)
        self.embeddings = np.array(embeddings)

    def compute_scores(self):
        df = self.metadata
        model_score = self.compute_accuracies(self.embeddings, df["model"])
        color_score = self.compute_accuracies(self.embeddings, df["color_name"])
        bg_color_score = self.compute_accuracies(self.embeddings, df["bg_color_name"])

        return {
            "model_score": reshape_score(model_score),
            "color_score": reshape_score(color_score),
            "bg_color_score": reshape_score(bg_color_score),
        }


class DeepFashionDataset(Dataset):
    def load_metadata(self):
        df = pd.read_csv(os.path.join(METADATA_DIR, "deepfashon_metadata.csv"))
        self.metadata = df

    def load_embeddings(self):
        embeddings = []
        df = self.metadata
        for _, row in df.iterrows():
            filename = row["file_names"].replace(".jpg", ".json")
            path = f"{self.setting_path}/{filename}"
            with open(path, "r", encoding="utf-8") as f:
                line = f.readline()
                data = json.loads(line)
                embedding = data.get("embedding")
                if isinstance(embedding[0], list):
                    embedding = embedding[0]
            embeddings.append(embedding)
        self.embeddings = np.array(embeddings)

    def compute_scores(self):
        df = self.metadata
        clothing_category_score = self.compute_accuracies(self.embeddings, df["clothing_category"])
        texture_score = self.compute_accuracies(self.embeddings, df["texture"])
        fabric_score = self.compute_accuracies(self.embeddings, df["fabric"])
        fit_score = self.compute_accuracies(self.embeddings, df["fit"])
        return {
            "clothing_category_score": reshape_score(clothing_category_score),
            "texture_score": reshape_score(texture_score),
            "fabric_score": reshape_score(fabric_score),
            "fit_score": reshape_score(fit_score),
        }


class MoviePosterDataset(Dataset):
    def load_metadata(self):
        df = pd.read_csv(os.path.join(METADATA_DIR, "movie_poseter_metadata.tsv"), sep="\t")
        self.metadata = df

    def load_embeddings(self):
        embeddings = []
        df = self.metadata
        for _, row in df.iterrows():
            filename = f"{row['imdbID']}.json"
            path = f"{self.setting_path}/{filename}"
            with open(path, "r", encoding="utf-8") as f:
                line = f.readline()
                data = json.loads(line)
                embedding = data.get("embedding", None)
                if not embedding:
                    embedding = data.get("token_embedding")
                if isinstance(embedding[0], list):
                    embedding = embedding[0]
            embeddings.append(embedding)
        self.embeddings = np.array(embeddings)

    def compute_scores(self):
        df = self.metadata
        genre_score = self.compute_accuracies(self.embeddings, df["Genre"])
        country_score = self.compute_accuracies(self.embeddings, df["Country"])
        return {
            "genre_score": reshape_score(genre_score),
            "country_score": reshape_score(country_score),
        }


class CUB200Dataset(Dataset):
    def load_metadata(self):
        # Labels are extracted from filenames in load_embeddings
        pass

    def load_embeddings(self):
        embeddings = []
        labels = []
        paths = glob.glob(f"{self.setting_path}/*.json")
        for path in paths:
            label = " ".join(path.split("/")[-1].split(".")[0].split("_")[:-2])
            with open(path, "r", encoding="utf-8") as f:
                line = f.readline()
                data = json.loads(line)
                embedding = data.get("embedding")
                if isinstance(embedding[0], list):
                    embedding = embedding[0]
            embeddings.append(embedding)
            labels.append(label)
        self.embeddings = np.array(embeddings)
        self.labels = np.array(labels)

    def compute_scores(self):
        species_score = self.compute_accuracies(self.embeddings, self.labels)
        return {"species_score": reshape_score(species_score)}


class UNedDataset(Dataset):
    def load_metadata(self):
        raise NotImplementedError("Subclasses should implement this method.")

    def load_embeddings(self):
        embeddings = []
        labels = []
        for _, row in self.metadata.iterrows():
            name = f"{row['name'].split('.')[0]}.json"
            label = row["label"]
            path = f"{self.setting_path}/{name}"
            with open(path, "r", encoding="utf-8") as f:
                line = f.readline()
                data = json.loads(line)
                embedding = data.get("embedding")
                if isinstance(embedding[0], list):
                    embedding = embedding[0]
            embeddings.append(embedding)
            labels.append(label)
        self.embeddings = np.array(embeddings)
        self.labels = np.array(labels)

    def compute_scores(self):
        score = self.compute_accuracies(self.embeddings, self.labels)
        return {"overall_score": reshape_score(score)}


def _load_info_file_metadata(subdir: str) -> pd.DataFrame:
    with open(os.path.join(INFO_FILES_DIR, subdir, "test.json"), mode="r") as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    df.rename(columns={"class_id": "label"}, inplace=True)
    df["name"] = df["path"].apply(lambda x: x.split("/")[-1])
    return df


class CarsDataset(UNedDataset):
    def load_metadata(self):
        self.metadata = _load_info_file_metadata("cars")


class GLDv2Dataset(UNedDataset):
    def load_metadata(self):
        self.metadata = _load_info_file_metadata("gldv2")


class InatDataset(UNedDataset):
    def load_metadata(self):
        self.metadata = _load_info_file_metadata("inat")


class InShopDataset(UNedDataset):
    def load_metadata(self):
        self.metadata = _load_info_file_metadata("inshop")


class MetDataset(UNedDataset):
    def load_metadata(self):
        self.metadata = _load_info_file_metadata("met")


class RP2kDataset(UNedDataset):
    def load_metadata(self):
        self.metadata = _load_info_file_metadata("rp2k")


class SOPDataset(UNedDataset):
    def load_metadata(self):
        self.metadata = _load_info_file_metadata("sop")


class Food2kDataset(UNedDataset):
    def load_metadata(self):
        self.metadata = _load_info_file_metadata("food2k")


def find_target_score_key(setting_name: str, result_keys: list) -> str:
    """Find target score key from setting name by matching aspect_score pattern."""
    parts = setting_name.split("-")
    for part in parts:
        score_key = f"{part}_score"
        if score_key in result_keys:
            return score_key
    return None


def print_score(key: str, value):
    """Print a single score."""
    if isinstance(value, dict):
        print(f"  {key}:")
        for k, v in value.items():
            if isinstance(v, float):
                print(f"    {k}: {v:.4f}")
            else:
                print(f"    {k}: {v}")
    elif isinstance(value, float):
        print(f"  {key}: {value:.4f}")
    else:
        print(f"  {key}: {value}")


def main(args):
    dataset_classes = {
        "synthetic_cars": SyntheticCarsDataset,
        "deepfashion": DeepFashionDataset,
        "movie_posters": MoviePosterDataset,
        "cub200": CUB200Dataset,
        "cars196": CarsDataset,
        "gldv2": GLDv2Dataset,
        "inat": InatDataset,
        "inshop": InShopDataset,
        "met": MetDataset,
        "rp2k": RP2kDataset,
        "sop": SOPDataset,
        "food2k": Food2kDataset,
    }

    embedding_dir = args.embedding_dir.rstrip("/")
    settings_paths = [path for path in sorted(glob.glob(f"{embedding_dir}/{args.setting_pattern}"))]

    os.makedirs("results", exist_ok=True)
    all_results = []

    for setting_path in tqdm(settings_paths):
        setting_name = setting_path.split("/")[-1]
        if not os.path.isdir(setting_path) or not glob.glob(f"{setting_path}/*.json"):
            print(f"Skipping {setting_name} (no embedding files)")
            continue
        dataset_prefix = setting_name.split("-")[0]
        DatasetClass = dataset_classes.get(dataset_prefix)
        if DatasetClass is None:
            print(f"No dataset class found for setting {setting_name}")
            continue
        output_path = f"results/{setting_name}.json"
        try:
            dataset = DatasetClass(setting_path)
            score = dataset.evaluate()
            all_results.append(score)
            with open(output_path, mode="w") as f:
                json.dump(score, f, ensure_ascii=False, indent=4)
        except Exception as e:
            print(f"Error occuers in {setting_name}, {e}")

    # Print results
    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    for result in all_results:
        setting_name = result.get("setting_name", "unknown")
        print(f"\nSetting: {setting_name}")

        # Find target score key from setting name
        target_score_key = find_target_score_key(setting_name, list(result.keys()))

        # Print only target aspect score if found, otherwise print all
        if target_score_key:
            print_score(target_score_key, result[target_score_key])
        else:
            # Fallback: print all scores
            for key, value in result.items():
                if key == "setting_name":
                    continue
                print_score(key, value)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Embedding evaluation script")
    parser.add_argument(
        "--embedding_dir", default="embeddings", type=str, help="Path to embedding directory (default: embeddings)"
    )
    parser.add_argument("--setting_pattern", type=str, required=True, help="Setting pattern to evaluate (glob format)")
    parser.add_argument("--mode", choices=["all", "one_setting"], default="one_setting", type=str)

    args = parser.parse_args()
    main(args)
