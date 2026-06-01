from typing import Optional

from pydantic import BaseModel, Field


class DatasetConfig(BaseModel):
    name: str = Field(...)
    aspects: list[Optional[str]] = Field(...)


class Datasets:
    SYNTHETIC_CARS = DatasetConfig(name="synthetic_cars", aspects=[None, "car model", "car color", "background color"])
    CARS196 = DatasetConfig(name="cars196", aspects=[None, "car model"])
    CUB200 = DatasetConfig(name="cub200", aspects=[None, "bird species"])
    DEEPFASHION = DatasetConfig(
        name="deepfashion", aspects=[None, "clothing category", "texture type", "fabric type", "fit type"]
    )
    MOVIE_POSTERS = DatasetConfig(name="movie_posters", aspects=[None, "movie genre", "movie's production country"])
    WIKIART = DatasetConfig(name="wikiart", aspects=[None])
    DOMAIN_NET = DatasetConfig(name="domain_net", aspects=[None])
    CARS = DatasetConfig(name="cars", aspects=["car model"])
    GLDV2 = DatasetConfig(name="gldv2", aspects=[None])
    INAT = DatasetConfig(name="inat", aspects=[None])
    INSHOP = DatasetConfig(name="inshop", aspects=[None])
    MET = DatasetConfig(name="met", aspects=[None])
    RP2K = DatasetConfig(name="rp2k", aspects=[None])
    SOP = DatasetConfig(name="sop", aspects=[None])
    FOOD2K = DatasetConfig(name="food2k", aspects=[None])

    @classmethod
    def get(cls, name: str) -> DatasetConfig:
        for attr in dir(cls):
            if attr.startswith("_"):
                continue
            val = getattr(cls, attr)
            if isinstance(val, DatasetConfig) and val.name == name:
                return val
        raise ValueError(f"Unknown dataset: {name}")

    @classmethod
    def list_names(cls) -> list[str]:
        return [
            getattr(cls, attr).name
            for attr in dir(cls)
            if not attr.startswith("_") and isinstance(getattr(cls, attr), DatasetConfig)
        ]
