from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class PromptType(str, Enum):
    DESCRIBE = "describe"
    EXPRESS = "express"
    SUMMARIZE = "summarize"
    CAPTURE = "capture"
    CONVEY = "convey"
    DEPICT = "depict"
    DESCRIBE_UNCONSTRAINT = "describe_unconstraint"


class EmbeddingSource(str, Enum):
    INPUT_LAST = "input_last"
    FIRST_OUTPUT = "first_output"
    MEAN_OUTPUT = "mean_output"


class TextEncoder(str, Enum):
    SENTENCE_T5_BASE = "sentence-transformers/sentence-t5-base"
    SENTENCE_T5_LARGE = "sentence-transformers/sentence-t5-large"
    SENTENCE_T5_XL = "sentence-transformers/sentence-t5-xl"
    ALL_MINILM_L6 = "sentence-transformers/all-MiniLM-L6-v2"
    ALL_MPNET_BASE = "sentence-transformers/all-mpnet-base-v2"


class InferenceConfig(BaseModel):
    model_id: str = Field(..., description="HuggingFace model ID")
    trust_remote_code: bool = Field(default=True)

    dataset_name: str = Field(...)
    datasets_dir: str = Field(default="./datasets")
    dataset_info_dir: Optional[str] = Field(default=None)

    prompt_type: PromptType = Field(default=PromptType.DESCRIBE)

    embedding_dir: str = Field(default="./embeddings")
    num_layer: int = Field(default=-1)
    num_token: int = Field(default=-1)
    save_all: bool = Field(default=False)

    image_max_size: int = Field(default=700)

    use_cache: bool = Field(default=False)
    batch_size: int = Field(default=1)

    embedding_source: EmbeddingSource = Field(default=EmbeddingSource.INPUT_LAST)

    gen_max_new_tokens: int = Field(default=16)
    gen_do_sample: bool = Field(default=False)
    gen_temperature: float = Field(default=0.0)

    class Config:
        use_enum_values = True
