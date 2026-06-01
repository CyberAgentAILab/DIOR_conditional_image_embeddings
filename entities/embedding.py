from typing import Optional

from pydantic import BaseModel, Field


class EmbeddingOutput(BaseModel):
    embedding: list[float] = Field(...)
    prompt: str = Field(...)
    source: str = Field(...)
    layer: Optional[int] = Field(default=None)
    num_generated_tokens_used: Optional[int] = Field(default=None)
    stopped_by_eos: Optional[bool] = Field(default=None)
