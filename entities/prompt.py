from typing import Optional

from pydantic import BaseModel, Field

from .config import PromptType


class PromptTemplate(BaseModel):
    with_aspect: str = Field(...)
    without_aspect: str = Field(...)


class PromptTemplates:
    DESCRIBE = PromptTemplate(
        with_aspect='Describe this image in one word regarding {aspect}: "',
        without_aspect='Describe this image in one word: "',
    )
    EXPRESS = PromptTemplate(
        with_aspect='Express this image in one word in terms of {aspect}: "',
        without_aspect='Express this image in one word: "',
    )
    SUMMARIZE = PromptTemplate(
        with_aspect='Summarize this image in one word regarding {aspect}: "',
        without_aspect='Summarize this image in one word: "',
    )
    CAPTURE = PromptTemplate(
        with_aspect='Capture this image in one word based on {aspect}: "',
        without_aspect='Capture this image in one word: "',
    )
    CONVEY = PromptTemplate(
        with_aspect='Convey this image in one word with respect to {aspect}: "',
        without_aspect='Convey this image in one word: "',
    )
    DEPICT = PromptTemplate(
        with_aspect='Depict this image in one word, considering {aspect}: "',
        without_aspect='Depict this image in one word: "',
    )
    DESCRIBE_UNCONSTRAINT = PromptTemplate(
        with_aspect='Describe this image regarding {aspect}: "', without_aspect='Describe this image: "'
    )

    @classmethod
    def get(cls, prompt_type: PromptType) -> PromptTemplate:
        mapping = {
            PromptType.DESCRIBE: cls.DESCRIBE,
            PromptType.EXPRESS: cls.EXPRESS,
            PromptType.SUMMARIZE: cls.SUMMARIZE,
            PromptType.CAPTURE: cls.CAPTURE,
            PromptType.CONVEY: cls.CONVEY,
            PromptType.DEPICT: cls.DEPICT,
            PromptType.DESCRIBE_UNCONSTRAINT: cls.DESCRIBE_UNCONSTRAINT,
        }
        return mapping.get(prompt_type, cls.DESCRIBE)


def get_prompt(prompt_type: PromptType | str, aspect: Optional[str]) -> str:
    if isinstance(prompt_type, str):
        ptype_str = prompt_type.strip().lower()
        try:
            prompt_type = PromptType(ptype_str)
        except ValueError:
            prompt_type = PromptType.DESCRIBE

    template = PromptTemplates.get(prompt_type)

    if aspect:
        return template.with_aspect.format(aspect=aspect)
    return template.without_aspect
