from typing import Literal

from pydantic import BaseModel, Field, field_validator

DocType = Literal["contract", "invoice", "application", "other"]


class ExtractedFields(BaseModel):
    doc_type: DocType
    counterparty: str | None = None
    amount: float | None = None
    currency: str | None = None
    date: str | None = None
    number: str | None = None


class ExtractResponse(BaseModel):
    text_preview: str
    fields: ExtractedFields


class LLMOutput(BaseModel):
    doc_type: DocType
    parties: list[str] = Field(default_factory=list)
    amount: float | None = None
    currency: str | None = None
    date: str | None = None
    number: str | None = None

    @field_validator("parties", mode="before")
    @classmethod
    def flatten_parties(cls, value):
        names = []
        for party in value or []:
            if isinstance(party, dict):  # модель вернула объект вместо строки
                party = party.get("name") or party.get("title") or next(
                    (v for v in party.values() if isinstance(v, str)), None
                )
            if isinstance(party, str) and party.strip():
                names.append(party.strip())
        return names

    @field_validator("amount", mode="before")
    @classmethod
    def clean_amount(cls, value):
        if not isinstance(value, str):
            return value
        # «7 200 000» или «185 000,50»: пробелы разделяют тысячи, запятая — дробная часть
        digits = value.replace("\xa0", "").replace(" ", "").replace(",", ".")
        try:
            return float(digits)
        except ValueError:
            return None
