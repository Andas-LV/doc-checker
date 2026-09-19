from typing import Literal, Optional
from pydantic import BaseModel, field_validator

class ExtractedFields(BaseModel):
    doc_type: Literal["contract", "invoice", "application", "other"]
    counterparty: Optional[str] = None
    amount: Optional[float] = None
    currency: Optional[str] = None
    date: Optional[str] = None
    number: Optional[str] = None

class LLMOutput(BaseModel):
    doc_type: Literal["contract", "invoice", "application", "other"]
    parties: list[str] = []
    amount: Optional[float] = None
    currency: Optional[str] = None
    date: Optional[str] = None
    number: Optional[str] = None

    @field_validator("parties", mode="before")
    @classmethod
    def flatten_parties(cls, v):
        out = []
        for p in v or []:
            if isinstance(p, dict):  # модель вернула объект вместо строки
                p = p.get("name") or p.get("title") or next(
                    (x for x in p.values() if isinstance(x, str)), None)
            if isinstance(p, str) and p.strip():
                out.append(p.strip())
        return out

    @field_validator("amount", mode="before")
    @classmethod
    def clean_amount(cls, v):
        if isinstance(v, str):  # "7 200 000" или "7200000,00"
            v = v.replace("\xa0", "").replace(" ", "").replace(",", ".")
            try:
                return float(v)
            except ValueError:
                return None
        return v