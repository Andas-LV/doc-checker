import json
from pathlib import Path

import pytest

from app.extract import (
    LLMResponseError,
    _parse_json_object,
    extract_fields,
    pick_counterparty,
    resolve_number,
)
from app.ocr import DocumentReadError, extract_text
from app.schemas import LLMOutput

DOCS = Path(__file__).resolve().parent.parent / "data" / "test_docs"
EXPECTED = json.loads((DOCS / "expected.json").read_text(encoding="utf-8"))
FIELDS = ["doc_type", "counterparty", "amount", "currency", "date", "number"]


@pytest.mark.parametrize(
    "ours",
    ["ТОО «Демо Компания»", "ТОО «Демо-Компания»", 'ТОО "ДЕМО  КОМПАНИЯ"'],
)
def test_our_company_is_never_the_counterparty(ours):
    assert pick_counterparty([ours, "ТОО «Альфа»"]) == "ТОО «Альфа»"


def test_counterparty_is_none_when_only_our_company():
    assert pick_counterparty(["ТОО «Демо Компания»"]) is None


@pytest.mark.parametrize(
    ("text", "from_model", "expected"),
    [
        ("Счёт № СЧ-0457 от", None, "СЧ-0457"),  # знак важнее модели
        ("Договор № ____ от", "12/26", None),  # пусто после знака — явный null
        ("Заявка без знака", "ЗК-118", "ЗК-118"),  # знака нет — доверяем модели
        ("Заявка без знака", None, None),
    ],
)
def test_resolve_number(text, from_model, expected):
    assert resolve_number(text, from_model) == expected


def test_parse_json_object_unwraps_markdown_fence():
    assert _parse_json_object('```json\n{"doc_type": "other"}\n```') == {"doc_type": "other"}


@pytest.mark.parametrize("content", [None, "", "извините, не могу"])
def test_parse_json_object_reports_missing_json(content):
    with pytest.raises(LLMResponseError):
        _parse_json_object(content)


def test_amount_accepts_spaced_number():
    assert LLMOutput(doc_type="other", amount="7 200 000").amount == 7_200_000.0


def test_parties_flattens_objects_and_drops_blanks():
    parsed = LLMOutput(doc_type="other", parties=[{"name": "ТОО «Х»"}, "  ", "ТОО «У»"])
    assert parsed.parties == ["ТОО «Х»", "ТОО «У»"]


def test_unsupported_format_is_rejected():
    with pytest.raises(DocumentReadError, match="Неподдерживаемый формат"):
        extract_text(b"PK\x03\x04", "contract.docx")


def test_broken_pdf_is_rejected():
    with pytest.raises(DocumentReadError, match="Повреждённый PDF"):
        extract_text(b"not a pdf", "contract.pdf")


@pytest.mark.llm
@pytest.mark.parametrize("filename", sorted(EXPECTED))
def test_fields_match_expected(filename):
    text = extract_text((DOCS / filename).read_bytes(), filename)
    got = extract_fields(text).model_dump()
    expected = EXPECTED[filename]
    assert {f: got[f] for f in FIELDS} == {f: expected[f] for f in FIELDS}
