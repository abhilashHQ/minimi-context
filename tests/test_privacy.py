from __future__ import annotations

from person_context.models import Alias
from person_context.privacy import gate

LRM = "\N{LEFT-TO-RIGHT MARK}"


def alias(value: str, status: str = "confirmed", kind: str = "name") -> Alias:
    return Alias(1, value, kind, status, None)


def test_only_blocks_with_an_alias_pass() -> None:
    blocks = ["Email from Asha Rao about pricing", "Karan Mehta asked about lunch", "noise"]
    result = gate(blocks, [alias("Asha Rao")])
    assert result.passed == ("Email from Asha Rao about pricing",)
    assert result.dropped == 2


def test_word_boundary_after_normalize() -> None:
    aliases = [alias("Ana")]
    assert gate([f"{LRM}ANA  said yes"], aliases).passed
    assert gate(["Ana, are you there?"], aliases).passed
    assert not gate(["Banana bread", "Anand called"], aliases).passed


def test_email_and_proposed_aliases_pass() -> None:
    aliases = [alias("asha@example.com", "proposed", "email")]
    assert gate(["From: Asha@Example.com"], aliases).passed
    assert not gate(["From: masha@example.com.au"], aliases).passed


def test_rejected_and_empty_aliases_never_pass() -> None:
    aliases = [alias("Acme", "rejected", "company"), alias("  ")]
    result = gate(["Acme invoice", "anything"], aliases)
    assert result.passed == ()
    assert result.dropped == 2


def test_no_aliases_drops_everything() -> None:
    assert gate(["Asha Rao"], []).dropped == 1
