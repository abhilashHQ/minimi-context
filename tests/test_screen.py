from __future__ import annotations

from helpers import ENCRYPTION_NOTICE, LRM, whatsapp_tree

from person_context.screen import AXNode, ScreenMessage, parse_screen

OUTGOING = f"{LRM}Your message, Hi, 10:05 AM, Sent to Bhavya, Red"
INCOMING = f"{LRM}Bhavya, any update on the order?, 10:07 AM"


def test_parses_section_11_fixture() -> None:
    tree = whatsapp_tree("Bhavya", [ENCRYPTION_NOTICE, OUTGOING, INCOMING], composer="draft")
    screen = parse_screen(tree)
    assert screen is not None
    assert screen.person_name == "Bhavya"
    assert screen.messages == (
        ScreenMessage("self", OUTGOING),
        ScreenMessage("them", INCOMING),
    )


def test_composer_is_never_read() -> None:
    tree = whatsapp_tree("Bhavya", [], composer="secret draft")
    screen = parse_screen(tree)
    assert screen is not None and screen.messages == ()


def test_header_from_description() -> None:
    tree = AXNode(
        children=(
            AXNode(
                role="AXButton", identifier="NavigationBar_HeaderViewButton", description="Asha"
            ),
            AXNode(identifier="ChatMessagesTableView"),
        )
    )
    screen = parse_screen(tree)
    assert screen is not None and screen.person_name == "Asha"


def test_no_open_chat_or_no_header_is_none() -> None:
    assert parse_screen(AXNode(role="AXWindow")) is None
    no_list = AXNode(
        children=(AXNode(role="AXButton", identifier="NavigationBar_HeaderViewButton", title="A"),)
    )
    assert parse_screen(no_list) is None
    no_header = AXNode(children=(AXNode(identifier="ChatMessagesTableView"),))
    assert parse_screen(no_header) is None
    wrong_role = AXNode(
        children=(
            AXNode(role="AXStaticText", identifier="NavigationBar_HeaderViewButton", title="A"),
            AXNode(identifier="ChatMessagesTableView"),
        )
    )
    assert parse_screen(wrong_role) is None


def test_unknown_shapes_keep_raw_text() -> None:
    odd_cell = AXNode(
        role="AXGroup",
        identifier="WAMessageBubbleTableViewCell",
        children=(AXNode(role="AXStaticText", value="Photo"), AXNode(title="10:09 AM")),
    )
    empty_cell = AXNode(identifier="WAMessageBubbleTableViewCell")
    tree = AXNode(
        children=(
            AXNode(role="AXButton", identifier="NavigationBar_HeaderViewButton", title="Bhavya"),
            AXNode(identifier="ChatMessagesTableView", children=(odd_cell, empty_cell)),
        )
    )
    screen = parse_screen(tree)
    assert screen is not None
    assert screen.messages == (ScreenMessage("unknown", "Photo 10:09 AM"),)
