"""The palette is a claim; this is the check.

Every colour in the app comes from .streamlit/config.toml, and app.css is
forbidden from hardcoding one. So the only place contrast can break is the
theme itself, which makes it testable: 4.5:1 for text, 3:1 for boundaries,
focus indicators and icons (WCAG 2.2 AA, and antislop R-25).

The formula is the one published in the antislop-human skill, which is also how
the current values were chosen. It is inlined here so CI does not depend on a
skill being installed.
"""
import re
import tomllib
from pathlib import Path

DEMO = Path(__file__).resolve().parents[1]
CONFIG = DEMO.parent / ".streamlit" / "config.toml"
CSS = DEMO / "src" / "app.css"

TEXT_MIN = 4.5
BOUNDARY_MIN = 3.0


def luminance(color):
    channels = [int(color[i : i + 2], 16) / 255 for i in (1, 3, 5)]
    linear = [
        c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
        for c in channels
    ]
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


def contrast(foreground, background):
    first, second = luminance(foreground), luminance(background)
    lighter, darker = max(first, second), min(first, second)
    return (lighter + 0.05) / (darker + 0.05)


def palette(base):
    theme = tomllib.loads(CONFIG.read_text())["theme"]
    merged = dict(theme)
    merged.update(theme.get(base, {}))
    sidebar = dict(theme.get("sidebar", {}))
    sidebar.update(theme.get(base, {}).get("sidebar", {}))
    return {
        "text": merged["textColor"],
        "muted": merged.get("linkColor", merged["textColor"]),
        "link": merged["linkColor"],
        "primary": merged["primaryColor"],
        "bg": merged["backgroundColor"],
        "bg2": merged["secondaryBackgroundColor"],
        "sidebar_bg": sidebar.get("backgroundColor", merged["secondaryBackgroundColor"]),
        "border": merged["borderColor"],
        "sidebar_border": sidebar.get("borderColor", merged["borderColor"]),
    }


def test_theme_declares_what_the_app_needs():
    """Both modes need their own colours, or a mode switch shows the other one.

    A colour set in [theme] applies to both modes and overrides the mode table,
    which is exactly how the first attempt at this shipped a light palette in
    dark mode.
    """
    theme = tomllib.loads(CONFIG.read_text())["theme"]
    assert theme["base"] in ("light", "dark")
    for mode in ("light", "dark"):
        for key in (
            "primaryColor",
            "backgroundColor",
            "secondaryBackgroundColor",
            "textColor",
            "linkColor",
            "borderColor",
        ):
            assert key in theme.get(mode, {}), f"[theme.{mode}] is missing {key}"
        assert "sidebar" in theme.get(mode, {}), f"[theme.{mode}.sidebar] missing"


def test_text_pairs_meet_aa_in_both_themes():
    for base in ("light", "dark"):
        colors = palette(base)
        for name in ("text", "muted", "link"):
            for surface in ("bg", "bg2", "sidebar_bg"):
                value = contrast(colors[name], colors[surface])
                assert value >= TEXT_MIN, (
                    f"{base}: {name} on {surface} is {value:.2f}:1, below {TEXT_MIN}"
                )


def test_boundaries_and_indicators_meet_three_to_one():
    for base in ("light", "dark"):
        colors = palette(base)
        pairs = [
            ("border on page", colors["border"], colors["bg"]),
            ("border on widget", colors["border"], colors["bg2"]),
            ("sidebar border", colors["sidebar_border"], colors["sidebar_bg"]),
            ("focus ring on page", colors["primary"], colors["bg"]),
            ("focus ring on widget", colors["primary"], colors["bg2"]),
            # A filled primary control carries a white icon or label.
            ("white on primary fill", "#ffffff", colors["primary"]),
        ]
        for label, foreground, background in pairs:
            value = contrast(foreground, background)
            assert value >= BOUNDARY_MIN, (
                f"{base}: {label} is {value:.2f}:1, below {BOUNDARY_MIN}"
            )


def test_the_stylesheet_owns_no_colours():
    """Colours belong to the theme, so the reader's light/dark choice wins and
    there is exactly one place to verify. A hex literal in the CSS means a
    component that will be wrong in one of the two themes."""
    css = CSS.read_text()
    found = re.findall(r"#[0-9a-fA-F]{3,8}\b", css)
    assert not found, f"app.css hardcodes colours: {found}"


def test_the_stylesheet_does_not_fake_secondary_text_with_transparency():
    """Opacity creates a colour nobody measures.

    The audited defect was light grey text on a light surface; re-breaking it
    with `opacity: 0.7` instead of a lighter hex would slip past both the colour
    test and a reviewer. Hierarchy comes from size and weight instead.
    """
    declarations = re.findall(r"^\s*opacity\s*:", CSS.read_text(), flags=re.M)
    assert not declarations, f"app.css mutes text with opacity: {declarations}"
