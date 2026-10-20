"""Rebuild vector assets with fontTools and an upstream Manrope variable font.

Usage: python docs/branding/generate_logo.py --font /path/to/Manrope[wght].ttf
The font is only needed at build time; all distributed lettering is outlined.
"""

import argparse
from pathlib import Path
from xml.sax.saxutils import escape

from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.ttLib import TTFont
from fontTools.varLib.instancer import instantiateVariableFont

ROOT = Path(__file__).resolve().parents[2]
STATIC = ROOT / "docs" / "_static"
TOP = "M16 48 64 12 112 40 112 64C76 88 52 24 16 48Z"
BOTTOM = "M16 64C52 40 76 104 112 80L64 116 16 88Z"
# A slightly broader tip and flatter curve preserve the identity at 16–24 px.
MICRO_TOP = "M16 44 64 12 112 40 112 60C76 80 52 28 16 52Z"
MICRO_BOTTOM = "M16 68C52 48 76 100 112 76L112 84 64 116 16 88Z"
PALETTES = {
    "": ("#4355DB", "#008577", "#18243B", "#526078"),
    "-dark": ("#929DFF", "#49D5BE", "#F1F4FC", "#B2BDD1"),
    "-mono": ("#000000",) * 4,
    "-white": ("#FFFFFF",) * 4,
}


def icon(colors, micro=False):
    a, b = (MICRO_TOP, MICRO_BOTTOM) if micro else (TOP, BOTTOM)
    return f'<path fill="{colors[0]}" d="{a}"/><path fill="{colors[1]}" d="{b}"/>'


def svg(width, height, body, title, description):
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">\n'
        f'  <title id="title">{escape(title)}</title>\n'
        f'  <desc id="desc">{escape(description)}</desc>\n  {body}\n</svg>\n'
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--font", required=True, type=Path)
    args = parser.parse_args()
    source = TTFont(args.font)
    fonts = {w: instantiateVariableFont(source, {"wght": w}, inplace=False) for w in (500, 650, 750)}

    def lettering(value, x, y, size, weight, fill, tracking=0):
        font = fonts[weight]
        glyphs, cmap = font.getGlyphSet(), font.getBestCmap()
        scale = size / font["head"].unitsPerEm
        parts = []
        for char in value:
            glyph = glyphs[cmap[ord(char)]]
            pen = SVGPathPen(glyphs)
            glyph.draw(pen)
            if pen.getCommands():
                parts.append(
                    f'<path transform="translate({x:.3f} {y}) scale({scale:.7f} {-scale:.7f})" d="{pen.getCommands()}"/>'
                )
            x += glyph.width * scale + tracking
        return f'<g fill="{fill}">{"".join(parts)}</g>', x

    description = (
        "Two curved quadrilateral Bézier cells, separated by an S-shaped boundary, "
        "represent product-of-simplices domains and smooth fitting."
    )
    for suffix, colors in PALETTES.items():
        mark = f'<g transform="translate(4 8)">{icon(colors)}</g>'
        word, end = lettering("PyTorch", 158, 76, 43, 500, colors[2])
        dash, end = lettering("-", end + 2, 76, 43, 500, colors[2])
        bsf, end = lettering("BSF", end + 2, 76, 43, 750, colors[2])
        sub, _ = lettering("BÉZIER SIMPLOID FITTING", 160, 106, 12.5, 650, colors[3], 1.3)
        body = mark + word + dash + bsf + sub
        STATIC.joinpath(f"logo{suffix}.svg").write_text(
            svg(448, 144, body, "PyTorch-BSF — Bézier Simploid Fitting", description)
        )
        STATIC.joinpath(f"logo-icon{suffix}.svg").write_text(svg(128, 128, icon(colors), "PyTorch-BSF", description))
        compact = mark + f'<g transform="translate(0 12)">{word}{dash}{bsf}</g>'
        STATIC.joinpath(f"logo-compact{suffix}.svg").write_text(svg(448, 144, compact, "PyTorch-BSF", description))
        if suffix in ("", "-dark"):
            background = "#FFFFFF" if not suffix else "#101827"
            banner = (
                f'<path fill="{background}" d="M0 0H1200V360H0Z"/><g transform="translate(196.8 50.4) scale(1.8)">{body}</g>'
            )
            STATIC.joinpath(f"logo-banner{suffix}.svg").write_text(
                svg(
                    1200,
                    360,
                    banner,
                    "PyTorch-BSF — Bézier Simploid Fitting",
                    description,
                )
            )

    # The favicon has its own dark backing, so it is independent of browser theme.
    favicon = '<rect width="128" height="128" rx="28" fill="#18243B"/>'
    favicon += f'<g transform="translate(4 4) scale(.9375)">{icon(PALETTES["-dark"], micro=True)}</g>'
    STATIC.joinpath("favicon.svg").write_text(svg(128, 128, favicon, "PyTorch-BSF", description))


if __name__ == "__main__":
    main()
