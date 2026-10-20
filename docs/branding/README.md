# PyTorch-BSF visual identity

Two curved quadrilateral cells form a compact geometric mark. A quadrilateral
represents a product of two 1-simplices, Δ¹ × Δ¹: a simploid, rather than only
a triangle. The cells are drawn apart along corresponding cubic Bézier edges;
the open channel suggests an S. Smooth fitted geometry is the central idea.
This is a symbol, not a literal surface or control-net diagram.

[Open the comparison sheet](preview.html) or [view its PNG](preview.png).
The sheet includes light and dark backgrounds, one-ink reproduction and
favicons displayed at 16, 24 and 32 CSS pixels.

## Assets

All assets are in [`../_static/`](../_static/). SVGs contain outlined lettering,
flat fills and transparent backgrounds, except the banners and favicon, which
have their own backgrounds. No installed fonts, linked images or effects are
required to display them.

| Use | Light background | Dark background |
| --- | --- | --- |
| Full signature with descriptor | `logo.svg` | `logo-dark.svg` |
| Compact signature without descriptor | `logo-compact.svg` | `logo-compact-dark.svg` |
| Standalone symbol | `logo-icon.svg` | `logo-icon-dark.svg` |
| 1200 × 360 banner | `logo-banner.svg` | `logo-banner-dark.svg` |
| One-ink print | `logo-mono.svg` | `logo-white.svg` |
| Compact one-ink print | `logo-compact-mono.svg` | `logo-compact-white.svg` |
| One-ink symbol | `logo-icon-mono.svg` | `logo-icon-white.svg` |

`logo.png` is a transparent 1344 × 432 raster fallback; `logo-icon.png` is
512 × 512. Prefer SVG for print. `favicon.svg` has a fixed dark rounded backing
and optically broader tips for small sizes; `favicon.ico` includes
16, 24, 32, 48, 64, 128 and 256 px images.

Use the full signature at **360 px wide or larger**, the compact signature
at **200 px or larger**, and the icon for smaller placements. Use the dedicated
favicon at 16–24 px. Keep at least one eighth of the icon canvas width clear
around the mark. Preserve the aspect ratio. For grayscale or one-color printing,
choose an explicit black or white variant so the result does not rely on color
separation. For complex backgrounds, place the logo on a solid light or dark panel.

The README selects light/dark assets using `<picture>`. Sphinx uses the white
compact signature on the Read the Docs theme's colored sidebar, and the ICO favicon.

## Color and lettering

| Role | Light version | Dark version |
| --- | --- | --- |
| Indigo surface | `#4355DB` | `#929DFF` |
| Teal surface | `#008577` | `#49D5BE` |
| Name | `#18243B` | `#F1F4FC` |
| Descriptor | `#526078` | `#B2BDD1` |

Lettering is Manrope, weight 500 for PyTorch, 750 for BSF, and 650 for the
descriptor. The upstream font is licensed under the SIL Open Font License;
see [Manrope-OFL.txt](Manrope-OFL.txt). The distributed SVGs use paths.
The PyTorch flame and scikit-learn shapes are not part of this identity.

Reference material: [PyTorch brand guidelines](https://pytorch.org/brand-guidelines/)
and [scikit-learn's official logo source](https://github.com/scikit-learn/scikit-learn/tree/main/doc/logos).
These served as references for simplicity, spacing and use at different sizes.

## Rebuild

The generator requires `fonttools` and the upstream Manrope variable TTF:

```sh
curl -L --fail 'https://raw.githubusercontent.com/google/fonts/main/ofl/manrope/Manrope%5Bwght%5D.ttf' -o /tmp/bsf-manrope.ttf
python docs/branding/generate_logo.py --font /tmp/bsf-manrope.ttf
```

The font used for these exports has SHA-256
`3ae11c49db0455a3cc33e37d380f20fdb8c7f8b41dc07625c177e3d87a9d6ae6`.
Upstream may change; use that font revision for byte-identical SVGs. The generator
rebuilds the SVGs. Raster exports and the preview PNG must be re-rendered after
changes; render the SVGs in a browser with a transparent background, then derive
the multi-size ICO from the favicon raster. No font is needed to open the assets.
