# Demo 1 design direction

Direction for the Streamlit interface. The rules that enforce it are
tests/test_theme_contrast.py (WCAG, both modes) and the copy rules in
tests/test_app.py (no emoji, no em dash in user-facing text).

## Read

Reading this as: a document reader for upstream oil and gas engineers, in an
archival and precise visual language, dial **ENERGY 1 / RHYTHM 1 / MOTION 1**.

Calm on purpose. This is a tool someone uses while checking a specification on
a laptop in daylight or on a phone at a site, not a landing page that has to
say hello. The document is the hero; the interface gets out of its way.

## Who it is for

- Site and HSE engineers: time pressure, mobile, need an answer they can verify.
- Procurement and contracts engineers: know the vocabulary, check compliance.
- IT and digital evaluators: judge whether this looks like a product.

## Palette

Warm paper and ink, with one amber accent drawn from the industrial subject.
The accent marks actions and focus only: zero accents is sterile, an accent
everywhere is decoration.

Values live in `.streamlit/config.toml`, in `[theme.light]` and `[theme.dark]`
so a mode switch actually switches the palette. `src/app.css` is forbidden from
owning a colour: it sets metrics, rhythm and one motif, and inherits everything
else, which is what makes the reader's light or dark choice work.

Both palettes are measured, not chosen by eye: 4.5:1 for text, 3:1 for
boundaries, focus rings and icons.

## Typography

IBM Plex Sans for text and headings, IBM Plex Mono for identifiers and quoted
source text, where a fixed-width face genuinely helps (clause numbers, file
names). Mono is never used as an aesthetic on labels, never below 0.8125rem,
and no uppercase micro-labels with wide tracking.

## Structure

- One column at about 80 characters: specification clauses are read, not
  scanned.
- The empty state is an index of what this corpus answers, not a grid of
  identical cards.
- The one repeated motif is the evidence block: a claim and the quoted excerpt
  that supports it, always adjacent, because verifiability is the product.
- The sidebar holds the corpus and the sample questions, and nothing else.

## Copy

Plain, specific, sentence case. No emoji, no em dashes, no marketing
superlatives. When the app cannot answer, it says so and says why.
