# ─────────────────────────────────────────────
# Demo 1 — Configuration
# Edit this file to adapt the app to your own documents.
# No changes needed in app.py.
# ─────────────────────────────────────────────

# App branding
APP_TITLE    = "O&G Document Intelligence"
APP_SUBTITLE = "Ask questions from IOGP standards · JIP33 specifications · Process safety guidelines"

# Document sources shown in sidebar
# Update this list to match your actual PDFs
DOCUMENT_SOURCES = [
    "IOGP Report 459 — Life-Saving Rules",
    "IOGP Report 456 — Process Safety KPIs",
    "JIP33 S-737 — Deluge Skids (TRS + QRS)",
    "JIP33 S-717 — Noise Equipment (TRS + QRS)",
    "JIP33 S-719 — Water Mist Fire Protection",
]

# Sample questions shown as clickable buttons in sidebar
# Replace with questions relevant to your own documents
SAMPLE_QUESTIONS = {
    "🦺 HSE Rules": [
        "What are the life saving rules?",
        "What must I confirm before entering a confined space?",
        "What are the hot work requirements in a hazardous area?",
    ],
    "📊 Process Safety": [
        "What is the difference between Tier 1 and Tier 2 process safety events?",
        "How are process safety KPIs measured?",
        "What does LOPC stand for and what are its consequences?",
    ],
    "⚙️ Equipment Standards": [
        "What does IOGP S-737 specify for deluge skid design?",
        "What standards does S-737 reference for electrical installations?",
        "What does IOGP S-717 cover for noise emitting equipment?",
    ],
    "🇮🇩 Bahasa Indonesia": [
        "Apa saja aturan keselamatan jiwa menurut IOGP?",
        "Apa yang harus dilakukan sebelum memasuki ruang tertutup?",
        "Apa perbedaan antara kejadian keselamatan proses Tier 1 dan Tier 2?",
    ],
}

# Capability cards shown on empty state (first load)
# Replace with descriptions relevant to your own documents
CAPABILITY_CARDS = [
    {
        "title": "🦺 HSE & Safety Rules",
        "desc": (
            "Ask about IOGP Life-Saving Rules, confined space entry, "
            "hot work requirements, energy isolation, working at height."
        ),
    },
    {
        "title": "📊 Process Safety KPIs",
        "desc": (
            "Tier 1 and Tier 2 process safety events, LOPC definitions, "
            "consequence thresholds, KPI measurement frameworks."
        ),
    },
    {
        "title": "⚙️ Equipment Specifications",
        "desc": (
            "JIP33 S-737 deluge skids, S-717 noise equipment, "
            "S-719 water mist fire protection — technical and quality requirements."
        ),
    },
    {
        "title": "🇮🇩 Bahasa Indonesia",
        "desc": (
            "Tanya dalam Bahasa Indonesia. Sistem mendeteksi bahasa "
            "otomatis dan menjawab dalam bahasa yang sama."
        ),
    },
]

# Friendly display names for raw PDF filenames
# Used in source citations — maps filename → readable name
# Add entries for your own PDFs
SOURCE_FRIENDLY = {
    "459.pdf":                                    "IOGP 459 Life-Saving Rules",
    "456.pdf":                                    "IOGP 456 Process Safety KPIs",
    "S-737v2026-03 TRS.pdf":                      "S-737 Deluge Skids (Technical)",
    "S-737Qv2026-03 QRS.pdf":                     "S-737 Deluge Skids (Quality)",
    "S-717v2025-03 TRS.pdf":                      "S-717 Noise Equipment (Technical)",
    "S-717Qv2020-06 QRS.pdf":                     "S-717 Noise Equipment (Quality)",
    "S-719v2025-01 TRS.pdf":                      "S-719 Water Mist (Technical)",
    "S-719Qv2025-01 QRS.pdf":                     "S-719 Water Mist (Quality)",
    "S-719Jv2025-01 TRS with Justification.pdf":  "S-719 Water Mist (Justification)",
}