"""
Fabric-specific ease reference: how much room a WELL-FITTING garment of a
given fabric type should have relative to the wearer's own body, per body
part. Hand-provided reference data (not derived from any single real
product's size chart) -- garment ease depends heavily on fabric stretch
behavior (a rigid woven needs several cm of positive ease to allow movement;
a 4-way stretch knit can have zero or even NEGATIVE ease and still fit
comfortably, relying on stretch recovery instead of extra room).

Used by measurement_heatmap.py to score a REAL garment's actual measurements
(from real_garment.py's PRODUCTS) against what this fabric type would be
EXPECTED to need for this specific wearer's own predicted body measurements
-- entirely independent of the 3D garment-draping pipeline in
third_party/GarmentCode. That pipeline drapes a simulated 3D mesh and reads
tightness off it geometrically; this instead computes tightness directly
from measurements and colors the avatar's own body mesh. Two different,
independent ways of estimating fit -- this one doesn't need a garment mesh,
wrap/relax physics, or any of the failure modes that came with those.
"""

STANDARD_BODY_MEASUREMENTS_CM = {
    "S":    {"chest": 91,  "waist": 76,  "hip": 96,  "shoulder": 43,   "sleeve_length": 61,   "neck": 37},
    "M":    {"chest": 97,  "waist": 82,  "hip": 102, "shoulder": 44.5, "sleeve_length": 62.5, "neck": 38.5},
    "L":    {"chest": 103, "waist": 88,  "hip": 108, "shoulder": 46,   "sleeve_length": 64,   "neck": 40},
    "XL":   {"chest": 109, "waist": 94,  "hip": 114, "shoulder": 47.5, "sleeve_length": 65.5, "neck": 41.5},
    "XXL":  {"chest": 115, "waist": 100, "hip": 120, "shoulder": 49,   "sleeve_length": 67,   "neck": 43},
    "XXXL": {"chest": 121, "waist": 106, "hip": 126, "shoulder": 50.5, "sleeve_length": 68.5, "neck": 44.5},
}

# How wide the "good fit" color band is (+-tolerance_cm around a diff of 0)
# for each fabric category. Since the standardized chart's garment
# measurement is built as standard_body[size] + this fabric's own ease, the
# ease itself always cancels back out of (actual_ease - expected_ease) --
# comparing the SAME user against two fabrics at the same size gives an
# IDENTICAL raw diff for both (verified directly: cotton_jersey_knit and
# lycra_spandex_blend produced the same chest/waist/hip/shoulder numbers for
# an identical body). That's algebraically correct, but on its own it would
# make fabric choice cosmetic -- a stretch fabric's whole point is that it
# physically tolerates being cut with less room (or even negative ease)
# without actually feeling tight, while a rigid weave has no give to absorb
# the same shortfall. So the fabric's ease values shape the *garment* number,
# and its stretch behavior separately widens or narrows how far that same
# diff can go before it's shown as genuinely tight/loose rather than "fine."
# These widths are this project's own engineering judgment (not literally
# specified in the fabric chart, which only gave stretch_percent as a
# descriptive range) -- roughly proportional to each category's stretch
# capacity, replacing the previous fixed +-2cm used for every fabric alike.
CATEGORY_TOLERANCE_CM = {
    "woven": 2.0,               # 0-3% stretch: little to no give
    "woven_stretch": 3.0,       # 2-8% stretch: a bit of forgiveness
    "woven_light_stretch": 2.5, # 1-4% stretch (chinos): between woven and woven_stretch
    "knit": 3.5,                # 8-20% stretch: noticeably forgiving
    "stretch_knit": 5.5,        # 25-50% stretch: wide comfortable range
}

# ease_by_part_cm: how much bigger (or, if negative, smaller) than the body
# a well-fitting garment of this fabric should measure, per body part.
FABRIC_TYPES = {
    "cotton_woven": {
        "label": "Cotton Shirt (Woven, Non-stretch)",
        "category": "woven",
        "stretch_percent": "0-2%",
        "fit_style": "regular/formal fit",
        "tolerance_cm": CATEGORY_TOLERANCE_CM["woven"],
        "ease_by_part_cm": {"chest": 8, "waist": 6, "hip": 6, "shoulder": 1, "sleeve_length": 2, "neck": 1.5},
    },
    "cotton_jersey_knit": {
        "label": "Cotton T-shirt (Single Jersey Knit)",
        "category": "knit",
        "stretch_percent": "10-20%",
        "fit_style": "regular fit",
        "tolerance_cm": CATEGORY_TOLERANCE_CM["knit"],
        "ease_by_part_cm": {"chest": 4, "waist": 4, "hip": 4, "shoulder": 0.5, "sleeve_length": 1.5, "neck": 1},
    },
    "lycra_spandex_blend": {
        "label": "Lycra/Spandex Blend Shirt (4-way stretch)",
        "category": "stretch_knit",
        "stretch_percent": "30-50%",
        "fit_style": "slim/compression fit",
        "tolerance_cm": CATEGORY_TOLERANCE_CM["stretch_knit"],
        "ease_by_part_cm": {"chest": -2, "waist": -2, "hip": -1.5, "shoulder": 0, "sleeve_length": 0.5, "neck": 0},
    },
    "denim_rigid": {
        "label": "Denim (Rigid Woven)",
        "category": "woven",
        "stretch_percent": "0-1%",
        "fit_style": "regular fit",
        "tolerance_cm": CATEGORY_TOLERANCE_CM["woven"],
        "ease_by_part_cm": {"chest": 9, "waist": 7, "hip": 7, "shoulder": 1.5, "sleeve_length": 2, "neck": 1.5},
    },
    "denim_stretch": {
        "label": "Denim (Stretch/Comfort Denim, ~2% elastane)",
        "category": "woven_stretch",
        "stretch_percent": "2-8%",
        "fit_style": "slim fit",
        "tolerance_cm": CATEGORY_TOLERANCE_CM["woven_stretch"],
        "ease_by_part_cm": {"chest": 5, "waist": 3, "hip": 3, "shoulder": 1, "sleeve_length": 1.5, "neck": 1},
    },
    "polyester_formal": {
        "label": "Polyester Formal Shirt (Woven, minimal stretch)",
        "category": "woven",
        "stretch_percent": "0-3%",
        "fit_style": "regular/formal fit",
        "tolerance_cm": CATEGORY_TOLERANCE_CM["woven"],
        "ease_by_part_cm": {"chest": 7, "waist": 5.5, "hip": 5.5, "shoulder": 1, "sleeve_length": 2, "neck": 1.5},
    },
    "poly_cotton_knit": {
        "label": "Polyester-Cotton Blend T-shirt (Knit)",
        "category": "knit",
        "stretch_percent": "8-15%",
        "fit_style": "regular fit",
        "tolerance_cm": CATEGORY_TOLERANCE_CM["knit"],
        "ease_by_part_cm": {"chest": 4.5, "waist": 4.5, "hip": 4, "shoulder": 0.5, "sleeve_length": 1.5, "neck": 1},
    },
    "rib_knit": {
        "label": "Rib Knit (Fitted tees/bodysuits)",
        "category": "stretch_knit",
        "stretch_percent": "25-40%",
        "fit_style": "fitted/second-skin",
        "tolerance_cm": CATEGORY_TOLERANCE_CM["stretch_knit"],
        "ease_by_part_cm": {"chest": -3, "waist": -3, "hip": -2, "shoulder": 0, "sleeve_length": 0, "neck": -0.5},
    },
    "linen": {
        "label": "Linen Shirt (Woven, no stretch)",
        "category": "woven",
        "stretch_percent": "0%",
        "fit_style": "relaxed fit",
        "tolerance_cm": CATEGORY_TOLERANCE_CM["woven"],
        "ease_by_part_cm": {"chest": 10, "waist": 8, "hip": 8, "shoulder": 2, "sleeve_length": 2.5, "neck": 1.5},
    },
    "fleece_sweatshirt": {
        "label": "Fleece/Sweatshirt Fabric (Knit, brushed)",
        "category": "knit",
        "stretch_percent": "5-12%",
        "fit_style": "loose/relaxed fit",
        "tolerance_cm": CATEGORY_TOLERANCE_CM["knit"],
        "ease_by_part_cm": {"chest": 10, "waist": 9, "hip": 8, "shoulder": 2, "sleeve_length": 2, "neck": 1.5},
    },
    # Below: added from a separate, richer garment catalog (user-provided),
    # each ease_by_part_cm part an explicit {"min","ideal","max"} dict
    # rather than one plain ideal number -- see _compute_fit_label's
    # _resolve_band for how both forms are handled uniformly. These are
    # NEW keys, not replacements for the hand-estimated entries above (e.g.
    # "cotton_single_jersey" here is NOT the same entry as "cotton_jersey_
    # knit" above -- different source, kept separate rather than silently
    # overwriting a fabric that nobero_polo/graphic_oversized_tee already
    # depend on for their own fit_label).
    "cotton_single_jersey": {
        "label": "Cotton Single Jersey Crew Neck T-Shirt (Regular Fit)",
        "category": "knit",
        "stretch_percent": "20% width, 10% length",
        "fit_style": "regular fit",
        "tolerance_cm": CATEGORY_TOLERANCE_CM["knit"],
        "ease_by_part_cm": {
            "chest": {"min": 4, "ideal": 8, "max": 12},
            "waist": {"min": 6, "ideal": 10, "max": 16},
            "shoulder": {"min": 0, "ideal": 1, "max": 2},
        },
    },
    "cotton_poplin": {
        "label": "Cotton Poplin Formal Button-Down Shirt",
        "category": "woven",
        "stretch_percent": "0%",
        "fit_style": "regular fit",
        "tolerance_cm": CATEGORY_TOLERANCE_CM["woven"],
        "ease_by_part_cm": {
            "chest": {"min": 8, "ideal": 12, "max": 16},
            "waist": {"min": 8, "ideal": 12, "max": 18},
            "shoulder": {"min": 0.5, "ideal": 1.5, "max": 2.5},
            "neck": {"min": 1, "ideal": 2, "max": 3},
        },
    },
    "rayon_challis": {
        "label": "Rayon Challis A-Line Maxi Dress",
        "category": "woven",
        "stretch_percent": "0%",
        "fit_style": "relaxed (fitted bust, flowy skirt)",
        "tolerance_cm": CATEGORY_TOLERANCE_CM["woven"],
        "ease_by_part_cm": {
            "chest": {"min": 4, "ideal": 6, "max": 10},   # bust, reusing "chest" per dress-kind convention
            "waist": {"min": 3, "ideal": 6, "max": 10},
            "hip": {"min": 12, "ideal": 25, "max": 60},   # deliberately wide/lopsided -- a flared skirt barely constrains hip ease at all
        },
    },
    "cotton_spandex_jersey": {
        "label": "Cotton-Spandex Jersey Bodycon Maxi Dress",
        "category": "stretch_knit",
        "stretch_percent": "50% width, 30% length",
        "fit_style": "body-hugging",
        "tolerance_cm": CATEGORY_TOLERANCE_CM["stretch_knit"],
        "ease_by_part_cm": {
            "chest": {"min": -4, "ideal": -2, "max": 1},
            "waist": {"min": -4, "ideal": -2, "max": 1},
            "hip": {"min": -5, "ideal": -3, "max": 0},
        },
    },
}

# Which fabric type best represents each existing real_garment.py product,
# so the heatmap has a sensible default without needing a manual picker yet.
PRODUCT_FABRIC_TYPE = {
    "nobero_polo": "cotton_jersey_knit",
    "nobero_sweatshirt": "fleece_sweatshirt",
    "graphic_oversized_tee": "cotton_jersey_knit",
    "nobero_cargo_pants": "denim_rigid",
    "shasmi_maxi_dress": "linen",
    "tshirt_cotton_jersey": "cotton_single_jersey",
    "shirt_cotton_poplin": "cotton_poplin",
    "trousers_cotton_twill": "cotton_twill",
    "jeans_stretch_denim": "stretch_denim",
    "leggings_nylon_spandex": "nylon_spandex",
    "maxi_cotton_spandex": "cotton_spandex_jersey",
}

# =========================================================
# Bottoms (pants/shorts) -- same generic size+fabric mechanism as tops
# above, but with waist/hip/thigh/knee/ankle_hem/inseam/outseam/rise
# instead of chest/shoulder/sleeve_length/neck. Kept as separate dicts
# rather than merged into STANDARD_BODY_MEASUREMENTS_CM/FABRIC_TYPES:
# a bottoms fabric's ease_by_part_cm uses a completely different set of
# body parts than a tops fabric, so merging them would mean every fabric
# carries irrelevant keys for the other garment kind.
# =========================================================

# Keyed by waist size IN INCHES (as a string), not letter sizes -- this
# matches how pants are actually sized in the store this data came from
# (unlike tops, which use S/M/L/XL/XXL/XXXL). Each size's own "waist_in"
# duplicates the dict key as a number rather than a string; not stored as
# a measurement field here since it's not a body PART with an ease value
# (every fabric's ease_by_part_cm would need a meaningless "waist_in"
# entry otherwise) -- the key itself already carries that information.
STANDARD_BOTTOMS_MEASUREMENTS_CM = {
    "28": {"waist": 71.1,  "hip": 91.1,  "thigh": 51.6, "knee": 35.8, "ankle_hem": 33.2, "inseam": 74.4, "outseam": 97.6,  "rise": 25.2},
    "30": {"waist": 76.2,  "hip": 96.2,  "thigh": 54.1, "knee": 37.1, "ankle_hem": 34.0, "inseam": 76.1, "outseam": 100.1, "rise": 26.0},
    "32": {"waist": 81.3,  "hip": 101.3, "thigh": 56.6, "knee": 38.3, "ankle_hem": 34.9, "inseam": 77.8, "outseam": 102.6, "rise": 26.9},
    "34": {"waist": 86.4,  "hip": 106.4, "thigh": 59.2, "knee": 39.6, "ankle_hem": 35.7, "inseam": 79.5, "outseam": 105.2, "rise": 27.7},
    "36": {"waist": 91.4,  "hip": 111.4, "thigh": 61.7, "knee": 40.9, "ankle_hem": 36.6, "inseam": 81.2, "outseam": 107.7, "rise": 28.6},
    "38": {"waist": 96.5,  "hip": 116.5, "thigh": 64.3, "knee": 42.1, "ankle_hem": 37.4, "inseam": 82.8, "outseam": 110.3, "rise": 29.4},
    "40": {"waist": 101.6, "hip": 121.6, "thigh": 66.8, "knee": 43.4, "ankle_hem": 38.3, "inseam": 84.5, "outseam": 112.8, "rise": 30.3},
}

BOTTOMS_FABRIC_TYPES = {
    "cotton_formal_trousers": {
        "label": "Cotton Formal Trousers (Woven, Non-stretch)",
        "category": "woven",
        "stretch_percent": "0-2%",
        "fit_style": "regular/formal fit",
        "tolerance_cm": CATEGORY_TOLERANCE_CM["woven"],
        "ease_by_part_cm": {"waist": 3, "hip": 6, "thigh": 5, "knee": 3, "ankle_hem": 4, "rise": 2},
    },
    "denim_rigid_bottoms": {
        "label": "Rigid Denim Jeans (Woven, no stretch)",
        "category": "woven",
        "stretch_percent": "0-1%",
        "fit_style": "regular fit",
        "tolerance_cm": CATEGORY_TOLERANCE_CM["woven"],
        "ease_by_part_cm": {"waist": 2.5, "hip": 6, "thigh": 4, "knee": 2.5, "ankle_hem": 3, "rise": 1.5},
    },
    "denim_stretch_bottoms": {
        "label": "Stretch Denim Jeans (~2% elastane)",
        "category": "woven_stretch",
        "stretch_percent": "2-8%",
        "fit_style": "slim fit",
        "tolerance_cm": CATEGORY_TOLERANCE_CM["woven_stretch"],
        "ease_by_part_cm": {"waist": 0.5, "hip": 3, "thigh": 1.5, "knee": 1, "ankle_hem": 1.5, "rise": 1},
    },
    "lycra_leggings": {
        "label": "Lycra/Spandex Leggings (4-way stretch)",
        "category": "stretch_knit",
        "stretch_percent": "30-50%",
        "fit_style": "compression/second-skin",
        "tolerance_cm": CATEGORY_TOLERANCE_CM["stretch_knit"],
        "ease_by_part_cm": {"waist": -4, "hip": -3, "thigh": -2.5, "knee": -1.5, "ankle_hem": -1.5, "rise": -1},
    },
    "cotton_chinos": {
        "label": "Cotton Chinos (Woven, light stretch)",
        "category": "woven_light_stretch",
        "stretch_percent": "1-4%",
        "fit_style": "tailored fit",
        "tolerance_cm": CATEGORY_TOLERANCE_CM["woven_light_stretch"],
        "ease_by_part_cm": {"waist": 2, "hip": 5, "thigh": 3.5, "knee": 2, "ankle_hem": 3, "rise": 1.5},
    },
    "polyester_formal_trousers": {
        "label": "Polyester Formal Trousers (Woven, minimal stretch)",
        "category": "woven",
        "stretch_percent": "0-3%",
        "fit_style": "regular/formal fit",
        "tolerance_cm": CATEGORY_TOLERANCE_CM["woven"],
        "ease_by_part_cm": {"waist": 3, "hip": 6, "thigh": 5, "knee": 3, "ankle_hem": 4, "rise": 2},
    },
    "fleece_joggers": {
        "label": "Fleece Joggers (Knit, brushed, elastic waist)",
        "category": "knit",
        "stretch_percent": "8-15%",
        "fit_style": "loose/relaxed fit",
        "tolerance_cm": CATEGORY_TOLERANCE_CM["knit"],
        "ease_by_part_cm": {"waist": -1, "hip": 7, "thigh": 6, "knee": 4, "ankle_hem": -2, "rise": 3},
    },
    "cotton_shorts": {
        "label": "Cotton Shorts (Woven, non-stretch)",
        "category": "woven",
        "stretch_percent": "0-2%",
        "fit_style": "regular fit",
        "tolerance_cm": CATEGORY_TOLERANCE_CM["woven"],
        # Shorts don't reach the knee or ankle -- no ease value for those
        # parts at all (not zero, genuinely not applicable), same honest-
        # gap treatment used everywhere else in this file.
        "ease_by_part_cm": {"waist": 3, "hip": 6, "thigh": 5, "rise": 2},
    },
    "cargo_pants_heavy": {
        "label": "Cargo Pants (Heavy Woven, non-stretch)",
        "category": "woven",
        "stretch_percent": "0-2%",
        "fit_style": "relaxed/utility fit",
        "tolerance_cm": CATEGORY_TOLERANCE_CM["woven"],
        "ease_by_part_cm": {"waist": 4, "hip": 8, "thigh": 6, "knee": 4, "ankle_hem": 5, "rise": 2.5},
    },
    "athletic_track_pants": {
        "label": "Athletic Track Pants (Polyester Knit, stretch)",
        "category": "stretch_knit",
        "stretch_percent": "15-25%",
        "fit_style": "athletic relaxed fit",
        "tolerance_cm": CATEGORY_TOLERANCE_CM["stretch_knit"],
        "ease_by_part_cm": {"waist": -0.5, "hip": 4, "thigh": 3, "knee": 2, "ankle_hem": -1, "rise": 1.5},
    },
    # Same user-provided richer catalog as FABRIC_TYPES above -- new keys,
    # explicit {"min","ideal","max"} bands, not replacing existing entries.
    "cotton_twill": {
        "label": "Cotton Twill Regular Fit Trousers / Chinos",
        "category": "woven",
        "stretch_percent": "0%",
        "fit_style": "regular fit",
        "tolerance_cm": CATEGORY_TOLERANCE_CM["woven"],
        "ease_by_part_cm": {
            "waist": {"min": 1, "ideal": 2, "max": 3},
            "hip": {"min": 4, "ideal": 6, "max": 9},
            "thigh": {"min": 4, "ideal": 6, "max": 9},
        },
    },
    "stretch_denim": {
        "label": "Stretch Denim Slim Fit Jeans",
        "category": "woven_stretch",
        "stretch_percent": "15% width, 3% length",
        "fit_style": "slim fit",
        "tolerance_cm": CATEGORY_TOLERANCE_CM["woven_stretch"],
        "ease_by_part_cm": {
            "waist": {"min": -1, "ideal": 0, "max": 2},
            "hip": {"min": 0, "ideal": 2, "max": 4},
            "thigh": {"min": 0, "ideal": 2, "max": 4},
        },
    },
    "nylon_spandex": {
        "label": "Nylon-Spandex Compression Leggings",
        "category": "stretch_knit",
        "stretch_percent": "80% width, 60% length",
        "fit_style": "compression",
        "tolerance_cm": CATEGORY_TOLERANCE_CM["stretch_knit"],
        "ease_by_part_cm": {
            "waist": {"min": -10, "ideal": -7, "max": -4},
            "hip": {"min": -14, "ideal": -10, "max": -6},
            "thigh": {"min": -9, "ideal": -6, "max": -3},
        },
    },
}
