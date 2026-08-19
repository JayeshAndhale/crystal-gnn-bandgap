"""
Tier 1, Pass 1: pull a filtered, sampled pool of candidate material IDs
from Materials Project. No structures yet — just the cheap summary fields,
so this runs fast and lets us sanity-check filtering before the slow pull.
"""

import json
import os
import random
from pathlib import Path

from dotenv import load_dotenv
from mp_api.client import MPRester

# --- config -----------------------------------------------------------
STABILITY_CUTOFF = (0, 0.05)   # energy_above_hull range in eV/atom
SAMPLE_SIZE = 5000
RANDOM_SEED = 42                # fixed seed = reproducible sample, defensible in interviews
OUTPUT_PATH = Path("data/raw/candidate_pool.json")

# --- load API key from .env, not hardcoded ----------------------------
load_dotenv()
api_key = os.environ["MP_API_KEY"]  # KeyError here means .env isn't loading — fail loud, not silent


def fetch_candidate_pool():
    """Query MP for stable-ish structures, lightweight fields only."""
    with MPRester(api_key) as mpr:
        docs = mpr.materials.summary.search(
            energy_above_hull=STABILITY_CUTOFF,
            nelements=(2, None),  #excludes elemental phases (trivial bonding, ~0 formation energy)
            fields=[
                "material_id",
                "formula_pretty",
                "band_gap",
                "formation_energy_per_atom",
                "energy_above_hull",
            ],
        )
    return docs


def summarize(docs):
    n = len(docs)
    band_gaps = [d.band_gap for d in docs if d.band_gap is not None]
    n_metals = sum(1 for bg in band_gaps if bg == 0.0)
    print(f"Total candidates after stability filter: {n}")
    print(f"  Metals (band_gap == 0): {n_metals} ({n_metals/n:.1%})")
    print(f"  Band gap range: {min(band_gaps):.2f}–{max(band_gaps):.2f} eV")


def sample_and_save(docs, k, seed):
    random.seed(seed)
    sampled = random.sample(docs, k)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    records = [
        {
            "material_id": str(d.material_id),
            "formula_pretty": d.formula_pretty,
            "band_gap": d.band_gap,
            "formation_energy_per_atom": d.formation_energy_per_atom,
            "energy_above_hull": d.energy_above_hull,
        }
        for d in sampled
    ]
    with open(OUTPUT_PATH, "w") as f:
        json.dump(records, f, indent=2)

    print(f"\nSampled {len(records)} candidates -> {OUTPUT_PATH}")


if __name__ == "__main__":
    docs = fetch_candidate_pool()
    summarize(docs)
    sample_and_save(docs, SAMPLE_SIZE, RANDOM_SEED)