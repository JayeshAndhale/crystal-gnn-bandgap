"""
Tier 1, Pass 2: fetch full crystal structures for the sampled material IDs.
Chunked + resumable: writes one JSON line per material as it goes, so a
dropped connection mid-pull only costs the current chunk, not the whole run.
"""

import json
import os
from pathlib import Path

from dotenv import load_dotenv
from mp_api.client import MPRester
from monty.json import MontyEncoder

CANDIDATE_POOL_PATH = Path("data/raw/candidate_pool.json")
OUTPUT_PATH = Path("data/raw/structures.jsonl")
CHUNK_SIZE = 200

load_dotenv()
api_key = os.environ["MP_API_KEY"]


def load_target_ids():
    with open(CANDIDATE_POOL_PATH) as f:
        candidates = json.load(f)
    return [c["material_id"] for c in candidates]


def load_already_fetched():
    """Read whatever's already on disk, so a rerun skips completed work."""
    if not OUTPUT_PATH.exists():
        return set()
    fetched = set()
    with open(OUTPUT_PATH) as f:
        for line in f:
            if line.strip():
                fetched.add(json.loads(line)["material_id"])
    return fetched


def chunk(lst, size):
    for i in range(0, len(lst), size):
        yield lst[i : i + size]


def fetch_and_save(remaining_ids):
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_PATH, "a") as out_file:  # append mode — never overwrite
        with MPRester(api_key) as mpr:
            for batch_ids in chunk(remaining_ids, CHUNK_SIZE):
                docs = mpr.materials.summary.search(
                    material_ids=batch_ids,
                    fields=["material_id", "structure", "band_gap", "formation_energy_per_atom"],
                )
                for d in docs:
                    record = {
                        "material_id": str(d.material_id),
                        "structure": d.structure,  # pymatgen Structure object
                        "band_gap": d.band_gap,
                        "formation_energy_per_atom": d.formation_energy_per_atom,
                    }
                    out_file.write(json.dumps(record, cls=MontyEncoder) + "\n")
                out_file.flush()  # write to disk now, don't wait for buffer
                print(f"  fetched {len(docs)} structures (batch of {len(batch_ids)} requested)")


if __name__ == "__main__":
    target_ids = load_target_ids()
    already_fetched = load_already_fetched()
    remaining = [mid for mid in target_ids if mid not in already_fetched]

    print(f"Target: {len(target_ids)} | Already fetched: {len(already_fetched)} | Remaining: {len(remaining)}")

    if remaining:
        fetch_and_save(remaining)
    else:
        print("Nothing to do — all structures already fetched.")

    print(f"Done. Structures saved to {OUTPUT_PATH}")