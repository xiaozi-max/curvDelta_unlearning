"""Select IDs from an explicit caller-owned GPU pool; no external dependencies."""
import argparse
import re


def select_ids(pool: str, count: int) -> str:
    ids = [item for item in re.split(r"[,\s]+", pool.strip()) if item]
    if count < 1 or count > len(ids) or len(set(ids)) != len(ids):
        raise ValueError("GPU count must fit a nonempty pool of unique IDs")
    if any(not re.fullmatch(r"[A-Za-z0-9_-]+", item) for item in ids):
        raise ValueError("GPU IDs must be numeric IDs or GPU/MIG UUIDs")
    return ",".join(ids[:count])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ids", required=True, help="Comma- or space-separated GPU IDs")
    parser.add_argument("--count", type=int, required=True)
    args = parser.parse_args()
    try:
        print(select_ids(args.ids, args.count))
    except ValueError as error:
        parser.error(str(error))
