"""Create company_names.csv and register_key_hashes.npy from the Companies House company data.

Usage: python prepare_dataset.py --download
       python prepare_dataset.py --source <BasicCompanyDataAsOneFile-YYYY-MM-DD.zip>
"""

import argparse
import csv
import hashlib
import io
import random
import re
import sys
import unicodedata
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np

from constants import (
    SNAPSHOT, SOURCE_URL, SOURCE_SHA256, DATA_FILE, REGISTER_FILE, SEED, SAMPLE_SIZE,
    TEST_SHARE, VALIDATION_SHARE, CATEGORY, MAX_LENGTH, ALLOWED_CHARACTERS, NAME_ENDINGS,
    IGNORED_ENDINGS, SPECIAL_LETTERS, SYMBOLS, WORDS, DIGITS, LEGAL_ENDINGS, OTHER_ENDINGS,
)


def sha256_of_file(path):
    """Return the SHA-256 hex digest of a file."""
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def clean_name(name):
    """Return the name with typographic apostrophes replaced and the spaces collapsed."""
    name = name.replace("’", "'").replace("‘", "'")
    return " ".join(name.split())


def name_words(name):
    """Return the words of a name without punctuation, a leading THE or WWW and its endings."""
    text = name.upper().replace("&", " AND ").replace("'", "").replace(".", "")
    words = re.sub(r"[^A-Z0-9 ]", " ", text).split()
    while words and words[0] in ("THE", "WWW"):
        words = words[1:]
    return strip_endings(words, IGNORED_ENDINGS)


def fold_letters(text):
    """Return the text in capitals with accented letters replaced by their base letter."""
    text = "".join(SPECIAL_LETTERS.get(char, char) for char in text.upper())
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(char for char in decomposed if not unicodedata.combining(char))


def strip_endings(words, endings):
    """Remove the given endings from the end of a word list, as long as one word is left."""
    stripped = True
    while stripped:
        stripped = False
        for ending in endings:
            ending_words = ending.split()
            if len(words) > len(ending_words) and words[-len(ending_words):] == ending_words:
                words = words[:-len(ending_words)]
                stripped = True
    return words


def same_as_key(name):
    """Return the key of a name for the register check, my own version of the "same as" rules."""
    text = fold_letters(" ".join(name.split())[:MAX_LENGTH])
    # these signs are ignored among the first three characters
    text = "".join(char for char in text[:3] if char not in "*=#%+") + text[3:]
    for symbol, word in SYMBOLS.items():
        text = text.replace(symbol, word)
    text = " ".join(re.sub(r"[^A-Z0-9 ]", " ", text).split())
    text = re.sub(r"\bPER CENT(UM)?\b", "PERCENTUM", text)
    words = text.split()
    while len(words) > 1 and words[0] in ("THE", "WWW"):
        words = words[1:]
    words = strip_endings(strip_endings(words, LEGAL_ENDINGS), OTHER_ENDINGS)
    key = "".join(WORDS.get(word, word) for word in words)
    key = "".join(DIGITS.get(char, char) for char in key)
    # a final S is ignored, so "HANDS" and "HAND'S" give the same key
    if key.endswith("S") and len(key) > 1:
        key = key[:-1]
    return key


def key_hash(key):
    """Return a 64-bit hash of a key, the same on every machine (unlike Python's hash())."""
    return int.from_bytes(hashlib.blake2b(key.encode("utf-8"), digest_size=8).digest(), "little")


def read_register(source_path):
    """Read the snapshot once and return the key hashes of all live companies and the kept names."""
    hashes = []
    names = []
    removed = Counter()
    with zipfile.ZipFile(source_path) as archive, archive.open(archive.namelist()[0]) as raw:
        reader = csv.reader(io.TextIOWrapper(raw, encoding="utf-8", newline=""))
        # some column names start with a space
        header = [column.strip() for column in next(reader)]
        name_column = header.index("CompanyName")
        category_column = header.index("CompanyCategory")
        for row in reader:
            if len(row) != len(header):
                removed["empty or broken row"] += 1
                continue
            name = clean_name(row[name_column])
            hashes.append(key_hash(same_as_key(name)))
            if len(hashes) % 1_000_000 == 0:
                print(f"processed {len(hashes):,} rows", flush=True)
            if row[category_column] != CATEGORY:
                removed["other category"] += 1
            elif not name:
                removed["empty name"] += 1
            elif not set(name) <= ALLOWED_CHARACTERS:
                removed["character outside the allowed set"] += 1
            elif len(name) > MAX_LENGTH:
                removed[f"longer than {MAX_LENGTH} characters"] += 1
            elif name.split()[-1] not in NAME_ENDINGS:
                removed["other ending"] += 1
            else:
                names.append(name)
    broken = removed["empty or broken row"]
    print(f"{len(hashes):,} live companies in the snapshot, {broken} empty or broken rows skipped")
    reasons = ["other category", "empty name", "character outside the allowed set",
               f"longer than {MAX_LENGTH} characters", "other ending"]
    for reason in reasons:
        print(f"removed, {reason}: {removed[reason]:,}")
    print(f"{len(names):,} names kept")
    return np.unique(np.array(hashes, dtype=np.uint64)), names


def draw_sample(names):
    """Draw the random sample, the file is sorted by name, so the first rows would not do."""
    unique_names = sorted(set(names))
    print(f"{len(names) - len(unique_names):,} repeated names removed")
    return random.Random(SEED).sample(unique_names, SAMPLE_SIZE)


def group_key(name):
    """Return the first two words of the name without its ending as the key of its group."""
    words = name_words(name)
    return " ".join(words[:2]) if words else name


def assign_splits(sample):
    """Split the name groups about 80/10/10 into train, validation and test, one row per name."""
    keys = [group_key(name) for name in sample]
    groups = sorted(set(keys))
    random.Random(SEED).shuffle(groups)
    test_count = round(len(groups) * TEST_SHARE)
    validation_count = round(len(groups) * VALIDATION_SHARE)
    split_of_group = {}
    for position, group in enumerate(groups):
        if position < test_count:
            split_of_group[group] = "test"
        elif position < test_count + validation_count:
            split_of_group[group] = "validation"
        else:
            split_of_group[group] = "train"
    group_id = {group: number for number, group in enumerate(sorted(groups))}
    sizes = Counter(keys)
    shared = sum(1 for size in sizes.values() if size > 1)
    members = sum(size for size in sizes.values() if size > 1)
    print(f"{len(groups):,} name groups, {shared:,} of them with more than one name "
          f"({members:,} names)")
    return [{"name": name, "group": group_id[key], "split": split_of_group[key]}
            for name, key in zip(sample, keys)]


def check_rows(rows):
    """Stop if a name appears twice or a group is spread over two splits."""
    if len({row["name"] for row in rows}) != len(rows):
        sys.exit("A name appears more than once.")
    splits_per_group = {}
    for row in rows:
        splits_per_group.setdefault(row["group"], set()).add(row["split"])
    if any(len(splits) > 1 for splits in splits_per_group.values()):
        sys.exit("A name group is spread over more than one split.")


def print_splits(rows):
    """Print the number of names per split."""
    counts = Counter(row["split"] for row in rows)
    for split in ("train", "validation", "test"):
        print(f"{split}: {counts[split]:,} names ({counts[split] / len(rows):.1%})")


def write_output(rows, hashes, output_dir):
    """Write company_names.csv and register_key_hashes.npy and return their paths."""
    output_dir.mkdir(parents=True, exist_ok=True)
    data_path = output_dir / DATA_FILE
    with data_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["name", "group", "split"], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    keys_path = output_dir / REGISTER_FILE
    np.save(keys_path, hashes)
    return data_path, keys_path


def prepare(source_path, output_dir):
    """Run all steps and write the two output files into output_dir."""
    # only a warning and not a stop like in project 4, because Companies House lists only the
    # latest snapshot on its page and does not say how long it keeps the older files
    if sha256_of_file(source_path) != SOURCE_SHA256:
        print(f"Warning: {source_path} is not the snapshot of {SNAPSHOT}, "
              "the results will differ from the files in the repository.")
    hashes, names = read_register(source_path)
    rows = assign_splits(draw_sample(names))
    check_rows(rows)
    print_splits(rows)
    print(f"{len(hashes):,} different keys in the register")
    data_path, keys_path = write_output(rows, hashes, output_dir)
    print(f"wrote {data_path}, SHA-256 {sha256_of_file(data_path)}")
    print(f"wrote {keys_path}, SHA-256 {sha256_of_file(keys_path)}")


def download(target):
    """Download the snapshot and check its checksum."""
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(".partial")
    print(f"downloading {SOURCE_URL} (about 470 MB) ...", flush=True)
    urllib.request.urlretrieve(SOURCE_URL, partial)
    if sha256_of_file(partial) != SOURCE_SHA256:
        sys.exit("The downloaded file does not match the expected SHA-256.")
    partial.replace(target)


def main():
    """Parse the command line and run the preparation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=Path(f"data/BasicCompanyDataAsOneFile-{SNAPSHOT}.zip"),
        help=f"path of the snapshot (default: data/BasicCompanyDataAsOneFile-{SNAPSHOT}.zip)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("."),
        help="folder for the output files (default: current folder)",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="download the snapshot first if --source does not exist",
    )
    args = parser.parse_args()
    if not args.source.exists():
        if args.download:
            download(args.source)
        else:
            sys.exit(f"{args.source} not found. Run with --download or pass --source.")
    prepare(args.source, args.output)


if __name__ == "__main__":
    main()
