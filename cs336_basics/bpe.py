import os
import regex as re
from typing import BinaryIO
import multiprocessing as mp
from collections import Counter, defaultdict
from pprint import pprint
from tqdm import tqdm
import json

NUM_PROC = 8
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(
        split_special_token, bytes
    ), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def init_vocab(special_tokens: list[str]) -> dict[int, bytes]:
    vocab = {i: bytes([i]) for i in range(256)}

    next_id = 256
    for tok in special_tokens:
        vocab[next_id] = tok.encode("utf-8")
        next_id += 1

    return vocab


def _pretokenize(args):
    path, start, end, special_tokens = args
    with open(path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
        parts = re.split("|".join([re.escape(tok) for tok in special_tokens]), chunk)

    # setup
    freq_dict = Counter()

    for part in parts:
        matches = re.finditer(PAT, part)
        for match in matches:
            token = match.group()
            freq_dict[tuple(token.encode("utf-8"))] += 1
    return freq_dict


def pretokenize(input_path, special_tokens):
    with open(input_path, "rb") as f:
        num_processes = 4
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

    tasks = [
        (input_path, start, end, special_tokens)
        for start, end in zip(boundaries[:-1], boundaries[1:])
    ]

    with mp.Pool(processes=NUM_PROC) as pool:
        partial_counts = pool.map(_pretokenize, tasks)

    freq_dict = sum(partial_counts, Counter())
    return freq_dict


def get_pair_counts_in_seq(seq):
    return Counter(zip(seq, seq[1:]))


def apply_merge_to_seq(seq, pair, new_id):
    """
    Left-to-right non-overlapping merge.
    Example:
        seq = (1, 1, 1), pair = (1, 1), new_id = 256
        returns (256, 1)
    """
    new_seq = []
    i = 0

    while i < len(seq):
        if i < len(seq) - 1 and (seq[i], seq[i + 1]) == pair:
            new_seq.append(new_id)
            i += 2
        else:
            new_seq.append(seq[i])
            i += 1

    return tuple(new_seq)


def merge(freq_dict, vocab, vocab_size):
    merges = []

    pair_counts = Counter()
    pair_to_ids = defaultdict(set)

    id_to_seq = {}
    id_to_count = {}

    # Assign each unique pre-token sequence an ID
    for idx, (seq, count) in enumerate(freq_dict.items()):
        id_to_seq[idx] = seq
        id_to_count[idx] = count

        for pair in zip(seq, seq[1:]):
            pair_counts[pair] += count
            pair_to_ids[pair].add(idx)

    num_merges = vocab_size - len(vocab)

    with tqdm(total=num_merges, desc="Training BPE merges") as pbar:
        while len(vocab) < vocab_size:
            # Pick most frequent pair.
            # Tie-break by actual byte values, not token IDs.
            most_freq_pair = max(
                pair_counts,
                key=lambda pair: (
                    pair_counts[pair],
                    vocab[pair[0]],
                    vocab[pair[1]],
                ),
            )

            # Store merges as bytes, because tests expect (bytes, bytes)
            merges.append(
                (
                    vocab[most_freq_pair[0]],
                    vocab[most_freq_pair[1]],
                )
            )

            new_id = len(vocab)
            vocab[new_id] = vocab[most_freq_pair[0]] + vocab[most_freq_pair[1]]

            # Important: copy because pair_to_ids will be mutated
            affected_ids = list(pair_to_ids[most_freq_pair])

            for idx in affected_ids:
                old_seq = id_to_seq[idx]
                count = id_to_count[idx]

                # Lazy stale check: maybe this pair no longer exists in this sequence
                if most_freq_pair not in zip(old_seq, old_seq[1:]):
                    pair_to_ids[most_freq_pair].discard(idx)
                    continue

                new_seq = apply_merge_to_seq(old_seq, most_freq_pair, new_id)

                if new_seq == old_seq:
                    continue

                old_local_pair_counts = get_pair_counts_in_seq(old_seq)
                new_local_pair_counts = get_pair_counts_in_seq(new_seq)

                # Remove old pair contributions
                for pair, local_count in old_local_pair_counts.items():
                    pair_counts[pair] -= local_count * count

                    if pair_counts[pair] <= 0:
                        del pair_counts[pair]

                    pair_to_ids[pair].discard(idx)
                    if not pair_to_ids[pair]:
                        del pair_to_ids[pair]

                # Add new pair contributions
                for pair, local_count in new_local_pair_counts.items():
                    pair_counts[pair] += local_count * count
                    pair_to_ids[pair].add(idx)

                id_to_seq[idx] = new_seq

            pbar.update(1)

    return vocab, merges


def save_vocab(vocab, path):
    serializable_vocab = {
        str(token_id): list(token_bytes) for token_id, token_bytes in vocab.items()
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(serializable_vocab, f, indent=2)

def save_merges(merges, path):
    serializable_merges = [
        [list(left), list(right)]
        for left, right in merges
    ]

    with open(path, "w", encoding="utf-8") as f:
        json.dump(serializable_merges, f, indent=2)


def train_bpe(input_path, vocab_size, special_tokens=[]):
    vocab = init_vocab(special_tokens)
    freq_dict = pretokenize(input_path, special_tokens)
    return merge(freq_dict, vocab, vocab_size)


if __name__ == "__main__":
    input_path = "data/TinyStoriesV2-GPT4-train.txt"
    vocab_path = "data/TinyStories_vocab.json"
    merges_path = "data/TinyStories_merges.json"
    vocab_size = 10000
    special_tokens = ["<|endoftext|>"]
    vocab, merges = train_bpe(input_path, vocab_size, special_tokens)
    save_vocab(vocab, vocab_path)
    save_merges(merges, merges_path)