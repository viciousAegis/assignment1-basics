import regex as re  # type: ignore
from typing import Iterable, Iterator
import json
import numpy as np

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


class Tokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        self.vocab = vocab
        self.merges = merges
        if special_tokens == None:
            special_tokens = []
        self.special_tokens = special_tokens
        existing = set(self.vocab.values())
        next_id = len(vocab)
        for token in self.special_tokens:
            if token.encode("utf-8") in existing:
                continue
            # print(token.encode("utf-8"))
            self.vocab[next_id] = token.encode("utf-8")
            next_id += 1
        self.tok_to_id = {vocab[idx]: idx for idx in vocab}
        self.merge_ranks = {pair: rank for rank, pair in enumerate(self.merges)}
        if not self.special_tokens:
            self.buffer_length = 0
        else:
            self.buffer_length = max(len(tok) for tok in self.special_tokens) - 1

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=[]):

        def load_vocab(vocab_filepath) -> dict[int, bytes]:
            with open(vocab_filepath, "r", encoding="utf-8") as f:
                data = json.load(f)

                return {int(idx): bytes(data[idx]) for idx in data}

        def load_merges(merges_filepath) -> list[tuple[bytes, bytes]]:
            with open(merges_filepath, "r", encoding="utf-8") as f:
                data = json.load(f)

                return [(bytes(merge[0]), bytes(merge[1])) for merge in data]

        vocab = load_vocab(vocab_filepath)
        merges = load_merges(merges_filepath)

        return cls(vocab, merges, special_tokens)

    def pretokenize(self, text: str):
        if self.special_tokens:
            special_tokens_sorted = sorted(self.special_tokens, key=len, reverse=True)

            pattern = (
                "("
                + "|".join(re.escape(tok) for tok in special_tokens_sorted)
                + ")"
            )
            parts = re.split(pattern, text)
        else:
            parts = [text]

        tok_list = []
        for part in parts:
            if part in self.special_tokens:
                tok_list.append([part.encode("utf-8")])
                continue
            toks = re.findall(PAT, part)
            pre_token_ids = [list(tok.encode("utf-8")) for tok in toks]

            for pre_token_id in pre_token_ids:
                pre_tokens = [bytes([byte]) for byte in pre_token_id]
                tok_list.append(pre_tokens)

        return tok_list

    def find_merge(self, pre_token):
        if len(pre_token) == 1:
            return pre_token
        pairs = list(zip(pre_token, pre_token[1:]))

        candidates = [
            (idx, pair, self.merge_ranks[pair])
            for idx, pair in enumerate(pairs)
            if pair in self.merge_ranks
        ]
        
        if not candidates:
            return pre_token

        idx, pair, _ = min(candidates, key=lambda x: x[2])
        pair = pairs[idx]
        # construct new pretoken and recurse
        merge = self.vocab[self.tok_to_id[bytes(pair[0]) + bytes(pair[1])]]
        new_pre_token = pre_token[:idx] + [merge] + pre_token[idx + 2 :]
        return self.find_merge(new_pre_token)

    def encode(self, text) -> list[int]:
        tok_list = self.pretokenize(text)
        token_ids = []
        for pre_token in tok_list:
            merged_token = self.find_merge(pre_token)
            token_ids.extend(self.tok_to_id[tok] for tok in merged_token)
        return token_ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        buffer = ""
        for chunk in iterable:
            buffer += chunk
            # check for endoftexts
            special_tokens_sorted = sorted(self.special_tokens, key=len, reverse=True)

            pattern = (
                "("
                + "|".join(re.escape(tok) for tok in special_tokens_sorted)
                + ")"
            )
            
            matches = re.finditer(pattern, buffer)
            match = next(matches, None)

            if match is None: # no special strings found
                continue
            
            end = match.end()
            
            safe_prefix = buffer[:end]
            buffer = buffer[end:]
            for token_id in self.encode(safe_prefix):
                yield token_id
        
        yield from self.encode(buffer)

    def decode(self, ids: list[int]) -> str:
        concat = bytes()
        for idx in ids:
            seq: bytes = self.vocab[idx]
            concat+=seq
        return concat.decode(errors='replace')

import random
from textwrap import indent

def test_tinystories_random_roundtrip(
    tokenizer,
    path: str = "data/TinyStoriesV2-GPT4-valid.txt",
    N: int = 5,
    special_token: str = "<|endoftext|>",
    seed: int = 42,
):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    docs = [doc for doc in text.split(special_token) if doc.strip()]

    rng = random.Random(seed)
    sampled_docs = rng.sample(docs, k=min(N, len(docs)))

    for i, doc in enumerate(sampled_docs, start=1):
        original = doc + special_token

        encoded_ids = list(tokenizer.encode_iterable([original]))
        decoded = tokenizer.decode(encoded_ids)

        passed = original == decoded

        print("=" * 100)
        print(f"Random sample {i}")
        print(f"Roundtrip passed: {passed}")
        print(f"Original length: {len(original)} chars")
        print(f"Decoded length:  {len(decoded)} chars")
        print(f"Encoded tokens:  {len(encoded_ids)}")
        print("-" * 100)

        print("ORIGINAL:")
        print(indent(repr(original[:1000]), "  "))

        print("\nDECODED:")
        print(indent(repr(decoded[:1000]), "  "))

        if not passed:
            print("\nFIRST DIFFERENCE:")
            for j, (a, b) in enumerate(zip(original, decoded)):
                if a != b:
                    print(f"  index: {j}")
                    print(f"  original char: {repr(a)}")
                    print(f"  decoded char:  {repr(b)}")
                    print(f"  original context: {repr(original[max(0, j-50):j+50])}")
                    print(f"  decoded context:  {repr(decoded[max(0, j-50):j+50])}")
                    break

            if len(original) != len(decoded):
                print("  Strings also differ in length.")

        print()

if __name__ == "__main__":
    vocab_path = "data/TinyStories_vocab.json"
    merges_path = "data/TinyStories_merges.json"
    special_tokens = ["<|pad|>", "<|endoftext|>"]
    tok = Tokenizer.from_files(
        vocab_filepath=vocab_path,
        merges_filepath=merges_path,
        special_tokens=special_tokens,
    )
    
    with open("data/TinyStoriesV2-GPT4-valid.txt") as f:
        text = f.read()
    
    text = "s"
    enc = tok.encode(text)
    print(enc)
    dec1 = tok.decode(enc)
    print(dec1)
    
    chunks = [
        "Hello wor",
        "ld<|endo",
        "ftext|>This is a te",
        "st<|endoftext|>",
        "done",
    ]
    
    enc = tok.encode_iterable(chunks)
    
    for t in enc:
        print(t)
        
    test_tinystories_random_roundtrip(
        tok,
        path="data/TinyStoriesV2-GPT4-valid.txt",
        N=10,
        seed=123,
    )