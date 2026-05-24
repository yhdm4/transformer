from collections import Counter


class CharTokenizer:
    def __init__(self, texts, min_freq=1):
        self.pad_token = "[PAD]"
        self.unk_token = "[UNK]"

        counter = Counter()
        for text in texts:
            counter.update(str(text))

        vocab = [self.pad_token, self.unk_token]
        vocab.extend(ch for ch, count in counter.items() if count >= min_freq)

        self.stoi = {token: idx for idx, token in enumerate(vocab)}
        self.itos = {idx: token for token, idx in self.stoi.items()}

    @property
    def vocab_size(self):
        return len(self.stoi)

    @property
    def pad_id(self):
        return self.stoi[self.pad_token]

    @property
    def unk_id(self):
        return self.stoi[self.unk_token]

    def encode(self, text, max_len):
        ids = [self.stoi.get(ch, self.unk_id) for ch in str(text)]
        ids = ids[:max_len]
        ids.extend([self.pad_id] * (max_len - len(ids)))
        return ids
