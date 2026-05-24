class CharTokenizer:
    def __init__(self, texts, min_freq=1):
        self.pad_token = "[PAD]"
        self.unk_token = "[UNK]"

        freq = {}

        for text in texts:
            for ch in text:
                freq[ch] = freq.get(ch, 0) + 1

        vocab = [self.pad_token, self.unk_token]

        for ch, count in freq.items():
            if count >= min_freq:
                vocab.append(ch)

        self.stoi = {ch: i for i, ch in enumerate(vocab)}
        self.itos = {i: ch for ch, i in self.stoi.items()}

    @property
    def vocab_size(self):
        return len(self.stoi)

    @property
    def pad_id(self):
        return self.stoi[self.pad_token]

    def encode(self, text, max_len):
        ids = [self.stoi.get(ch, self.stoi[self.unk_token]) for ch in text]

        ids = ids[:max_len]

        if len(ids) < max_len:
            ids += [self.pad_id] * (max_len - len(ids))

        return ids