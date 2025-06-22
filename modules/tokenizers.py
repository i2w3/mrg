import re
import json
from collections import Counter
from pathlib import Path

class Tokenizer:
    def __init__(self, args):
        self.ann_path = args.ann_path
        self.threshold = args.threshold

        # Special tokens
        self.pad_token = "<pad>"
        self.bos_token = "<bos>"
        self.eos_token = "<eos>"
        self.unk_token = "<unk>"

        self.special_tokens = [self.pad_token, self.bos_token, self.eos_token, self.unk_token]
        self.pad_idx = 0
        self.bos_idx = 1
        self.eos_idx = 2
        self.unk_idx = 3

        self.clean_report = self.clean_report_fair

        # Load annotation
        self.ann = json.load(open(self.ann_path, 'r'))

        # Create or load vocabulary
        self.token2idx, self.idx2token = self.create_vocabulary()

    def create_vocabulary(self):
        total_tokens = []
        for split in ['train', 'val', 'test']:
            for report in self.ann[split]:
                tokens = self.clean_report(report['En_Report']).split()
                total_tokens.extend(tokens)

        counter = Counter(total_tokens)
        vocab = [token for token, freq in counter.items() if freq >= self.threshold]
        vocab = sorted(vocab)

        # Add special tokens at the beginning
        vocab = self.special_tokens + vocab

        token2idx = {token: idx for idx, token in enumerate(vocab)}
        idx2token = {idx: token for token, idx in token2idx.items()}
        return token2idx, idx2token

    def clean_report_fair(self, report):
        report_cleaner = lambda t: t.replace('\n', ' ').replace('__', '_').replace('..', '.').replace('..', '.') \
            .replace('  ', ' ').strip().lower().split('. ')

        sent_cleaner = lambda t: re.sub(r'[.,?;*!%^&_+():-\[\]{}]', '', 
                                       t.replace('"', '').replace('/', '').replace('\\', '')
                                        .replace("'", '').strip().lower())

        tokens = [sent_cleaner(sent) for sent in report_cleaner(report) if sent_cleaner(sent) != '']
        report = ' . '.join(tokens) + ' .'
        return report

    def encode(self, report):
        tokens = self.clean_report(report).split()
        ids = [self.get_id_by_token(token) for token in tokens]
        return [self.bos_idx] + ids + [self.eos_idx]

    def decode(self, ids):
        tokens = []
        for idx in ids:
            if idx == self.eos_idx:
                break
            if idx != self.bos_idx and idx != self.pad_idx:
                tokens.append(self.idx2token.get(idx, self.unk_token))
        return ' '.join(tokens)

    def decode_batch(self, ids_batch):
        return [self.decode(ids) for ids in ids_batch]

    def get_id_by_token(self, token):
        return self.token2idx.get(token, self.unk_idx)

    def get_token_by_id(self, idx):
        return self.idx2token.get(idx, self.unk_token)

    def get_vocab_size(self):
        return len(self.token2idx)

    def save_vocab(self, save_path):
        with open(save_path, 'w') as f:
            json.dump({'token2idx': self.token2idx, 'idx2token': self.idx2token}, f)

    def load_vocab(self, load_path):
        with open(load_path, 'r') as f:
            vocab = json.load(f)
        self.token2idx = vocab['token2idx']
        self.idx2token = {int(k): v for k, v in vocab['idx2token'].items()}

    def __call__(self, report):
        return self.encode(report)

if __name__ == '__main__':
    class Args:
        ann_path = '/home/moment/PostGraduationProject/dataset/100/annotation.json'
        threshold = 3

    args = Args()
    tokenizer = Tokenizer(args)
    print(f"Vocabulary size: {tokenizer.get_vocab_size()}")
