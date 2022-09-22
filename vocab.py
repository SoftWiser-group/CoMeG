from collections import Counter
from itertools import chain
import json
import logging

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class VocabEntry:
    def __init__(self):
        self.word2id = dict()
        self.unk_id = 3
        self.word2id['<pad>'] = 0
        self.word2id['<s>'] = 1
        self.word2id['</s>'] = 2
        self.word2id['<unk>'] = 3

        self.id2word = {v: k for k, v in self.word2id.items()}

    def __getitem__(self, word):
        return self.word2id.get(word, self.unk_id)

    def is_unk(self, word):
        return word not in self.word2id

    def __contains__(self, word):
        return word in self.word2id

    def __setitem__(self, key, value):
        raise ValueError('vocabulary is readonly')

    def __len__(self):
        return len(self.word2id)

    def __repr__(self):
        return 'Vocabulary[size=%d]' % len(self)

    def id2word(self, wid):
        return self.id2word[wid]

    def add(self, word):
        if word not in self:
            wid = self.word2id[word] = len(self)
            self.id2word[wid] = word
            return wid
        else:
            return self[word]

    @staticmethod
    def from_corpus(corpus, size, freq_cutoff=5):
        """
        corpus: List[List[str]]
        """
        vocab_entry = VocabEntry()
        word_freq = Counter(chain(*corpus))
        freq_words = [w for w in word_freq if word_freq[w] >= freq_cutoff]
        logger.info(f'number of word types: {len(word_freq)}, number of word types w/ frequency >= {freq_cutoff}: {len(freq_words)}')
        top_k_words = word_freq.most_common(size)
        logger.info(f'top 10 words: {top_k_words[:10]}')

        for word, _ in top_k_words:
            if len(vocab_entry) < size:
                if word_freq[word] >= freq_cutoff:
                    vocab_entry.add(word)

        # store the work frequency table
        setattr(vocab_entry, 'word_freq', word_freq)
        return vocab_entry

    def save(self, path):
        params = dict(unk_id=self.unk_id,
                      word2id=self.word2id,
                      word_freq=self.word_freq)
        with open(path, 'w') as f:
            json.dump(params, f, indent=2)

    @staticmethod
    def load(path):
        entry = VocabEntry()
        with open(path) as f:
            params = json.load(f)
        setattr(entry, 'unk_id', params['unk_id'])
        setattr(entry, 'word2id', params['word2id'])
        setattr(entry, 'word_freq', params['word_freq'])
        setattr(entry, 'id2word', {v: k for k, v in params['word2id'].items()})
        return entry

    def decode(self, ids, example, args):
        tokens = []
        tokens1 = [''.join(s) for s in example[5]][:args.max_src]
        tokens2 = [''.join(s) for s in example[8]][:args.max_src]
        for i in ids:
            if i < len(self.word2id):
                tokens.append(self.id2word[i])
            else:
                i -= len(self.word2id)
                if i < len(tokens1):
                    tokens.append(tokens1[i].replace('\n', ''))
                else:
                    i -= args.max_src
                    if 0 <= i < len(tokens2):
                        tokens.append(tokens2[i].replace('\n', ''))
                    else:
                        tokens.append('<unk>')
        return tokens


def read_diff_from_jsonl(filename):
    data = []
    with open(filename) as f:
        for line in f:
            d = json.loads(line)
            data.append(d['diff'])
    return data


if __name__ == '__main__':
    # from prepare_data import load_data
    # data = load_data()
    # sub_corpus, node_corpus, tgt_corpus = [], [], []
    # for i in data:
    #     tgt_corpus.append(i[3])
    #     subs = set()
    #     for s in i[5] + i[8]:
    #         subs.update(s)
    #     sub_corpus.append(list(subs))
    #     nodes = set()
    #     for s in i[6] + i[9]:
    #         nodes.update(s)
    #     node_corpus.append(list(nodes))
    # sub_vocab = VocabEntry.from_corpus(sub_corpus, 30000, 2)
    # node_vocab = VocabEntry.from_corpus(node_corpus, 1500, 1)
    # tgt_vocab = VocabEntry.from_corpus(tgt_corpus, 3000, 2)
    pass