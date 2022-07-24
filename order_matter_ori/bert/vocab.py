import pickle
from pip import main
import tqdm
from collections import Counter


class TorchVocab(object):
    """
    一个领域的词汇表对象
    Attributes:
        freqs: collections.Counter对象，包含了被用于建立Vocab的全部（未进行min_freq和max_size过滤）tokens的出现频率
        itos: list,符合min_freq的max_size个词的词表
        stoi: collections.defaultdict对象，包含tokens字符串到index的映射，itos中词到索引的映射
        
    """

    def __init__(self, counter, max_size=None, min_freq=1, specials=['<pad>', '<oov>'],
                 vectors=None, unk_init=None, vectors_cache=None):
        """Create a Vocab object from a collections.Counter.
        Arguments:
            counter: collections.Counter，各个token出现的次数
            max_size: 词汇表的最大数量，默认为None，表示不限制词汇表数量
            min_freq: 词汇表中收录的词要出现的最小频率
            specials: 特殊词列表，列表中的词都会同一转化为<unk>，默认列表为['pad']
            vectors: One of either the available pretrained vectors
            unk_init (callback): 默认词汇表外的向量词向量为0向量，可以输入一个函数来进行词汇表外的词转换为指定向量
            vectors_cache: directory for cached vectors. Default: '.vector_cache'
        """
        self.freqs = counter
        counter = counter.copy()
        min_freq = max(min_freq, 1)

        self.itos = list(specials)
        
        # 将特殊词表中的词在词表中删除
        for tok in specials:
            del counter[tok]

        max_size = None if max_size is None else max_size + len(self.itos)

        # 按照频率、首字母对词表counter进行排序
        words_and_frequencies = sorted(counter.items(), key=lambda tup: tup[0])
        words_and_frequencies.sort(key=lambda tup: tup[1], reverse=True)

        # 将符合min_freq的max_size个词加入itos
        for word, freq in words_and_frequencies:
            if freq < min_freq or len(self.itos) == max_size:
                break
            self.itos.append(word)

        # 建立itos中的词和索引的映射关系字典stoi
        self.stoi = {tok: i for i, tok in enumerate(self.itos)}

        self.vectors = None
        if vectors is not None:
            self.load_vectors(vectors, unk_init=unk_init, cache=vectors_cache)
        else:
            assert unk_init is None and vectors_cache is None

    def __eq__(self, other):
        if self.freqs != other.freqs:
            return False
        if self.stoi != other.stoi:
            return False
        if self.itos != other.itos:
            return False
        if self.vectors != other.vectors:
            return False
        return True

    def __len__(self):
        return len(self.itos)




class Vocab(TorchVocab):
    """
        主要是定义了一些特殊的toeken及其对应的索引
    """
    def __init__(self, counter, max_size=None, min_freq=1):
        self.pad_index = 0  # 填充标记索引
        self.unk_index = 1  # 
        self.eos_index = 2  # 起始标记索引
        self.sos_index = 3  # 结束标记索引
        self.mask_index = 4 # mask标记索引
        super().__init__(counter, specials=["<pad>", "<unk>", "<eos>", "<sos>", "<mask>"],
                         max_size=max_size, min_freq=min_freq)

    def to_seq(self, sentece, seq_len, with_eos=False, with_sos=False) -> list:
        pass

    def from_seq(self, seq, join=False, with_pad=False):
        pass

    @staticmethod
    def load_vocab(vocab_path: str) -> 'Vocab':
        with open(vocab_path, "rb") as f:
            return pickle.load(f)

    def save_vocab(self, vocab_path):
        with open(vocab_path, "wb") as f:
            pickle.dump(self, f)


# Building Vocab with text files
class WordVocab(Vocab):
    def __init__(self, texts, max_size=None, min_freq=1):
        print("Building Vocab")
        counter = Counter()
        for line in tqdm.tqdm(texts):
            if isinstance(line, list):
                words = line
            else:
                words = line.replace("\n", " ").replace("\t", " ").split()  # 将一个指令对中的指令按照token进行切分
            for word in words:
                counter[word] += 1
        super().__init__(counter, max_size=max_size, min_freq=min_freq)

    def to_seq(self, sentence, seq_len=None, with_eos=False, with_sos=False, with_len=False):
        """
        将指令中的token转换为index形式
        """
        if isinstance(sentence, str):
            sentence = sentence.split()

        seq = [self.stoi.get(word, self.unk_index) for word in sentence]

        if with_eos:
            seq += [self.eos_index]  # this would be index 1
        if with_sos:
            seq = [self.sos_index] + seq

        origin_seq_len = len(seq)

        if seq_len is None:
            pass
        elif len(seq) <= seq_len:
            seq += [self.pad_index for _ in range(seq_len - len(seq))]
        else:
            seq = seq[:seq_len]

        return (seq, origin_seq_len) if with_len else seq

    def from_seq(self, seq, join=False, with_pad=False):
        """
            将index组成的指令转化回tokens形式
        """
        words = [self.itos[idx]
                 if idx < len(self.itos)
                 else "<%d>" % idx
                 for idx in seq
                 if not with_pad or idx != self.pad_index]

        return " ".join(words) if join else words

    @staticmethod
    def load_vocab(vocab_path: str) -> 'WordVocab':
        with open(vocab_path, "rb") as f:
            return pickle.load(f)

if __name__=="__main__":
    train_cfg_dataset = "data/order_matter/cfg_train.txt"
    vocab_path = "data/order_matter/vocab"

    with open(train_cfg_dataset, "r", encoding="utf-8") as f1:
        vocab = WordVocab(f1, max_size=13000, min_freq=1)

    print("VOCAB SIZE:", len(vocab))
    vocab.save_vocab(vocab_path)

    vocab = WordVocab.load_vocab(vocab_path)
