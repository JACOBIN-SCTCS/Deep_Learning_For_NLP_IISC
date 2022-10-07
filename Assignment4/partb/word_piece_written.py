from collections import defaultdict
import pickle
import string
class WordPieceTokenizer:

    def __init__(self,dataset):
        self.dataset = dataset
        self.word_freqs = defaultdict(int)
        self.vocab = list(string.ascii_lowercase)
        self.vocab.append('<unk>')
        self.vocab_size = 32768
        self.splits = {}
        pass


    def init_tokenizer(self):
        for text in self.dataset:
            words = text.split(' ')
            for word in words:
                self.word_freqs[word] += 1

        for word in self.word_freqs.keys():
            if word[0] not in self.vocab:
                self.vocab.append(word[0])
            for letter in word[1:]:
                if f"##{letter}" not in self.vocab:
                   self.vocab.append(f"##{letter}")

        self.vocab.sort()

        self.splits = {
                word: [c if i == 0 else f"##{c}" for i, c in enumerate(word)]
                for word in self.word_freqs.keys()
        }

    def compute_pair_scores(self):
        letter_freqs = defaultdict(int)
        pair_freqs = defaultdict(int)
        
        for word, freq in self.word_freqs.items():
            split = self.splits[word]
            if len(split) == 1:
                letter_freqs[split[0]] += freq
                continue
            for i in range(len(split) - 1):
                pair = (split[i], split[i + 1])
                letter_freqs[split[i]] += freq
                pair_freqs[pair] += freq
            letter_freqs[split[-1]] += freq

        scores = {
            pair:    freq / (letter_freqs[pair[0]] * letter_freqs[pair[1]])
            for pair, freq in pair_freqs.items()
        }
        return scores

    def compute_best_pair(self):
        pair_scores = self.compute_pair_scores()
        best_pair = ""
        max_score = None
        for pair, score in pair_scores.items():
            if max_score is None or max_score < score:
                best_pair = pair
                max_score = score

        return (best_pair,max_score)

    def merge_pair(self,a, b):
        for word in self.word_freqs:
            split = self.splits[word]
            if len(split) == 1:
                continue
            i = 0
            while i < len(split) - 1:
                if split[i] == a and split[i + 1] == b:
                    merge = a + b[2:] if b.startswith("##") else a + b
                    split = split[:i] + [merge] + split[i + 2 :]
                else:
                    i += 1
            self.splits[word] = split

    def run(self):
        while len(self.vocab) < self.vocab_size:
            best_pair,max_score = self.compute_best_pair()
        
            splits =self. merge_pair(*best_pair)
            new_token = (
                best_pair[0] + best_pair[1][2:] if best_pair[1].startswith("##") else best_pair[0] + best_pair[1]
            )
            self.vocab.append(new_token)

    def tokenize(self,sentence):
        tokens = []
        words = sentence.split(' ')
        for word in words:
            if word == '.':
                tokens.append('.')
            if word == '':
                continue

            current_word = word
            while len(current_word) > 0:
                i = len(current_word)

                while i > 0 and current_word[:i] not in self.vocab:
                    i -= 1
                if i == 0:
                    tokens.append('<unk>')
                    break
                tokens.append(current_word[:i])
                current_word = current_word[i:]
                if len(current_word) > 0:
                    current_word = f"##{current_word}"

        return tokens

    def save_vocab():

        vocab_file = open('vocab_pickled', 'wb')
        pickle.dumps(vocab_file)


    
'''vocab = ['jacob','james','amy','thankyou','jack','kale','ascka']
word_piece = WordPieceTokenizer(vocab)
word_piece.init_tokenizer()
word_piece.run()


tokens = word_piece.tokenize('jacob james k is a bastard . I dont know what he is upto')'''

def process_text(text):
    text = re.sub(r'<br /><br />',".",text)
    text = BeautifulSoup(text,'lxml').get_text().strip()
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = ' '.join(re.findall(r"[\w']+|[.,!;/\"]", text))
    
    new_text = []
    for word in text.split():
        if word == '':
            continue
        new_text.append(word)
    
    text = ' '.join(new_text)
    return text

corpus = []
with open("./Train dataset.csv",encoding='utf-8') as csvfile:
    csvFile = csv.reader(csvfile)
    next(csvFile)

    for line in csvFile:
        processed_text = process_text(line[0])
        corpus.append(processed_text)
        

word_piece_tokenizer = WordPieceTokenizer(corpus)
word_piece_tokenizer.init_tokenizer()
word_piece_tokenizer.run()
word_piece_tokenizer.save_vocab()