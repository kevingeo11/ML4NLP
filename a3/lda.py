import os
import re

import nltk
import numpy as np
from collections import Counter
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


def get_nltk_data():
    nltk_data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nltk_data")
    if nltk_data_dir not in nltk.data.path:
        nltk.data.path.insert(0, nltk_data_dir)

    resources = {
        "stopwords": "corpora/stopwords",
        "punkt": "tokenizers/punkt",
        "punkt_tab": "tokenizers/punkt_tab",
    }

    for name, resource_path in resources.items():
        try:
            nltk.data.find(resource_path, paths=[nltk_data_dir])
        except LookupError:
            nltk.download(name, download_dir=nltk_data_dir)



class Preprocess:

    def __init__(self, document_file, min_frequency=10): # remove documents when not needed

        self.document_file = document_file
        self.min_frequency = min_frequency

        self.documents = self._load_documents()
        self.tokenized_documents = self._tokenize_documents()
        self.vocabulary = self._create_vocabulary()
        self.word_to_index = self._map_vocabulary_to_indices()
        self.filtered_documents = self._filter_tokenized_documents()
        self.documents_indices = self._convert_documents_to_indices()

    # load documents from the file and extract text within <TEXT> tags
    def _load_documents(self):
        with open(self.document_file, 'r') as f:
            content = f.read()
        documents = [t.strip() for t in re.findall(r"<TEXT>(.*?)</TEXT>", content, re.DOTALL)]
        print(f"Documents loaded:       {len(documents):,}")

        return documents

    # go through each document and tokenize it (convert to lowercase, remove punctuation and hyphened words, stopwords)
    def _tokenize_documents(self):
        tokenized_documents = []
        stop_words = set(stopwords.words('english'))
        stop_words |= {
            "said", "say", "told", "news",
            "would", "could", "may",
            "one", "two", "three", "first", "last",
            "year", "years", "day", "week", "today", "time", "since", "new",
            "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
            "also", "many", "made", "people",
        }
        for doc in self.documents:
            text = doc.lower()
            tokens = [
                token for token in word_tokenize(text, language="english")
                if token.isalpha()            # words only
                and len(token) > 2
                and token not in stop_words   # remove stopwords
            ]
            tokenized_documents.append(tokens)

        n_tokens = sum(len(t) for t in tokenized_documents)
        n_unique = len({t for tokens in tokenized_documents for t in tokens})
        print(f"Tokens after cleaning:  {n_tokens:,}  ({n_unique:,} unique)")

        return tokenized_documents

    # create a vocabulary of all words in the tokenized documents and filter based on minimum frequency
    def _create_vocabulary(self):
        word_count = Counter(t for tokens in self.tokenized_documents for t in tokens)

        filtered_word_count = {word: count for word, count in word_count.items() if count >= self.min_frequency}
        vocabulary = sorted(filtered_word_count)

        print(f"Vocabulary size:        {len(word_count):,} -> {len(vocabulary):,} (min frequency {self.min_frequency})")
        print("Top 20 words:", ", ".join(w for w, _ in word_count.most_common(20)))

        return vocabulary
        
    # map vocabulary to indices
    def _map_vocabulary_to_indices(self):
        return {word: idx for idx, word in enumerate(self.vocabulary)}

    def _filter_tokenized_documents(self):
        return [
            [token for token in tokens if token in self.word_to_index]
            for tokens in self.tokenized_documents
        ]

    # convert documents to indices based on the filtered vocabulary
    def _convert_documents_to_indices(self):
        return [
            [self.word_to_index[token] for token in tokens]   # already filtered, so no check needed
            for tokens in self.filtered_documents
        ]



class LDA:

    def __init__(self, documents, V, alpha, beta, k, n_iter = 100, seed=0):

        self.documents = documents
        self.D = len(documents)
        self.V = V

        self.alpha = alpha
        self.beta = beta
        self.k = k
        self.n_iter = n_iter

        # randomly initialize the topic assignments for each word in each document
        self.rng = np.random.default_rng(seed)
        self.z = [self.rng.choice(self.k, size=len(doc)) for doc in self.documents]
    
        self.n_dk = np.zeros((self.D, self.k), dtype=int)
        self.m_kv = np.zeros((self.k, self.V), dtype=int)
        self.m_k = np.zeros(self.k, dtype=int)

        self._init_counts()

    def _init_counts(self):
        # go through every token once and count its random starting label
        for d, doc in enumerate(self.documents):
            for j, v in enumerate(doc):
                self._add(d, v, self.z[d][j])

    def _add(self, d, v, k):
        # put one token (word v in doc d, topic k) into the counts
        self.n_dk[d, k] += 1
        self.m_kv[k, v] += 1
        self.m_k[k] += 1

    def _remove(self, d, v, k):
        self.n_dk[d, k] -= 1
        self.m_kv[k, v] -= 1
        self.m_k[k] -= 1

    def p_k(self, d, v):
        # probability of each topic for word v in doc d (current token already removed)
        q_k = (self.alpha + self.n_dk[d]) * (self.beta + self.m_kv[:, v]) / (self.beta * self.V + self.m_k)
        return q_k / q_k.sum()

    def collapsed_gibbs(self):
        for it in range(self.n_iter):              # one iteration = one pass over every token
            changed = 0
            for d, doc in enumerate(self.documents):
                for j, v in enumerate(doc):
                    old_k = self.z[d][j]
                    self._remove(d, v, old_k)                        # 1. take the label off
                    new_k = self.rng.choice(self.k, p=self.p_k(d, v))  # 2-3. score topics and sample
                    self._add(d, v, new_k)                           # 4. put it back
                    self.z[d][j] = new_k
                    changed += int(new_k != old_k)

            if (it + 1) % 10 == 0:
                print(f"iteration {it + 1}/{self.n_iter}: {changed:,} labels changed")

        return self.z


    # def choose_dj(self): 
    #     # choose a word from the corpus within a document d_j, randomly 
    #     doc_idx = np.random.randint(len(self.documents))
    #     word_idx = np.random.randint(len(self.d[doc_idx]))
    #     return doc_idx, word_idx

    # def n_dk(self, k, d, d_j):
    #     # number of times topic k appears in document d, not counting the current word d_j
    #     occurrences = np.sum(self.z[d] == k)
    #     if d_j[0] == d and self.z[d_j[0]][d_j[1]] == k:
    #         occurrences -= 1
    #     return occurrences

    
    # def m_kv(self, k, v, d_j):
    #     # number of times word v is assigned to topic k, not counting the current word d_j
    #     occurrences = 0
    #     for d in range(len(self.d)):
    #         for j, word in enumerate(self.d[d]):
    #             if word == v and self.z[d][j] == k:
    #                 occurrences += 1
    #     if self.d[d_j[0]][d_j[1]] == v and self.z[d_j[0]][d_j[1]] == k:
    #         occurrences -= 1
    #     return occurrences
    
    # def m_k(self, k, d_j):
    #     # number of times topic k appears in the corpus, not counting the current word d_j
    #     occurrences = sum(np.sum(self.z[d] == k) for d in range(len(self.d)))
    #     if self.z[d_j[0]][d_j[1]] == k:
    #         occurrences -= 1
    #     return occurrences

    # def p_k(self, d_j):
        
    #     d = d_j[0]
    #     j = d_j[1]
    #     v = self.d[d][j]

    #     q_k = []
    #     for k in range(self.k):
    #         n_dk = self.n_dk(k, d, d_j)
    #         m_kv = self.m_kv(k, v, d_j)
    #         m_k = self.m_k(k, d_j)
    #         q_k.append((self.alpha + n_dk) * (self.beta + m_kv) / (self.beta * len(self.vocab) + m_k))
       
    #     return q_k / np.sum(q_k)

    # def collapsed_gibbs(self):
    #     for i in range(self.n_iter):

    #         d_j = self.choose_dj()
    #         p__k = self.p_k(d_j)
    #         self.z[d_j[0], d_j[1]] = np.random.choice(self.k, p = p__k)
        
    #     return self.z




if __name__ == "__main__":

    get_nltk_data()

    document_file = "data/ap.txt"

    p = Preprocess(document_file=document_file, min_frequency=10)

    # # save vocabulary
    # with open("vocabulary.txt", "w", encoding="utf-8") as f:
    #     f.write("\n".join(p.vocabulary) + "\n")

    # # just for sanity check
    # with open("p_sanity_check.txt", "w") as f:
    #     f.write("tokenized_documents: " + str(p.tokenized_documents) + "\n")
    #     f.write("word_count: " + str(p.word_count) + "\n")
    #     f.write("vocabulary: " + str(p.vocabulary) + "\n")
    #     f.write("vocabulary_to_index: " + str(p.word_to_index) + "\n")
    #     f.write("filtered_documents: " + str(p.filtered_documents) + "\n")
    #     f.write("documents_indices: " + str(p.documents_indices) + "\n")

    lda = LDA(
        documents=p.documents_indices,
        V = len(p.vocabulary),
        alpha = 0.1,
        beta = 0.1,
        k = 10
    )
