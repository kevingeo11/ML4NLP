import os
import re
import argparse
import sys

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
        print("Top 20 words in vocabulary:", ", ".join(w for w, _ in word_count.most_common(20)))

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

    def __init__(self, documents, V, alpha, beta, k, n_iter = 100, seed=0, M=20):

        self.documents = documents
        self.D = len(documents)
        self.V = V

        self.alpha = alpha
        self.beta = beta
        self.k = k
        self.n_iter = n_iter
        self.M = M

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


    def umass_coherence(self):

        # create a document-word occurrence matrix Dv where Dv[d, v] = 1 if word v appears in document d
        Dv = np.zeros((self.D, self.V))
        for d, doc in enumerate(self.documents):
            Dv[d, list(set(doc))] = 1

        scores = []
        for topic in range(self.k):
            topM = np.argsort(self.m_kv[topic])[::-1][:self.M] # get the top M words for this topic

            Dv_sub = Dv[:, topM] # subset of Dv containing only the top M words for this topic
            Dvv_sub = Dv_sub.T @ Dv_sub # co-occurrence matrix for the top M words in this topic

            score = 0.0
            for m in range(1, self.M):
                for l in range(m):
                    score += np.log((Dvv_sub[m, l] + 1) / Dvv_sub[l, l]) # UMass coherence for word pair (l, m)
            scores.append(score)

        return np.array(scores)




if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--corpora", default="data/ap.txt")
    parser.add_argument("--k", default=10)
    parser.add_argument("--alpha", default=0.1)
    parser.add_argument("--beta", default=0.1)
    parser.add_argument("--n_iter", default=100)
    parser.add_argument("--output", default="output.txt")
    args = parser.parse_args()
    
    get_nltk_data()

    document_file = args.corpora

    p = Preprocess(document_file=document_file, min_frequency=5)

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
        alpha = args.alpha,
        beta = args.beta,
        k = args.k, 
        M=20, 
        n_iter = args.n_iter
    )

    coherence_scores = lda.umass_coherence()

    # Print to terminal and save to output file
    class Tee:
        def __init__(self, *streams):
            self.streams = streams

        def write(self, text):
            for stream in self.streams:
                stream.write(text)

        def flush(self):
            for stream in self.streams:
                stream.flush()

    # Generate outputs
    with open(args.output, "w", encoding="utf-8") as f:
        original_stdout = sys.stdout
        sys.stdout = Tee(original_stdout, f)
 
        try:
            print("Names: Kevin George, Sofia Larsson, Thémis Halka")
            print("Output")
            print("-" * 20)
            print(" ")
            print("Corpora: AP corpus")
            print("Model parameters: ")
            print("     Topics K:", args.k)
            print("     Alpha:", args.alpha)
            print("     Beta:", args.beta)
            print("     Number of iterations through corpora:", args.n_iter)

            for topic in range(lda.k):
                print(f"\nTop words for topic {topic} with coherence score {coherence_scores[topic]:.4f}:")
                for i in np.argsort(lda.m_kv[topic])[::-1][:10]:
                    print('\t' + p.vocabulary[i])
 
        finally:
            sys.stdout = original_stdout
 
    print(f"\nEvaluation results saved to {args.output}")