import re

# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize


class Preprocess:

    def __init__(self, document_file, min_frequency=2): # remove documents when not needed
        self.document_file = document_file
        self.min_frequency = min_frequency

        self.documents = self._load_documents()
        self.tokenized_documents = self._tokenize_documents()
        self.word_count, self.filtered_word_count, self.vocabulary = self._create_vocabulary()
        self.word_to_index = self._map_vocabulary_to_indices()
        self.filtered_documents = self._filter_tokenized_documents()
        self.documents_indices = self._convert_documents_to_indices()

    # load documents from the file and extract text within <TEXT> tags
    def _load_documents(self):
        with open(self.document_file, 'r') as f:
            content = f.read()
        return [t.strip() for t in re.findall(r"<TEXT>(.*?)</TEXT>", content, re.DOTALL)]

    # go through each document and tokenize it (convert to lowercase, remove punctuation and hyphened words, stopwords)
    def _tokenize_documents(self):
        tokenized_documents = []
        # stop_words = set(stopwords.words('english')) # NLTK
        for doc in self.documents:
            tokens = doc.lower().split()  # simple whitespace tokenizer and convert to lowercase
            # tokens = word_tokenize(doc.lower())  # NLTK convert to lowercase and word tokenize
            tokens = [token for token in tokens if token.isalnum()]  # remove punctuation and hyphened words
            # tokens = [token for token in tokens if token not in stop_words]  # NLTK remove stopwords
            tokenized_documents.append(tokens)
        return tokenized_documents

    # create a vocabulary of all words in the tokenized documents and filter based on minimum frequency
    def _create_vocabulary(self):
        word_count = {}
        for tokens in self.tokenized_documents:
            for token in tokens:
                if token not in word_count:
                    word_count[token] = 0
                word_count[token] += 1

        filtered_word_count = {word: count for word, count in word_count.items() if count >= self.min_frequency}
        vocabulary = list(filtered_word_count.keys())
        return word_count, filtered_word_count, vocabulary
        
    # map vocabulary to indices
    def _map_vocabulary_to_indices(self):
        return {word: idx for idx, word in enumerate(self.vocabulary)}

    # filter tokenized documents based on vocabulary
    def _filter_tokenized_documents(self):
        filtered_documents = []
        for tokens in self.tokenized_documents:
            filtered_tokens = [token for token in tokens if token in self.vocabulary]
            filtered_documents.append(filtered_tokens)
        return filtered_documents

    # convert documents to indices based on the filtered vocabulary
    def _convert_documents_to_indices(self):
        documents_indices = []
        for tokens in self.filtered_documents:
            indices = [self.word_to_index[token] for token in tokens if token in self.word_to_index]
            documents_indices.append(indices)
        return documents_indices



class LDA():

    def __init__(self, documents, alpha, beta, k, n_iter = 100):

        self.d = documents
        self.D = len(documents)

        self.vocab = set(word for doc in self.d for word in doc)
        self.V = len(self.vocab)

        self.alpha = alpha
        self.beta = beta
        self.k = k
        self.n_iter = n_iter

        # randomly initialize the topic assignments for each word in each document
        self.z = [np.random.choice(self.k, size=len(doc)) for doc in self.d]
    
        self.n_dk = np.zeros((self.D, self.k), dtype=int)
        self.m_kv = np.zeros((self.k, self.V), dtype=int)
        self.m_k = np.zeros(self.k, dtype=int)

    def choose_dj(self): 
        # choose a word from the corpus within a document d_j, randomly 
        doc_idx = np.random.randint(len(self.d))
        word_idx = np.random.randint(len(self.d[doc_idx]))
        return doc_idx, word_idx



    def n_dk(self, k, d, d_j):
        # number of times topic k appears in document d, not counting the current word d_j
        occurences = np.sum(self.z[d] == k)
        if d_j[0] == d and self.z[d_j[0]][d_j[1]] == k:
            occurrences -= 1
        return occurrences

    
    def m_kv(self, k, v, d_j):
        # number of times word v is assigned to topic k, not counting the current word d_j
        occurrences = 0
        for d in range(len(self.d)):
            for j, word in enumerate(self.d[d]):
                if word == v and self.z[d][j] == k:
                    occurrences += 1
        if self.d[d_j[0]][d_j[1]] == v and self.z[d_j[0]][d_j[1]] == k:
            occurrences -= 1
        return occurrences
    
    def m_k(self, k, d_j):
        # number of times topic k appears in the corpus, not counting the current word d_j
        occurrences = sum(np.sum(self.z[d] == k) for d in range(len(self.d)))
        if self.z[d_j[0]][d_j[1]] == k:
            occurrences -= 1
        return occurrences

    def p_k(self, d_j):
        
        d = d_j[0]
        j = d_j[1]
        v = self.d[d][j]

        q_k = []
        for k in range(self.k):
            n_dk = self.n_dk(k, d, d_j)
            m_kv = self.m_kv(k, v, d_j)
            m_k = self.m_k(k, d_j)
            q_k.append((self.alpha + n_dk) * (self.beta + m_kv) / (self.beta * len(self.vocab) + m_k))
       
        return q_k / np.sum(q_k)


    def collapsed_gibbs(self):
        for i in range(self.n_iter):

            d_j = self.choose_dj()
            p__k = self.p_k(d_j)
            self.z[d_j[0], d_j[1]] = np.random.choice(self.k, p = p__k)
        
        return self.z




if __name__ == "__main__":

    document_file = "test_ap.txt"

    p = Preprocess(document_file=document_file, min_frequency=2)

    # save vocabulary
    with open("vocabulary.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(p.vocabulary) + "\n")

    # just for sanity check
    with open("p_sanity_check.txt", "w") as f:
        f.write("tokenized_documents: " + str(p.tokenized_documents) + "\n")
        f.write("vocabulary: " + str(p.vocabulary) + "\n")
        f.write("vocabulary_to_index: " + str(p.word_to_index) + "\n")
        f.write("filtered_documents: " + str(p.filtered_documents) + "\n")
        f.write("documents_indices: " + str(p.documents_indices) + "\n")
