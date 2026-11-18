

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


