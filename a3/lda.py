

class LDA():

    def __init__(self, alpha, beta, k, n_iter = 100):

        self.d = []

        self.alpha = alpha
        self.beta = beta
        self.k = k
        self.n_iter = n_iter

        # randomly initialize the topic assignments for each word in each document
        self.theta = np.random.dirichlet([self.alpha] * self.k, size = len(self.d))
        self.phi = np.random.dirichlet([self.beta] * self.k, size = len(self.d))
    
    
    def choose_dj(): 
        # choose a word from the corpus within a document d_j, randomly 
        doc = np.random.choice(self.d)
        word = np.random.choice(doc)
        return word



    def n_dk():
    
    def m_

    def collapsed_gibbs():
        for i in range(self.n_iter):

            d_j = self.choose_dj()
            