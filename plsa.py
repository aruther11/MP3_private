import numpy as np
import math
from functools import reduce

def normalize(input_matrix):
    """
    Normalizes the rows of a 2d input_matrix so they sum to 1
    """

    row_sums = input_matrix.sum(axis=1)
    try:
        assert (np.count_nonzero(row_sums)==np.shape(row_sums)[0]) # no row should sum to zero
    except Exception:
        raise Exception("Error while normalizing. Row(s) sum to zero")
    new_matrix = input_matrix / row_sums[:, np.newaxis]
    return new_matrix

       
class Corpus(object):

    """
    A collection of documents.
    """

    def __init__(self, documents_path):
        """
        Initialize empty document list.
        """
        self.documents = []
        self.vocabulary = []
        self.likelihoods = []
        self.documents_path = documents_path
        self.term_doc_matrix = None 
        self.document_topic_prob = None  # P(z | d)
        self.topic_word_prob = None  # P(w | z)
        self.topic_prob = None  # P(z | d, w)
        self.number_of_topics = 0
        self.number_of_documents = 0
        self.vocabulary_size = 0

    def build_corpus(self):
        """
        Read document, fill in self.documents, a list of list of word
        self.documents = [["the", "day", "is", "nice", "the", ...], [], []...]
        
        Update self.number_of_documents
        """
        with open('data/test.txt') as file:
            lists = file.readlines()
        master_list = []
        for i in lists:
            words = i.strip()
            master_list.append(words.split())
        self.documents = master_list
        self.number_of_documents = len(master_list)
        

    def build_vocabulary(self):
        """
        Construct a list of unique words in the whole corpus. Put it in self.vocabulary
        for example: ["rain", "the", ...]

        Update self.vocabulary_size
        """

        flat_ls = []
        for i in self.documents:
            for j in i:
                if j not in flat_ls:
                    flat_ls.append(j)
                    flat_ls = [item for item in flat_ls if not item.isdigit()]
        self.vocabulary = flat_ls
        
        self.vocabulary_size = len(self.vocabulary)

    def build_term_doc_matrix(self):
        """
        Construct the term-document matrix where each row represents a document, 
        and each column represents a vocabulary term.

        self.term_doc_matrix[i][j] is the count of term j in document i
        """

        total_dict = []
        for i in self.documents:
            cur_dict = list(map(lambda word: {self.vocabulary.index(word): i.count(word)}, self.vocabulary))
            reduced = reduce(lambda r, d: r.update(d) or r, cur_dict, {})
            total_dict.append(reduced)
        self.term_doc_matrix = total_dict


    def initialize_randomly(self, number_of_topics):
        """
        Randomly initialize the matrices: document_topic_prob and topic_word_prob
        which hold the probability distributions for P(z | d) and P(w | z): self.document_topic_prob, and self.topic_word_prob

        Don't forget to normalize! 
        HINT: you will find numpy's random matrix useful [https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.random.random.html]
        """
        self.document_topic_prob = np.random.random_sample((self.number_of_documents, number_of_topics))
        self.document_topic_prob = normalize(self.document_topic_prob)

        self.topic_word_prob = np.random.random_sample((number_of_topics, len(self.vocabulary)))
        self.topic_word_prob = normalize(self.topic_word_prob)


    def initialize_uniformly(self, number_of_topics):
        """
        Initializes the matrices: self.document_topic_prob and self.topic_word_prob with a uniform 
        probability distribution. This is used for testing purposes.

        DO NOT CHANGE THIS FUNCTION
        """
        self.document_topic_prob = np.ones((self.number_of_documents, number_of_topics))
        self.document_topic_prob = normalize(self.document_topic_prob)

        self.topic_word_prob = np.ones((number_of_topics, len(self.vocabulary)))
        self.topic_word_prob = normalize(self.topic_word_prob)

    def initialize(self, number_of_topics, random=False):
        """ Call the functions to initialize the matrices document_topic_prob and topic_word_prob
        """
        print("Initializing...")
        self.number_of_topics = number_of_topics
        if random:
            self.initialize_randomly(number_of_topics)
        else:
            self.initialize_uniformly(number_of_topics)

    def expectation_step(self):
        """ The E-step updates P(z | w, d)
        """
        print("E step:")
        self.topic_prob = np.ones((self.number_of_documents,self.number_of_topics,self.vocabulary_size))
        cur_doc_num = 0
        cur_vocab_num = 0
        while cur_doc_num < self.number_of_documents:
            while cur_vocab_num < self.vocabulary_size:
                probability_stage = self.topic_word_prob[:,cur_vocab_num] * self.document_topic_prob[cur_doc_num ,:]
                probability = normalize(probability_stage[np.newaxis,:])
                self.topic_prob[cur_vocab_num,:,cur_doc_num] = probability

                cur_vocab_num = cur_vocab_num + 1
            
            cur_doc_num = cur_doc_num + 1
            

    def maximization_step(self, number_of_topics):
        """ The M-step updates P(w | z)
        """
        print("M step:")
        
        # update P(w | z)
        cur_topic_num = 0
        cur_vocab_num = 0
        cur_doc_num = 0
        while cur_topic_num < number_of_topics:
            while cur_vocab_num < self.vocabulary_size:
                total = 0
                while cur_doc_num < self.number_of_documents:
                    total = total + self.term_doc_matrix[cur_doc_num][cur_vocab_num] + self.topic_prob[cur_doc_num, cur_topic_num, cur_vocab_num]
                    cur_doc_num = cur_doc_num + 1
                self.topic_word_prob[cur_topic_num][cur_vocab_num] = total
                cur_vocab_num = cur_vocab_num + 1  
            cur_topic_num = cur_topic_num + 1  
        self.topic_word_prob = normalize(self.topic_word_prob)                

        
        cur_topic_num = 0
        cur_vocab_num = 0
        cur_doc_num = 0
        while cur_doc_num < self.number_of_documents:
            while cur_topic_num < number_of_topics:
                total = 0
                while cur_vocab_num < self.vocabulary_size:
                    total = total + self.term_doc_matrix[cur_doc_num][cur_vocab_num] + self.topic_prob[cur_doc_num, cur_topic_num, cur_vocab_num]
                    cur_vocab_num= cur_vocab_num + 1
                self.document_topic_prob[cur_doc_num][cur_topic_num] = total 

                
                cur_topic_num = cur_topic_num + 1
            cur_doc_num = cur_doc_num + 1   
        self.document_topic_prob = normalize(self.document_topic_prob)                
        
        


    def calculate_likelihood(self, number_of_topics):
        """ Calculate the current log-likelihood of the model using
        the model's updated probability matrices
        
        Append the calculated log-likelihood to self.likelihoods

        """
        cur_topic_num = 0
        cur_vocab_num = 0
        cur_doc_num = 0
        
        while cur_doc_num < self.number_of_documents - 1:
            while cur_topic_num < number_of_topics :
                total = 0
                while cur_vocab_num < self.vocabulary_size - 1:
                    total = (self.topic_prob[cur_doc_num, cur_topic_num, cur_vocab_num])
                    cur_vocab_num = cur_vocab_num + 1
                total = np.log(total) + total
                cur_topic_num = cur_topic_num + 1
            cur_doc_num = cur_doc_num + 1
        return total

    def plsa(self, number_of_topics, max_iter, epsilon):

        """
        Model topics.
        """
        print ("EM iteration begins...")
        
        # build term-doc matrix
        self.build_term_doc_matrix()
        
        # Create the counter arrays.
        
        # P(z | d, w)
        self.topic_prob = np.zeros([self.number_of_documents, number_of_topics, self.vocabulary_size], dtype=np.float)

        # P(z | d) P(w | z)
        self.initialize(number_of_topics, random=True)

        # Run the EM algorithm
        current_likelihood = 0.0

        for iteration in range(max_iter):
            print("Iteration #" + str(iteration + 1) + "...")

            self.expectation_step()
            self.maximization_step(number_of_topics)
            self.calculate_likelihood(number_of_topics)

            




def main():
    documents_path = 'data/test.txt'
    corpus = Corpus(documents_path)  # instantiate corpus
    corpus.build_corpus()
    corpus.build_vocabulary()
    print(corpus.vocabulary)
    print("Vocabulary size:" + str(len(corpus.vocabulary)))
    print("Number of documents:" + str(len(corpus.documents)))
    number_of_topics = 2
    max_iterations = 50
    epsilon = 0.001
    corpus.plsa(number_of_topics, max_iterations, epsilon)



if __name__ == '__main__':
    main()
