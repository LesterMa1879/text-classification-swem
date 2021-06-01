
import numpy as np
import MicroTokenizer


class SWEM():

    def __init__(self,choise):
        self.choise = choise

    def get_swem(self, data):
        swem_result = []
        for word_embeddings in data:
            if self.choise == 0:
                swem_result.append(self.average_pooling(np.array(word_embeddings)))
            elif self.choise == 1:
                swem_result.append(self.max_pooling(np.array(word_embeddings)))
            elif self.choise == 2:
                swem_result.append(self.concat_average_max_pooling(np.array(word_embeddings)))
            elif self.choise == 3:
                swem_result.append(self.hierarchical_pooling(np.array(word_embeddings), 5))
        return swem_result

    def average_pooling(self, word_embeddings):
        return np.mean(word_embeddings, axis=0)

    def max_pooling(self, word_embeddings):
        return np.max(word_embeddings, axis=0)

    def concat_average_max_pooling(self, word_embeddings):
        return np.r_[np.mean(word_embeddings, axis=0), np.max(word_embeddings, axis=0)]

    def hierarchical_pooling(self, word_embeddings, n):

        text_len = word_embeddings.shape[0]
        if n > text_len:
            raise ValueError(f"window size must be less than text length / window_size:{n} text_length:{text_len}")
        window_average_pooling_vec = [np.mean(word_embeddings[i:i + n], axis=0) for i in range(text_len - n + 1)]

        return np.max(window_average_pooling_vec, axis=0)
