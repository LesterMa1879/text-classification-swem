from load_data import read_vectors, load_datasets


class Word2vec:
    def __init__(self, model_path, model_size, data_size):
        X_train_data, y_train, X_test_data, y_test = load_datasets(data_size)
        self.X_train_data = X_train_data
        self.y_train = y_train
        self.X_test_data = X_test_data
        self.y_test = y_test
        self.model = read_vectors(model_path, model_size)
        self.data_size = data_size

    def vectorization(self):
        result = []
        for data in [self.X_train_data, self.X_test_data]:
            vectors = []
            for f_index in range(len(data)):
                file = data[f_index]
                temp_vec = []
                for word in file:
                    if word in self.model:
                        temp_vec.append(self.model[word])
                vectors.append(temp_vec)
            result.append(vectors)
        return result[0], self.y_train, result[1], self.y_test