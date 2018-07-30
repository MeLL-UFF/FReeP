class Parameter():
    def __init__(self, value):
        self.value = value

    def evaluate(self, X):
        return X[self.value]