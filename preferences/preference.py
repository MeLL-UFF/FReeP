class Preference():
    def __init__(self, operand1, operator, operand2):
        self.operand1 = operand1
        self.operator = operator
        self.operand2 = operand2

    def evaluate(self, X):
        return X.loc[self.operation(self.operand1.evaluate(
            X), self.operator, self.operand2.evaluate(X))]

    def operation(self, operand1, operator, operand2):
        if operator == '=':
            return operand1[operand1 == operand2].index
        elif operator == '>':
            return operand1[operand1 > operand2].index
        elif operator == '<':
            return operand1[operand1 < operand2].index
        elif operator == '>=':
            return operand1[operand1 >= operand2].index
        elif operator == '<=':
            return operand1[operand1 <= operand2].index
