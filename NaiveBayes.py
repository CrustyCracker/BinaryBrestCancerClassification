from sklearn.naive_bayes import GaussianNB
from BinaryClassificationModel import BinaryClassificationModel


class NaiveBayes(BinaryClassificationModel):
    def __init__(self) -> None:
        super().__init__()
        self.name = 'NaiveBayes'
        self.model = GaussianNB()

    def predict(self, X):
        preds = self.model.predict_proba(X)

        return [pred[1] for pred in preds]

    def fit(self, X, y):
        self.model.fit(X, y)
