from sklearn.svm import SVC

from BinaryClassificationModel import BinaryClassificationModel


class SupportVectorMachine(BinaryClassificationModel):
    def __init__(self) -> None:
        super().__init__()
        self.name = "SupportVectorMachine"
        self.model = SVC(probability=True)

    def predict(self, X):
        preds = self.model.predict_proba(X)
        return [pred[1] for pred in preds]

    def fit(self, X, y) -> None:
        self.model.fit(X, y)
