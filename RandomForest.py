from sklearn.ensemble import RandomForestClassifier

from BinaryClassificationModel import BinaryClassificationModel


class RandomForest(BinaryClassificationModel):
    def __init__(self) -> None:
        super().__init__()
        self.name = "RandomForest"
        self.model = RandomForestClassifier(n_estimators=1000)

    def predict(self, X):
        preds = self.model.predict_proba(X)
        return [pred[1] for pred in preds]

    def fit(self, X, y) -> None:
        self.model.fit(X, y)
