import pytest

from BinaryClassificationModel import BinaryClassificationModel


class TestBinaryClassificationModel:
    def test_get_confusion_matrix_not_calculated(self):
        with pytest.raises(ValueError):
            BinaryClassificationModel().get_confusion_matrix()

    def test_get_accuracy_not_calculated(self):
        with pytest.raises(ValueError):
            BinaryClassificationModel().get_accuracy()

    def test_get_recall_not_calculated(self):
        with pytest.raises(ValueError):
            BinaryClassificationModel().get_recall()

    def test_get_sensitivity_not_calculated(self):
        with pytest.raises(ValueError):
            BinaryClassificationModel().get_sensitivity()

    def test_get_specificity_not_calculated(self):
        with pytest.raises(ValueError):
            BinaryClassificationModel().get_specificity()

    def test_get_precision_not_calculated(self):
        with pytest.raises(ValueError):
            BinaryClassificationModel().get_precision()

    def test_get_f1_score_not_calculated(self):
        with pytest.raises(ValueError):
            BinaryClassificationModel().get_f1_score()

    def test_get_support_not_calculated(self):
        with pytest.raises(ValueError):
            BinaryClassificationModel().get_support()

    def test_get_auc_not_calculated(self):
        with pytest.raises(ValueError):
            BinaryClassificationModel().get_auc()

    def test_calculate_quality_only_true(self):
        y_true = [1 for i in range(10)]
        y_pred = [1 for i in range(10)]

        model = BinaryClassificationModel()

        model.calculate_quality(y_true, y_pred)

        assert model.get_accuracy() == 1
        assert model.get_recall() == 1
        assert model.get_sensitivity() == 1
        # not calculated because of devision by zero
        with pytest.raises(ValueError):
            model.get_specificity()
        assert model.get_precision() == 1
        assert model.get_f1_score() == 1
        assert model.get_support() == 10

        with pytest.raises(ValueError):
            model.get_auc()

    def test_calculate_quality_only_false(self):
        y_true = [0 for i in range(10)]
        y_pred = [0 for i in range(10)]

        model = BinaryClassificationModel()

        model.calculate_quality(y_true, y_pred)

        assert model.get_accuracy() == 1
        with pytest.raises(ValueError):
            model.get_recall()
        # not calculated because of devision by zero
        with pytest.raises(ValueError):
            model.get_sensitivity()
        assert model.get_specificity() == 1
        with pytest.raises(ValueError):
            model.get_precision()  # not calculated because of devision by zero
        with pytest.raises(ValueError):
            model.get_f1_score()  # not calculated because of devision by zero
        assert model.get_support() == 0

    def test_calculate_quality_diffrent(self):
        y_true = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
        y_pred = [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]

        model = BinaryClassificationModel()

        model.calculate_quality(y_true, y_pred)

        assert model.get_accuracy() == 0.5
        assert model.get_recall() == 0.5
        assert model.get_sensitivity() == 0.5
        assert model.get_specificity() == 0.5
        assert model.get_precision() == 0.5
        assert model.get_f1_score() == 0.5
        assert model.get_support() == 6

        with pytest.raises(ValueError):
            model.get_auc()
