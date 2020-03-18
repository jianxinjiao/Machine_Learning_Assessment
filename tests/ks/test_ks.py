from unittest import TestCase

from Ks.ks_curve import get_ks_curve
from exceptions import AssessmentValueException


class MyTestCase(TestCase):
    def setUp(self):
        self.labels = [0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0]
        self.pos_label = ['F', 'T', 'T', 'F', 'T', 'T', 'F', 'F', 'T', 'T', 'T', 'T', 'F', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'F', 'T', 'T', 'F', 'F', 'F', 'T', 'T', 'T', 'T', 'F', 'F', 'T', 'F', 'T', 'T', 'T', 'T', 'T', 'F', 'F', 'F', 'T', 'T', 'T', 'F', 'T', 'F']
        self.rates = [0.6, 0.7, 0.3, 0.5, 0.7, 0.6, 0.5, 0.2, 0.5, 0.6, 0.9, 0.7, 0.2, 0.4, 0.9, 0.9, 0.9, 0.9, 0.7, 0.9, 0.9, 0.6, 0.2, 0.3, 0.9, 0.4, 0.9, 0.8, 0.7, 0.6, 0.2, 0.4, 0.6, 0.2, 0.7, 0.5, 0.1, 0.8, 0.9, 0.3, 0.4, 0.2, 0.7, 0.7, 0.8, 0.3, 0.8, 0.1]
        self.error_rates = [0.6, 0.7]

    def test_get_roc_curve(self):
        # 测试roc曲线，标签与概率不等长的情况
        self.assertRaises(AssessmentValueException, get_ks_curve, self.labels, self.error_rates)
        # 测试roc曲线
        fpr, tpr, thre, ks = get_ks_curve(self.labels, self.rates)
        # 测试标签非0 1 的情况
        fpr, tpr, thre, ks = get_ks_curve(self.pos_label, self.rates, pos_label='T')

    def tearDown(self):
        pass
