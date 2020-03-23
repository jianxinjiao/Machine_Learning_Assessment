from unittest import TestCase

from Kappa.kappa_coefficient import get_kappa_coefficient
from exceptions import AssessmentValueException


class TestGetKappaCoefficient(TestCase):
    def setUp(self):
        self.file_type_dir = r'./Data/label_data.txt'
        self.file_type2_dir = r'./Data/label_data_2.txt'

    def test_get_kappa_coefficient(self):
        # 测试type传入错误
        self.assertRaises(AssessmentValueException, get_kappa_coefficient, self.file_type_dir, type='3')
        # 测试type='1'类型
        res = get_kappa_coefficient(self.file_type_dir, type='1')
        assert res > 0
        assert res < 1
        # 测试type='2'类型
        res = get_kappa_coefficient(self.file_type2_dir, type='2')
        assert res > 0
        assert res < 1

    def tearDown(self):
        pass
