from unittest import TestCase

from Score.conlleval import confusion_matrix_score
from exceptions import AssessmentSavePathException

import os


class TestConlleval(TestCase):
    def setUp(self):
        self.file_type_dir = r'./Data/label_data.txt'
        self.file_type2_dir = r'./Data/label_data_2.txt'

        self.save_res_dir = r'./Data/res_score.csv'
        self.error_save_res_dir = r'./Data/res_score.txt'
        self.save_res_dir_2 = r'./Data/res_score_2.csv'

    def test_confusion_matrix_score(self):
        # 测试type='1'类型
        res = confusion_matrix_score(self.file_type_dir, type='1', save_res_dir='')
        assert len(res) != 0
        # 测试type='2'类型
        res = confusion_matrix_score(self.file_type2_dir, type='2', save_res_dir='')
        assert len(res) != 0
        # 测试保存地址
        res = confusion_matrix_score(self.file_type_dir, type='1', save_res_dir=self.save_res_dir)
        assert len(res) != 0
        self.assertRaises(AssessmentSavePathException, confusion_matrix_score, self.file_type_dir, type='1', save_res_dir=self.error_save_res_dir)

        res = confusion_matrix_score(self.file_type2_dir, type='2', save_res_dir=self.save_res_dir_2)

    def tearDown(self):
        os.remove(self.save_res_dir)
        os.remove(self.save_res_dir_2)
        pass
