import unittest
import framework.configparser.yamlparser as yamlparser

class TestYamlParser(unittest.TestCase):
    def setUp(self):
        self.test_yaml_file_path = "../../experiment/config/actual_sample.yaml"

    def test_parse_yaml_file_to_config_dict(self):
        config_dict = yamlparser.parse_yaml_file_to_config_dict(self.test_yaml_file_path)
        self.assertEqual(len(config_dict.keys()), 3)

