#!/usr/bin/env python3
"""
Simple test for RetroML
"""

import unittest
import json
import os
from pathlib import Path

class TestRetroML(unittest.TestCase):
    
    def test_config_files_exist(self):
        """Test that example config files exist"""
        self.assertTrue(Path('configs/classification_example.json').exists())
        self.assertTrue(Path('configs/regression_example.json').exists())
    
    def test_data_files_exist(self):
        """Test that sample data files exist"""
        self.assertTrue(Path('data/customers.csv').exists())
        self.assertTrue(Path('data/houses.csv').exists())
    
    def test_config_format(self):
        """Test that config files have correct format"""
        with open('configs/classification_example.json', 'r') as f:
            config = json.load(f)
        
        required_keys = ['dataset', 'problem_type', 'preprocessing', 'model', 'output']
        for key in required_keys:
            self.assertIn(key, config)

if __name__ == '__main__':
    unittest.main()
