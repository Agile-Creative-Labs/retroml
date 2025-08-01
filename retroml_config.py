# retroml_config.py (Simple version - no Pydantic)
'''
// Copyright (c) Agile Creative Labs Inc.
// Licensed under the MIT License.

'''
from typing import Optional, List, Dict, Any

class RetroMLConfig:
    """
    Simple configuration class for RetroML pipelines (no Pydantic dependencies)
    """
    def __init__(self, 
                 dataset_path: str,
                 problem_type: str,
                 target_column: Optional[str] = None,
                 features: Optional[List[str]] = None,
                 model_settings: Optional[Dict] = None,
                 preprocessing: Optional[Dict] = None,
                 **kwargs):
        self.dataset_path = dataset_path
        self.problem_type = problem_type
        self.target_column = target_column
        self.features = features
        self.model_settings = model_settings or {}
        self.preprocessing = preprocessing or {}
        
        # Store any additional fields
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def dict(self):
        """Return config as dictionary (for compatibility)"""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
    
    def model_dump(self):
        """Return config as dictionary (Pydantic v2 compatibility)"""
        return self.dict()