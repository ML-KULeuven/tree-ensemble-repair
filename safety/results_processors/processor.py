from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Tuple

class Processor:
    def __init__(self, mr_config, tree_config, fa_config):
        self.mr_config = mr_config
        self.tree_config = tree_config
        self.fa_config = fa_config
    
    @abstractmethod
    def get_result_file_name(self):
        """
            Abstract base method for identifying the correct filepath
        """
        pass
    
    @abstractmethod
    def write_results(self, logs, **kwargs):
        """
            Abstract base method for writing results from logs 
        """
        pass
    
    @abstractmethod
    def results_to_latex(self):
        pass