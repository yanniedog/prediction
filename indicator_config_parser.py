# indicator_config_parser.py

import ast
import os
from typing import Dict, List, Optional, Tuple

class IndicatorConfigParser(ast.NodeVisitor):
    """
    AST Node Visitor to parse indicators.py and extract configurable indicators and their parameters.
    """
    def __init__(self):
        self.configurable_indicators = {}
        self.current_indicator = None
        self.in_compute_function = False

    def visit_FunctionDef(self, node: ast.FunctionDef):
        if node.name == 'compute_configured_indicators':
            self.in_compute_function = True
            self.generic_visit(node)
            self.in_compute_function = False
        else:
            self.generic_visit(node)

    def visit_If(self, node: ast.If):
        if not self.in_compute_function:
            return

        test = node.test
        indicator_name = None

        if isinstance(test, ast.Compare):
            if (isinstance(test.left, ast.Name) and test.left.id == 'base_indicator' and
                len(test.ops) == 1 and isinstance(test.ops[0], ast.Eq) and
                len(test.comparators) == 1 and isinstance(test.comparators[0], ast.Str)):
                indicator_name = test.comparators[0].s

        if indicator_name:
            self.current_indicator = indicator_name
            for stmt in node.body:
                if isinstance(stmt, ast.Assign):
                    for target in stmt.targets:
                        if isinstance(target, ast.Name) and target.id == 'params':
                            params = self.extract_params(stmt.value)
                            if params:
                                self.configurable_indicators[indicator_name] = params
            self.current_indicator = None

        self.generic_visit(node)

    def extract_params(self, node: ast.Dict) -> Optional[Dict]:
        """
        Extract parameters from a Dict node.
        Returns a dictionary of parameter names and their default values.
        """
        params = {}
        for key, value in zip(node.keys, node.values):
            if isinstance(key, ast.Str):
                param_name = key.s
                param_value = self.get_constant_value(value)
                if param_value is not None:
                    params[param_name] = param_value
        return params if params else None

    def get_constant_value(self, node: ast.AST):
        """
        Extract the constant value from AST nodes.
        Supports Num, Str, List, etc.
        """
        if isinstance(node, ast.Num):
            return node.n
        elif isinstance(node, ast.Str):
            return node.s
        elif isinstance(node, ast.List):
            return [self.get_constant_value(elt) for elt in node.elts]
        elif isinstance(node, ast.Tuple):
            return tuple(self.get_constant_value(elt) for elt in node.elts)
        elif isinstance(node, ast.Dict):
            return self.extract_params(node)
        else:
            return None

def parse_indicators_py(indicators_py_path: str) -> Dict[str, Dict]:
    """
    Parse indicators.py to extract configurable indicators and their parameters.
    
    Args:
        indicators_py_path (str): Path to indicators.py
    
    Returns:
        Dict[str, Dict]: A dictionary mapping indicator names to their parameter dictionaries.
    """
    if not os.path.exists(indicators_py_path):
        raise FileNotFoundError(f"File not found: {indicators_py_path}")
    
    with open(indicators_py_path, 'r', encoding='utf-8') as file:
        tree = ast.parse(file.read(), filename=indicators_py_path)
    
    parser = IndicatorConfigParser()
    parser.visit(tree)
    
    return parser.configurable_indicators

def get_configurable_indicators(indicators_py_path: str = 'indicators.py') -> List[str]:
    """
    Get a list of indicators that have configurable numerical parameters.
    
    Args:
        indicators_py_path (str): Path to indicators.py
    
    Returns:
        List[str]: List of configurable indicator names.
    """
    configs = parse_indicators_py(indicators_py_path)
    return list(configs.keys())

def get_indicator_parameters(indicator_name: str, indicators_py_path: str = 'indicators.py') -> Optional[Dict]:
    """
    Get the parameters for a specific indicator.
    
    Args:
        indicator_name (str): Name of the indicator.
        indicators_py_path (str): Path to indicators.py
    
    Returns:
        Optional[Dict]: Dictionary of parameter names and their default values, or None if not found.
    """
    configs = parse_indicators_py(indicators_py_path)
    return configs.get(indicator_name)

if __name__ == "__main__":
    indicators_path = 'indicators.py'
    try:
        configurable = get_configurable_indicators(indicators_path)
        print("Configurable Indicators:")
        for ind in configurable:
            params = get_indicator_parameters(ind, indicators_path)
            print(f"- {ind}:")
            for param, value in params.items():
                print(f"    {param}: {value}")
    except Exception as e:
        print(f"Error: {e}")
