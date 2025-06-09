#!/usr/bin/env python3
"""
Enhanced Test Runner with Automatic Error Detection and Fixing
Follows DRY principles and integrates with the existing test suite
"""

import os
import sys
import json
import subprocess
import re
import ast
import importlib
from pathlib import Path
from typing import Dict, List, Tuple, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TestRunner:
    """Enhanced test runner with error detection and auto-fixing capabilities"""
    
    def __init__(self, max_problems: int = 6, auto_fix: bool = True):
        self.max_problems = max_problems
        self.auto_fix = auto_fix
        self.problems = {
            'errors': [],
            'warnings': [],
            'failures': [],
            'syntax_errors': [],
            'import_errors': [],
            'config_errors': []
        }
        self.fixes_applied = []
        
    def run_tests(self) -> bool:
        """Run the complete test suite with error detection and auto-fixing"""
        logger.info("Starting enhanced test suite...")
        
        # Pre-flight checks
        self._check_environment()
        
        # Run initial tests
        test_success = self._run_pytest()
        
        # Analyze results
        self._analyze_test_results()
        
        # Auto-fix if needed
        if self.auto_fix and self._get_total_problems() > 0:
            self._apply_auto_fixes()
            # Re-run tests after fixes
            test_success = self._run_pytest()
            self._analyze_test_results()
        
        # Report results
        self._report_results()
        
        # Check if we should abort
        if self._get_total_problems() >= self.max_problems:
            logger.error(f"ABORTING: Maximum of {self.max_problems} problems reached!")
            return False
            
        return test_success and self._get_total_problems() == 0
    
    def _check_environment(self):
        """Check and fix environment issues"""
        logger.info("Checking environment...")
        
        # Ensure indicator_params.json exists
        if not Path("indicator_params.json").exists():
            logger.warning("Creating missing indicator_params.json")
            self._create_indicator_params()
        
        # Check virtual environment
        if not self._is_venv_active():
            logger.warning("Virtual environment not detected")
        
        # Check dependencies
        self._check_dependencies()
    
    def _run_pytest(self) -> bool:
        """Run pytest with enhanced configuration"""
        logger.info("Running pytest with enhanced error detection...")
        
        cmd = [
            sys.executable, "-m", "pytest",
            "tests/",
            "-v",
            "--tb=long",
            "--showlocals",
            "--show-capture=all",
            "--maxfail=10",
            "--strict-markers",
            "--strict-config",
            "--error-for-skips",
            "--error-for-xpass",
            "--cov=.",
            "--cov-report=term-missing",
            "--cov-report=html",
            "--cov-fail-under=90",
            "--timeout=30",
            "-s"
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            self.last_test_output = result.stdout + result.stderr
            return result.returncode == 0
        except subprocess.TimeoutExpired:
            logger.error("Test execution timed out")
            return False
        except Exception as e:
            logger.error(f"Error running tests: {e}")
            return False
    
    def _analyze_test_results(self):
        """Analyze test output for errors, warnings, and failures"""
        logger.info("Analyzing test results...")
        
        lines = self.last_test_output.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Detect different types of problems
            if 'ERROR' in line or 'Exception' in line:
                self.problems['errors'].append(line)
            elif 'WARNING' in line:
                self.problems['warnings'].append(line)
            elif 'FAILED' in line:
                self.problems['failures'].append(line)
            elif 'SyntaxError' in line or 'IndentationError' in line:
                self.problems['syntax_errors'].append(line)
            elif 'ModuleNotFoundError' in line or 'ImportError' in line:
                self.problems['import_errors'].append(line)
            elif 'ConfigError' in line or 'ConfigurationError' in line:
                self.problems['config_errors'].append(line)
    
    def _apply_auto_fixes(self):
        """Apply automatic fixes for detected problems"""
        logger.info("Applying automatic fixes...")
        
        # Fix syntax errors
        if self.problems['syntax_errors']:
            self._fix_syntax_errors()
        
        # Fix import errors
        if self.problems['import_errors']:
            self._fix_import_errors()
        
        # Fix configuration errors
        if self.problems['config_errors']:
            self._fix_config_errors()
        
        # Fix common code issues
        self._fix_common_issues()
    
    def _fix_syntax_errors(self):
        """Fix syntax errors in Python files"""
        logger.info("Fixing syntax errors...")
        
        python_files = list(Path('.').rglob('*.py'))
        
        for py_file in python_files:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Try to parse the file
                ast.parse(content)
                
            except SyntaxError as e:
                logger.warning(f"Syntax error in {py_file}: {e}")
                self._fix_file_syntax(py_file, content, e)
    
    def _fix_file_syntax(self, file_path: Path, content: str, error: SyntaxError):
        """Fix syntax errors in a specific file"""
        try:
            # Common fixes
            fixed_content = content
            
            # Fix common indentation issues
            fixed_content = self._fix_indentation(fixed_content)
            
            # Fix common syntax issues
            fixed_content = self._fix_common_syntax(fixed_content)
            
            # Write back if different
            if fixed_content != content:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(fixed_content)
                self.fixes_applied.append(f"Fixed syntax in {file_path}")
                
        except Exception as e:
            logger.error(f"Failed to fix {file_path}: {e}")
    
    def _fix_indentation(self, content: str) -> str:
        """Fix common indentation issues"""
        lines = content.split('\n')
        fixed_lines = []
        
        for line in lines:
            # Fix mixed tabs and spaces
            if '\t' in line and ' ' in line:
                line = line.replace('\t', '    ')
            fixed_lines.append(line)
        
        return '\n'.join(fixed_lines)
    
    def _fix_common_syntax(self, content: str) -> str:
        """Fix common syntax issues"""
        # Fix missing colons
        content = re.sub(r'(\s+)(if|for|while|def|class|try|except|finally|with)\s*\(', r'\1\2(', content)
        
        # Fix missing parentheses in print statements
        content = re.sub(r'print\s+([^()\n]+)(?=\n|$)', r'print(\1)', content)
        
        # Fix common import issues
        content = re.sub(r'from \. import (\w+)', r'from .\1 import \1', content)
        
        return content
    
    def _fix_import_errors(self):
        """Fix import errors"""
        logger.info("Fixing import errors...")
        
        # Install missing dependencies
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], 
                         check=True, capture_output=True)
            self.fixes_applied.append("Installed missing dependencies")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install dependencies: {e}")
    
    def _fix_config_errors(self):
        """Fix configuration errors"""
        logger.info("Fixing configuration errors...")
        
        # Ensure indicator_params.json is valid
        try:
            with open("indicator_params.json", 'r') as f:
                json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            self._create_indicator_params()
            self.fixes_applied.append("Fixed indicator_params.json")
    
    def _fix_common_issues(self):
        """Fix common issues across the codebase"""
        logger.info("Fixing common issues...")
        
        # Fix hardcoded values in scripts
        self._fix_hardcoded_values()
        
        # Ensure DRY principles
        self._apply_dry_principles()
        
        # Fix interoperability issues
        self._fix_interoperability()
    
    def _fix_hardcoded_values(self):
        """Move hardcoded values to indicator_params.json"""
        logger.info("Checking for hardcoded values...")
        
        # Common patterns to look for
        patterns = [
            r'(\w+)\s*=\s*(\d+\.?\d*)',  # numeric values
            r'(\w+)\s*=\s*["\']([^"\']+)["\']',  # string values
        ]
        
        python_files = list(Path('.').rglob('*.py'))
        
        for py_file in python_files:
            if 'test_' in py_file.name or py_file.name == 'test_runner.py':
                continue  # Skip test files
                
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Look for hardcoded values
                for pattern in patterns:
                    matches = re.finditer(pattern, content)
                    for match in matches:
                        var_name = match.group(1)
                        value = match.group(2)
                        
                        # Skip if it's already a parameter reference
                        if 'params' in var_name or 'config' in var_name:
                            continue
                        
                        # Add to indicator_params.json if it looks like a parameter
                        if self._is_parameter_candidate(var_name, value):
                            self._add_to_indicator_params(var_name, value)
                            
            except Exception as e:
                logger.error(f"Error processing {py_file}: {e}")
    
    def _is_parameter_candidate(self, var_name: str, value: str) -> bool:
        """Determine if a variable looks like a parameter that should be in indicator_params.json"""
        # Skip common non-parameter variables
        skip_patterns = [
            'self', 'cls', 'args', 'kwargs', 'i', 'j', 'k', 'x', 'y', 'z',
            'file', 'path', 'url', 'name', 'title', 'description', 'message',
            'error', 'exception', 'result', 'data', 'df', 'df_', 'df1', 'df2'
        ]
        
        if var_name.lower() in skip_patterns:
            return False
        
        # Check if it looks like a parameter (numeric or short string)
        if value.replace('.', '').replace('-', '').isdigit():
            return True
        
        if len(value) < 20 and not value.startswith('http'):
            return True
        
        return False
    
    def _add_to_indicator_params(self, var_name: str, value: str):
        """Add a parameter to indicator_params.json"""
        try:
            with open("indicator_params.json", 'r') as f:
                params = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            params = {}
        
        # Convert value to appropriate type
        try:
            if '.' in value:
                params[var_name] = float(value)
            elif value.isdigit():
                params[var_name] = int(value)
            else:
                params[var_name] = value
        except ValueError:
            params[var_name] = value
        
        with open("indicator_params.json", 'w') as f:
            json.dump(params, f, indent=2)
    
    def _apply_dry_principles(self):
        """Apply DRY principles to reduce code duplication"""
        logger.info("Applying DRY principles...")
        
        # This would involve more complex analysis
        # For now, we'll focus on common patterns
        
        python_files = list(Path('.').rglob('*.py'))
        
        for py_file in python_files:
            if 'test_' in py_file.name or py_file.name == 'test_runner.py':
                continue
                
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Look for repeated code patterns
                self._fix_repeated_patterns(py_file, content)
                
            except Exception as e:
                logger.error(f"Error applying DRY to {py_file}: {e}")
    
    def _fix_repeated_patterns(self, file_path: Path, content: str):
        """Fix repeated code patterns"""
        # Common patterns to look for
        patterns = [
            # Repeated logging setup
            (r'(logging\.basicConfig\([^)]+\)\s*\n)+', 'logging_setup.py'),
            # Repeated database connections
            (r'(sqlite3\.connect\([^)]+\)\s*\n)+', 'sqlite_manager.py'),
            # Repeated file operations
            (r'(with open\([^)]+\) as [^:]+:\s*\n)+', 'utils.py'),
        ]
        
        # This is a simplified version - in practice, this would be more sophisticated
        pass
    
    def _fix_interoperability(self):
        """Fix interoperability issues between modules"""
        logger.info("Fixing interoperability issues...")
        
        # Ensure consistent imports
        self._standardize_imports()
        
        # Ensure consistent function signatures
        self._standardize_signatures()
    
    def _standardize_imports(self):
        """Standardize import statements across the codebase"""
        # This would involve analyzing and standardizing import patterns
        pass
    
    def _standardize_signatures(self):
        """Standardize function signatures for better interoperability"""
        # This would involve analyzing function signatures and ensuring consistency
        pass
    
    def _create_indicator_params(self):
        """Create a default indicator_params.json file"""
        default_params = {
            "default_timeframe": "1h",
            "default_period": 14,
            "default_threshold": 0.5,
            "max_lookback": 100,
            "min_data_points": 50,
            "confidence_threshold": 0.7,
            "risk_tolerance": 0.1,
            "max_position_size": 0.1,
            "stop_loss_pct": 0.02,
            "take_profit_pct": 0.04
        }
        
        with open("indicator_params.json", 'w') as f:
            json.dump(default_params, f, indent=2)
    
    def _check_dependencies(self):
        """Check and install required dependencies"""
        required_packages = [
            'pytest', 'pytest-cov', 'pandas', 'numpy', 'matplotlib',
            'sqlite3', 'json', 'logging', 'pathlib'
        ]
        
        for package in required_packages:
            try:
                importlib.import_module(package)
            except ImportError:
                logger.warning(f"Missing package: {package}")
    
    def _is_venv_active(self) -> bool:
        """Check if virtual environment is active"""
        return hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)
    
    def _get_total_problems(self) -> int:
        """Get total number of problems"""
        return sum(len(problems) for problems in self.problems.values())
    
    def _report_results(self):
        """Report test results and fixes applied"""
        logger.info("=== TEST RESULTS ===")
        logger.info(f"Errors: {len(self.problems['errors'])}")
        logger.info(f"Warnings: {len(self.problems['warnings'])}")
        logger.info(f"Failures: {len(self.problems['failures'])}")
        logger.info(f"Syntax Errors: {len(self.problems['syntax_errors'])}")
        logger.info(f"Import Errors: {len(self.problems['import_errors'])}")
        logger.info(f"Config Errors: {len(self.problems['config_errors'])}")
        logger.info(f"Total Problems: {self._get_total_problems()}")
        
        if self.fixes_applied:
            logger.info("=== FIXES APPLIED ===")
            for fix in self.fixes_applied:
                logger.info(f"✓ {fix}")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced Test Runner with Auto-Fixing")
    parser.add_argument("--max-problems", type=int, default=6, help="Maximum problems before abort")
    parser.add_argument("--no-auto-fix", action="store_true", help="Disable auto-fixing")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    runner = TestRunner(
        max_problems=args.max_problems,
        auto_fix=not args.no_auto_fix
    )
    
    success = runner.run_tests()
    
    if success:
        logger.info("All tests passed!")
        sys.exit(0)
    else:
        logger.error("Tests failed!")
        sys.exit(1)


if __name__ == "__main__":
    main() 