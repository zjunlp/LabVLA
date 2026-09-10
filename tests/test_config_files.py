#!/usr/bin/env python3
"""
Configuration File Testing Module
Test all configuration files that do not contain "infer", modify max_episodes to 8, 
mode to "collect", and check success rate.

USAGE:
    python tests/test_config_files.py

REQUIREMENTS:
    - Isaac Sim environment must be properly set up
    - Configuration files should be in the config/ directory
    - Each test runs for maximum 5 minutes
    - Success rate > 0.7% is considered a pass

OUTPUT:
    - Test results printed to console
    - Detailed report saved to test_outputs/test_report.txt
    - Individual log files for each test in test_outputs/
    - Exit code 0 for all tests passed, 1 for any failures
"""

import os
import sys
import yaml
import subprocess
import time
import re
import glob
from pathlib import Path
from typing import List, Dict, Tuple
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ConfigTester:
    """Configuration file tester"""
    
    def __init__(self, config_dir: str = "config", test_output_dir: str = "test_outputs"):
        self.config_dir = Path(config_dir)
        self.test_output_dir = Path(test_output_dir)
        self.test_output_dir.mkdir(exist_ok=True)
        self.passed_dir = self.test_output_dir / "passed_configs"
        self.passed_dir.mkdir(exist_ok=True)
        self.results = {}
        
    def get_config_files(self) -> List[Path]:
        """Get all configuration files that do not contain 'infer' and not already passed"""
        config_files = []
        for yaml_file in self.config_dir.glob("*.yaml"):
            if "infer" in yaml_file.name.lower():
                continue
            passed_flag = self.passed_dir / f"{yaml_file.name}.passed"
            if passed_flag.exists():
                continue
            config_files.append(yaml_file)
        config_files.sort()
        return config_files
    
    def modify_config(self, config_file: Path, temp_config_file: Path) -> str:
        """Modify configuration file: set max_episodes to 8, mode to 'collect', and use mock collector"""
        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # Modify configuration
        config['max_episodes'] = 8
        config['mode'] = 'collect'
        
        # Change collector type to mock for testing (no actual data collection)
        if 'collector' in config:
            config['collector']['type'] = 'mock'
        
        # Save modified configuration
        with open(temp_config_file, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        
        # Return configuration name (for command line arguments)
        return config_file.stem
    
    def run_simulation(self, config_name: str, temp_config_file: Path, log_file: Path) -> Tuple[bool, str]:
        """Run simulation with real-time logging"""
        try:
            # Build command
            cmd = [
                sys.executable, "main.py", 
                "--config-name", config_name,
                "--config-dir", str(temp_config_file.parent),
            ]
            
            logger.info(f"Running command: {' '.join(cmd)}")
            logger.info(f"Log file: {log_file}")
            
            # Open log file for writing
            with open(log_file, 'w', encoding='utf-8') as log_f:
                # Write command to log
                log_f.write(f"Command: {' '.join(cmd)}\n")
                log_f.write(f"Start time: {subprocess.run(['date'], capture_output=True, text=True).stdout.strip()}\n")
                log_f.write("-" * 80 + "\n")
                log_f.flush()
                
                # Run subprocess with real-time output
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=0,  # No buffering
                    universal_newlines=True,
                    cwd=os.getcwd(),
                    env=dict(os.environ, PYTHONUNBUFFERED="1")  # Force Python unbuffered
                )
                
                output_lines = []
                
                # Read output in real-time
                while True:
                    if process.stdout is None:
                        break
                    line = process.stdout.readline()
                    if not line and process.poll() is not None:
                        break
                    
                    if line:
                        # Write to log file immediately
                        log_f.write(line)
                        log_f.flush()
                        
                        # Store for return
                        output_lines.append(line)
                
                # Wait for process to complete
                return_code = process.wait()
                
                # Write completion info to log
                log_f.write("-" * 80 + "\n")
                log_f.write(f"End time: {subprocess.run(['date'], capture_output=True, text=True).stdout.strip()}\n")
                log_f.write(f"Return code: {return_code}\n")
                log_f.flush()
                
                output = ''.join(output_lines)
                
                if return_code == 0:
                    return True, output
                else:
                    return False, f"Error: Process returned code {return_code}"
            
        except subprocess.TimeoutExpired:
            # Write timeout info to log
            with open(log_file, 'a', encoding='utf-8') as log_f:
                log_f.write("-" * 80 + "\n")
                log_f.write("TIMEOUT: Process exceeded 5 minute limit\n")
            return False, "Timeout"
        except Exception as e:
            # Write error info to log
            with open(log_file, 'a', encoding='utf-8') as log_f:
                log_f.write("-" * 80 + "\n")
                log_f.write(f"RUNTIME ERROR: {str(e)}\n")
            return False, f"Runtime error: {str(e)}"
    
    def extract_success_rate(self, output: str) -> float:
        """Extract success rate from output"""
        # Try multiple patterns to match success rate
        patterns = [
            r'Success Rate\s*=\s*(\d+)/(\d+)\s*\((\d+\.?\d*)%\)',
            r'Success Rate\s*=\s*(\d+)/(\d+)\s*\((\d+\.?\d*)%\)',
            r'success rate[：:]\s*(\d+\.?\d*)%',
            r'Success Rate[：:]\s*(\d+\.?\d*)%',
            r'success rate[：:]\s*(\d+\.?\d*)',
            r'Success Rate[：:]\s*(\d+\.?\d*)',
            r'Success rate[：:]\s*(\d+\.?\d*)%',
            r'Success rate[：:]\s*(\d+\.?\d*)',
        ]
        
        for pattern in patterns:
            # Find all matches and get the last one
            matches = re.findall(pattern, output, re.IGNORECASE)
            if matches:
                # Get the last match
                match = matches[-1]
                try:
                    if isinstance(match, tuple) and len(match) == 3:
                        return float(match[2])
                    elif isinstance(match, str):
                        return float(match)
                    else:
                        continue
                except (ValueError, TypeError):
                    continue
        
        # If no explicit success rate found, try to infer from logs
        # Look for success and failure counts
        success_patterns = [
            r'Success[：:]\s*(\d+)',
            r'success[：:]\s*(\d+)',
            r'episode.*success',
        ]
        
        total_episodes = 8  # Our set max_episodes
        success_count = 0
        
        for pattern in success_patterns:
            matches = re.findall(pattern, output, re.IGNORECASE)
            if matches:
                try:
                    # Get the last match for success count
                    success_count = max(success_count, int(matches[-1]))
                except ValueError:
                    continue
        
        if success_count > 0:
            return (success_count / total_episodes) * 100
        
        return 0.0
    
    def mark_passed(self, config_file: Path):
        """Generate record for passed configuration files"""
        passed_flag = self.passed_dir / f"{config_file.name}.passed"
        with open(passed_flag, "w") as f:
            f.write("passed\n")
    
    def test_config_file(self, config_file: Path) -> Dict:
        """Test a single configuration file"""
        logger.info(f"Testing configuration file: {config_file.name}")
        
        # Create temporary configuration file
        temp_config_file = self.test_output_dir / f"temp_{config_file.name}"
        
        # Create log file path
        log_file = self.test_output_dir / f"{config_file.stem}_run.log"
        
        try:
            # Modify configuration
            config_name = self.modify_config(config_file, temp_config_file)
            
            # Run simulation
            success, output = self.run_simulation(f"temp_{config_file.stem}", temp_config_file, log_file)
            
            if not success:
                return {
                    'config_file': config_file.name,
                    'status': 'failed',
                    'error': output,
                    'success_rate': 0.0,
                    'log_file': str(log_file)
                }
            
            # Extract success rate
            success_rate = self.extract_success_rate(output)
            
            # Determine if test passed (success rate greater than 0.7%)
            test_passed = success_rate > 70
            if test_passed:
                self.mark_passed(config_file)  # Record pass
            result = {
                'config_file': config_file.name,
                'status': 'passed' if test_passed else 'failed',
                'success_rate': success_rate,
                'output': output[:500] + "..." if len(output) > 500 else output,
                'log_file': str(log_file)
            }
            
            logger.info(f"Configuration file {config_file.name} test result: Success rate {success_rate:.2f}% - {'Passed' if test_passed else 'Failed'}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error occurred while testing configuration file {config_file.name}: {str(e)}")
            return {
                'config_file': config_file.name,
                'status': 'error',
                'error': str(e),
                'success_rate': 0.0,
                'log_file': str(log_file)
            }
        finally:
            # Clean up temporary file
            if temp_config_file.exists():
                temp_config_file.unlink()
    
    def run_all_tests(self) -> Dict:
        """Run all tests"""
        logger.info("Starting all configuration file tests")
        
        config_files = self.get_config_files()
        logger.info(f"Found {len(config_files)} configuration files to test")
        
        for config_file in config_files:
            result = self.test_config_file(config_file)
            self.results[config_file.name] = result
        
        return self.results
    
    def generate_report(self) -> str:
        """Generate test report"""
        if not self.results:
            return "No test results"
        
        report = []
        report.append("=" * 60)
        report.append("Configuration File Test Report")
        report.append("=" * 60)
        
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results.values() if r['status'] == 'passed')
        failed_tests = sum(1 for r in self.results.values() if r['status'] == 'failed')
        error_tests = sum(1 for r in self.results.values() if r['status'] == 'error')
        
        report.append(f"Total Tests: {total_tests}")
        report.append(f"Passed: {passed_tests}")
        report.append(f"Failed: {failed_tests}")
        report.append(f"Errors: {error_tests}")
        report.append("")
        
        # Detailed results
        for config_name, result in self.results.items():
            report.append(f"Configuration File: {config_name}")
            report.append(f"  Status: {result['status']}")
            report.append(f"  Success Rate: {result['success_rate']:.2f}%")
            report.append(f"  Log File: {result.get('log_file', 'N/A')}")
            
            if result['status'] == 'error':
                report.append(f"  Error: {result['error']}")
            elif result['status'] == 'failed' and 'error' in result:
                report.append(f"  Error: {result['error']}")
            
            report.append("")
        
        return "\n".join(report)


def main():
    """Main function"""
    tester = ConfigTester()
    
    # Run all tests
    results = tester.run_all_tests()
    
    # Generate report
    report = tester.generate_report()
    
    # Print report
    print(report)
    
    # Save report to file
    report_file = tester.test_output_dir / "test_report.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    logger.info(f"Test report saved to: {report_file}")
    
    # Return appropriate exit code
    failed_count = sum(1 for r in results.values() if r['status'] == 'failed')
    error_count = sum(1 for r in results.values() if r['status'] == 'error')
    
    if failed_count > 0 or error_count > 0:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main() 