#!/usr/bin/env python3
"""
Single Configuration File Test Script
For quickly testing a specified configuration file.

USAGE:
    python tests/test_single_config.py <config_name>
    python tests/test_single_config.py level1_open

OPTIONS:
    --config-dir: Specify configuration file directory (default: config)
    --output-dir: Specify output directory (default: test_outputs)

REQUIREMENTS:
    - Isaac Sim environment must be properly set up
    - Configuration file should not contain "infer" in the name
    - Test runs for maximum 5 minutes
    - Success rate > 10% is considered a pass

OUTPUT:
    - Test results printed to console in real-time
    - Real-time log saved to <output_dir>/<config_name>_run.log
    - Final output saved to <output_dir>/<config_name>_output.txt
    - Exit code 0 for pass, 1 for fail
"""

import os
import sys
import yaml
import subprocess
import re
from pathlib import Path
import argparse

def modify_config(config_file: Path, temp_config_file: Path):
    """Modify configuration file: set max_episodes to 10, mode to 'collect', and use mock collector"""
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
    
    return config_file.stem

def run_simulation(config_name: str, config_path: str, log_file: Path):
    """Run simulation with real-time logging"""
    cmd = [
        sys.executable, "main.py", 
        "--config-name", config_name,
        "--config-dir", config_path,
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    print(f"Log file: {log_file}")
    
    try:
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
                    
                    # Also print to console
                    print(line.rstrip())
                    
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

def extract_success_rate(output: str) -> float:
    """Extract success rate from output"""
    patterns = [
        r'Success Rate\s*=\s*(\d+)/(\d+)\s*\((\d+\.?\d*)%\)',
        r'Success Rate\s*=\s*(\d+)/(\d+)\s*\((\d+\.?\d*)%\)',
        r'success rate[：:]\s*(\d+\.?\d*)%',
        r'success rate[：:]\s*(\d+\.?\d*)',
        r'Success Rate[：:]\s*(\d+\.?\d*)%',
        r'Success Rate[：:]\s*(\d+\.?\d*)',
        r'Success rate[：:]\s*(\d+\.?\d*)%',
        r'Success rate[：:]\s*(\d+\.?\d*)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, output, re.IGNORECASE)
        if match:
            try:
                if len(match.groups()) == 3:
                    return float(match.group(3))
                else:
                    return float(match.group(1))
            except ValueError:
                continue
    
    # If no explicit success rate found, try to infer from logs
    success_patterns = [
        r'success[：:]\s*(\d+)',
    ]
    
    total_episodes = 10
    success_count = 0
    
    for pattern in success_patterns:
        matches = re.findall(pattern, output, re.IGNORECASE)
        if matches:
            try:
                success_count = max(success_count, int(matches[0]))
            except ValueError:
                continue
    
    if success_count > 0:
        return (success_count / total_episodes) * 100
    
    return 0.0

def main():
    parser = argparse.ArgumentParser(description='Test a single configuration file')
    parser.add_argument('config_name', help='Configuration file name (without .yaml extension)')
    parser.add_argument('--config-dir', default='config', help='Configuration file directory')
    parser.add_argument('--output-dir', default='test_outputs', help='Output directory')
    
    args = parser.parse_args()
    
    # Build file paths
    config_file = Path(args.config_dir) / f"{args.config_name}.yaml"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    if not config_file.exists():
        print(f"Error: Configuration file {config_file} does not exist")
        sys.exit(1)
    
    # Check if it contains "infer"
    if "infer" in args.config_name.lower():
        print(f"Warning: Configuration file {args.config_name} contains 'infer', skipping test")
        sys.exit(0)
    
    print(f"Testing configuration file: {config_file}")
    
    # Create temporary configuration file
    temp_config_file = output_dir / f"temp_{args.config_name}.yaml"
    
    try:
        # Modify configuration
        config_name = modify_config(config_file, temp_config_file)
        
        # Create log file path
        log_file = output_dir / f"{args.config_name}_run.log"
        
        # Run simulation
        success, output = run_simulation(f"temp_{args.config_name}", str(temp_config_file.parent), log_file)
        
        if not success:
            print(f"Simulation run failed: {output}")
            sys.exit(1)
        
        # Extract success rate
        success_rate = extract_success_rate(output)
        
        # Determine if test passed
        test_passed = success_rate > 70
        
        print(f"\nTest Results:")
        print(f"Configuration File: {args.config_name}")
        print(f"Success Rate: {success_rate:.2f}%")
        print(f"Status: {'Passed' if test_passed else 'Failed'}")
        
        # Save output to file
        output_file = output_dir / f"{args.config_name}_output.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(output)
        
        print(f"Output saved to: {output_file}")
        print(f"Real-time log saved to: {log_file}")
        
        if test_passed:
            print("✅ Test passed")
            sys.exit(0)
        else:
            print("❌ Test failed")
            sys.exit(1)
            
    except Exception as e:
        print(f"Error occurred during testing: {str(e)}")
        sys.exit(1)
    finally:
        # Clean up temporary file
        if temp_config_file.exists():
            temp_config_file.unlink()

if __name__ == "__main__":
    main() 