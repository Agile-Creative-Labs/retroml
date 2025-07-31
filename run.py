#!/usr/bin/env python3
"""
RetroML Runner Script
====================
Convenience script to run RetroML with different configurations
"""

import sys
import os
import subprocess
from pathlib import Path

def main():
    print("🎮 RETROML RUNNER 🎮")
    print("=" * 40)
    
    # Check if config files exist
    configs = list(Path('configs').glob('*.json'))
    
    if not configs:
        print("❌ No configuration files found!")
        print("Run this script from the RetroML project directory.")
        return
    
    print("Available configurations:")
    for i, config in enumerate(configs):
        print(f"  {i + 1}. {config.name}")
    
    print(f"  {len(configs) + 1}. Run tests")
    print(f"  {len(configs) + 2}. Exit")
    
    try:
        choice = int(input("\nSelect an option: ")) - 1
        
        if choice < len(configs):
            config_path = configs[choice]
            print(f"\n🚀 Running RetroML with {config_path.name}...")
            subprocess.run([sys.executable, 'retroml.py', str(config_path)])
        elif choice == len(configs):
            print("\n🧪 Running tests...")
            subprocess.run([sys.executable, '-m', 'pytest', 'tests/', '-v'])
        else:
            print("👋 Goodbye!")
            
    except (ValueError, IndexError, KeyboardInterrupt):
        print("\n👋 Goodbye!")

if __name__ == '__main__':
    main()
