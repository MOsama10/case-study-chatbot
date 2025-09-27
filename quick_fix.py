#!/usr/bin/env python3
"""
Quick fix script to update imports in your source files.
Run this in your project root directory: python quick_fix.py
"""

import os
import re
from pathlib import Path

def fix_imports_in_file(file_path: Path):
    """Fix relative imports in a Python file."""
    print(f"Fixing imports in {file_path}")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Replace relative imports with absolute imports
        patterns = [
            (r'from \.config import', 'from src.config import'),
            (r'from \.data_loader import', 'from src.data_loader import'), 
            (r'from \.embeddings import', 'from src.embeddings import'),
            (r'from \.knowledge_graph import', 'from src.knowledge_graph import'),
            (r'from \.retriever import', 'from src.retriever import'),
            (r'from \.agent import', 'from src.agent import'),
            (r'from \.chatbot import', 'from src.chatbot import'),
            (r'from \.ui import', 'from src.ui import'),
            (r'from \.direct_answer_handler import', 'from src.direct_answer_handler import'),
            (r'from \.response_formatter import', 'from src.response_formatter import'),
            (r'from \.professional_prompts import', 'from src.professional_prompts import'),
        ]
        
        modified = False
        for pattern, replacement in patterns:
            if re.search(pattern, content):
                content = re.sub(pattern, replacement, content)
                modified = True
                print(f"  Fixed: {pattern}")
        
        if modified:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"  ✅ Updated {file_path}")
        else:
            print(f"  ✅ No changes needed in {file_path}")
            
    except Exception as e:
        print(f"  ❌ Error fixing {file_path}: {e}")

def main():
    """Fix imports in all Python files in src directory."""
    print("🔧 Fixing import statements...")
    
    project_root = Path.cwd()
    src_dir = project_root / "src"
    
    if not src_dir.exists():
        print("❌ src/ directory not found!")
        return
    
    # Find all Python files in src
    python_files = list(src_dir.glob("*.py"))
    
    print(f"Found {len(python_files)} Python files to check...")
    
    for py_file in python_files:
        if py_file.name != "__init__.py":  # Skip __init__.py
            fix_imports_in_file(py_file)
    
    # Make sure __init__.py is empty
    init_file = src_dir / "__init__.py"
    if init_file.exists():
        with open(init_file, 'w') as f:
            f.write('"""Package initialization."""\n')
        print("✅ Cleaned __init__.py")
    
    print("\n🎉 Import fixing complete!")
    print("Now run: python main.py")

if __name__ == "__main__":
    main()
    