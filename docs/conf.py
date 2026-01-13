import os
import sys
sys.path.insert(0, os.path.abspath('../src'))
project = 'AEGIS-O'
copyright = '2025, H M Shujaat Zaheer'
author = 'H M Shujaat Zaheer'
release = '0.1.0'
extensions = ['sphinx.ext.autodoc', 'sphinx.ext.napoleon', 'sphinx.ext.viewcode']
templates_path = ['_templates']
exclude_patterns = ['_build']
html_theme = 'alabaster'
html_static_path = ['_static']
