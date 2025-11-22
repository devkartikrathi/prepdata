import importlib
import inspect
import os
import sys
from pathlib import Path
from typing import List, Type, Dict

from core.interfaces import BaseStep, PreprocessingStep, AnalysisStep

class PluginRegistry:
    def __init__(self):
        self.steps: Dict[str, List[Type[BaseStep]]] = {
            "Preprocessing": [],
            "Analysis": [],
            "Postprocessing": []
        }

    def discover_plugins(self, base_path: str):
        """
        Scans the given directory for python files and loads classes 
        that implement BaseStep.
        """
        # Add the base path to sys.path so we can import modules
        if base_path not in sys.path:
            sys.path.append(base_path)

        # Define directories to scan mapping to groups
        dirs_to_scan = {
            "preprocessing": "Preprocessing",
            "analysis": "Analysis",
            "postprocessing": "Postprocessing"
        }

        for dir_name, group_name in dirs_to_scan.items():
            dir_path = Path(base_path) / dir_name
            if not dir_path.exists():
                continue

            for file_path in dir_path.rglob("*.py"):
                if file_path.name == "__init__.py" or file_path.name.startswith("__"):
                    continue

                # Construct module name
                # e.g. preprocessing.handle_missing_values
                rel_path = file_path.relative_to(base_path)
                module_name = str(rel_path).replace(os.sep, ".")[:-3]

                try:
                    module = importlib.import_module(module_name)
                    
                    # Find classes in the module
                    for name, obj in inspect.getmembers(module):
                        if (inspect.isclass(obj) and 
                            issubclass(obj, BaseStep) and 
                            obj is not BaseStep and 
                            obj is not PreprocessingStep and 
                            obj is not AnalysisStep):
                            
                            # Check if it's already registered to avoid duplicates
                            if obj not in self.steps[group_name]:
                                self.steps[group_name].append(obj)
                except Exception as e:
                    print(f"Failed to load module {module_name}: {e}")

    def get_steps(self, group: str) -> List[Type[BaseStep]]:
        return self.steps.get(group, [])
