#!/bin/bash
export PYTHONIOENCODING=utf-8
export PYTHONUTF8=1

export PYTHONPATH="$PWD:$PYTHONPATH"
python data_provider/fewshot_generator.py