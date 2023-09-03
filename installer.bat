@echo off
pip install opencv-python
pip install cvlib
pip install tracker
pip install tensorflow
pip install numpy
cd /d "%~dp0"
python -u main.py
pause