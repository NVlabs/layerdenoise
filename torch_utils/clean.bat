@echo off
rmdir /q /s build > nul 2>&1
rmdir /q /s dist > nul 2>&1
rmdir /q /s torch_utils.egg-info > nul 2>&1
rmdir /q /s tests\__pycache__ > nul 2>&1

