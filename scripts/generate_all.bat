@echo off
if not exist app.py (
    echo Please run this script from the project root directory.
    exit /b 1
)

if not exist output mkdir output

call scripts\generate_imagenet.bat
call scripts\generate_coco.bat
call scripts\generate_openimages.bat

echo All hierarchies generated in output/
