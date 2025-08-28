from setuptools import setup, find_packages

setup(
    name="sudoku-solver",
    version="0.1.0",
    description="Simple sudoku grid detection from camera",
    packages=find_packages(),
    install_requires=[
        "opencv-python>=4.8.0",
        "numpy>=1.24.0",
        "Pillow>=10.0.0",
    ],
    python_requires=">=3.8",
)
