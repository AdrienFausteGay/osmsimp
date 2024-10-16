from setuptools import setup, find_packages
from pathlib import Path

setup(
    name='OSMsimp',
    version='1.0.2',
    description='package to simplify networks from OpenstreetMap',
    author='Adrien Fauste-Gay',
    author_email='adrien.fauste-gay@univ-grenoble-alpes.fr',
    #url='https://github.com/yourusername/your-package',
    packages=find_packages(),
    install_requires=Path("requirements.txt").read_text().splitlines(),
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Programming Language :: Python :: 3.9',
    ],
)