from setuptools import setup, find_packages

setup(
    name='netsimp',
    version='1.0.0',
    description='package to simplify networks from OpenstreetMap',
    author='Adrien Fauste-Gay',
    author_email='adrien.faustegay@gmail.com',
    #url='https://github.com/yourusername/your-package',
    packages=find_packages(),
    install_requires=[],
    classifiers=[
        'Development Status :: Production',
        'Intended Audience :: Network Scientists',
        'License :: OSI Approved :: CC License',
        'Programming Language :: Python :: 3.9',
    ],
)