
from setuptools import setup, find_packages

setup(
    name='vtas-metric',
    version='1.3.0',
    description='VTAS: Visual-Text Alignment Score for Evaluating Vision-Language Models',
    long_description=open('README.md', encoding='utf-8').read(),
    long_description_content_type='text/markdown',
    author='Ritanshu Prasad',
    author_email='ritanshu.prasad@example.com',
    url='https://github.com/Ritanshu-Prasad/VTAS_Evaluation_Metric',
    package_dir={'': 'src'},
    packages=find_packages(where='src'),
    install_requires=[
        'torch',
        'transformers',
        'spacy',
        'sentence-transformers',
        'Pillow'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',
)

