from setuptools import setup, find_packages

setup(
    name='dl_projects',
    version='1.0.0',
    packages=find_packages(),
    install_requires=[
        'tensorflow>=2.11.0,<2.12.0',  # Pin to a specific version that works
        'streamlit>=1.31.0',
        'numpy>=1.23.5',
        'pandas>=2.2.0'
    ],
    python_requires='>=3.10,<3.11',  # Explicitly specify Python version
    classifiers=[
        'Programming Language :: Python :: 3.10',
        'Framework :: TensorFlow',
        'Framework :: Streamlit'
    ]
)
