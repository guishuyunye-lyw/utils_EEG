from setuptools import setup, find_packages

setup(
    name="EEG_utils_lyw",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        # List your package dependencies here
        # e.g., 'requests>=2.23.0'
        'numpy',
        'mne',
        'scipy',
        'pandas',
        'matplotlib',
        'seaborn',
        'statsmodels',
        'nilearn',
        'numba',
        'nilearn',
        'scikit-learn',
        'nilearn',
        'neurora'
    ],
    author="lyw",
    author_email="guishuyunye@gmail.com",
    description="A brief description of your package",
    license="MIT",
    keywords="EEG EEG-analysis decoding RSA",
)
