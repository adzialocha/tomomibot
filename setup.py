import io

from setuptools import setup
from setuptools import find_packages

from tomomibot import __version__

with io.open('README.md', 'rt', encoding='utf8') as file:
    readme = file.read()

setup(name='tomomibot',
      version=__version__,
      description='Artificial intelligence bot for live voice improvisation',
      long_description=readme,
      author='Andreas Dzialocha',
      author_email='kontakt@andreasdzialocha.com',
      url='https://github.com/adzialocha/tomomibot',
      license='MIT',
      install_requires=['Click>=6.7',
                        'librosa>=0.6.1',
                        'numpy>=1.14.5',
                        'SoundCard==0.1.2',
                        'pysoundfile>=0.9.0',
                        'scikit-learn>=0.19.1'],
      packages=find_packages(),
      dependency_links=[
        'git+https://github.com/bastibe/SoundCard.git#egg=SoundCard-0.1.2',
      ],
      entry_points={
            'console_scripts': {
                'tomomibot = tomomibot.cli:cli',
            }
      })
