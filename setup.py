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
      install_requires=['Click>=6.7'],
      packages=find_packages(),
      entry_points={
            'console_scripts': {
                'tomomibot = tomomibot.cli:cli',
            }
      })
