from setuptools import setup
import unittest


def my_test_suite():
    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover('tests', pattern='test_*.py')
    return test_suite


def readme():
    with open('README.md') as f:
        return f.read()


setup(name='goal_rec_utils',
      version='0.1',
      description='A set of goal recognition utility functions',
      long_description=readme(),
      classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.7',
        'Topic :: Plan :: Goal Recognition',
      ],
      keywords='goal recognition unibs ',
      url='',
      author='Mattia Chiari',
      author_email='m.chiari017@unibs.it',
      license='MIT',
      packages=['goal_rec_utils'],
      install_requires=[
        #   'tensorflow==2.4.0';,
        #   'keras==2.4.3',
          'numpy>=1.19.2'
      ],
      test_suite='tests',
      include_package_data=True,
      zip_safe=False)