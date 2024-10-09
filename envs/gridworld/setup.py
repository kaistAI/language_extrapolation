from setuptools import setup, find_packages

setup(
    name='gridworld',
    version='0.1',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    author='Your Name',
    author_email='youremail@example.com',
    description='gridworld pip package version',
    license='MIT',
    keywords='sample setuptools development',
    url='https://github.com/yourusername/mypackage'
)