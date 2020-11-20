from setuptools import setup

setup(
    name='tfaffine',
    version='0.1.a',
    packages=['tfaffine'],
    url='https://github.com/balbasty/tfaffine',
    license='',
    author='Yael Balbastre',
    author_email='yael.balbastre@gmail.com',
    description='Affine matrices encoded in their Lie algebra, in tensorflow',
    install_requires=['tensorflow']
)
