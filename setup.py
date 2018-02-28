#!/usr/bin/env python

import os
import sys


def main():
    from setuptools import setup, find_packages

    if sys.version_info < (3, 5):
        raise SystemError('You need Python version 3.5 or above to use ' +
                          'this package. Current\nversion detected is ' +
                          sys.version.split(' ')[0])

    # from nipype setup.py file
    ldict = locals()
    curr_path = os.path.dirname(__file__)
    ver_file = os.path.join(curr_path, 'snf', 'info.py')
    with open(ver_file) as infofile:
        exec(infofile.read(), globals(), ldict)

    setup(
        name=ldict['NAME'],
        version=ldict['VERSION'],
        description=ldict['DESCRIPTION'],
        maintainer=ldict['MAINTAINER'],
        download_url=ldict['DOWNLOAD_URL'],
        install_requires=ldict['INSTALL_REQUIRES'],
        packages=find_packages(exclude=['snf/tests']),
        package_data=ldict['PACKAGE_DATA'],
        tests_require=ldict['TESTS_REQUIRE'],
        license=ldict['LICENSE'])


if __name__ == '__main__':
    main()
