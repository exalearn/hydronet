from numpy.distutils.core import Extension
from urllib.request import urlretrieve
from zipfile import ZipFile
from glob import glob
import os

# Download TTM Fortran code if need be
_ttm_url = 'https://sites.uw.edu/wdbase/files/2019/01/pot_ttm-1p9hi7d.zip'
if not os.path.isdir('pot_ttm'):
    urlretrieve(_ttm_url, 'ttm.zip')
    with ZipFile('ttm.zip') as fp:
        fp.extractall(".")

# Write out the extension
ext = Extension(name='ttm.flib',
                sources=[os.path.join('ttm', 'ttm_from_f2py.f90')],
                extra_f90_compile_args=['-O2', '-fPIC'],
                include_dirs=['pot_ttm'],
                library_dirs=['pot_ttm'],
                extra_objects=['./pot_ttm/pot_ttm.a'],
                extra_link_args=['-fPIC', '-shared'])

if __name__ == "__main__":
    from numpy.distutils.core import setup
    setup(name='ttm', packages=['ttm'], ext_modules=[ext])
