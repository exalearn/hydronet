from numpy.distutils.core import Extension
import os

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
