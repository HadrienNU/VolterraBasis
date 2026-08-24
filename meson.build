project('VolterraBasis', 'c', 'fortran', version: '3.0')

# 1. Find Python and NumPy
py_mod = import('python')
py = py_mod.find_installation(pure: false)
py_dep = py.dependency()

incdir_numpy = run_command(py,
  ['-c', 'import numpy; print(numpy.get_include())'],
  check: true
).stdout().strip()

incdir_f2py = run_command(py,
  ['-c', 'import numpy.f2py; print(numpy.f2py.get_include())'],
  check: true
).stdout().strip()

inc_np = include_directories(incdir_numpy, incdir_f2py)
fortranobject_c = incdir_f2py / 'fortranobject.c'

# 2. Tell Meson to run F2PY on your Fortran file(s)
# Replace 'your_fortran_file.f90' and 'your_module_name' with your actual files
f2py_target = custom_target('f2py_wrapper',
  input: ['VolterraBasis/fkernel.f90'], # Your Fortran source
  output: ['fkernelmodule.c', 'fkernel-f2pywrappers2.f90'],
  command: [
    py, '-m', 'numpy.f2py', 
    '@INPUT@', 
    '-m', 'fkernel', 
    '--lower' 
  ]
)

lapack_dep = dependency('lapack')

# 3. Compile the Extension Module
py.extension_module('fkernel',
  'VolterraBasis/fkernel.f90', # The original fortran file
  f2py_target,                 # The generated C wrappers
  fortranobject_c,  	       # The NumPy/Fortran bridge C file
  include_directories: inc_np,
  dependencies: [py_dep, lapack_dep],
  install: true,
  subdir: 'VolterraBasis'      # Installs it inside your python package folder
)

# 4. Install your pure Python code
py.install_sources(
  'VolterraBasis/__init__.py',
  # add your other .py files here
  subdir: 'VolterraBasis'
)

install_subdir(
  'VolterraBasis', 
  install_dir: py.get_install_dir()
)
