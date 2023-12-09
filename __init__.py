import os
import subprocess
import sys
import numpy as np
from cythonanyarray import create_product_ordered, create_product_unordered, get_iterarray, get_pointer_array, \
    get_iterarray_shape


def _dummyimport():
    import Cython


try:
    from .cythoniterarray import findvalue, findmultivalues, valuesareinlastdim, valueisinlastdim

except Exception as e:
    cstring = r"""# distutils: language=c
# distutils: extra_compile_args=/openmp /O2
# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION
# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: language_level=3
# cython: initializedcheck=False

from cython.parallel cimport prange
cimport cython
import numpy as np
cimport numpy as np
import cython
np.import_array()
ctypedef fused real:
    cython.char
    cython.schar
    cython.uchar
    cython.short
    cython.ushort
    cython.int
    cython.uint
    cython.long
    cython.ulong
    cython.longlong
    cython.ulonglong
    cython.size_t
    cython.Py_ssize_t
    cython.float
    cython.double
    cython.longdouble
    cython.floatcomplex
    cython.doublecomplex
    cython.longdoublecomplex
    cython.Py_hash_t
    cython.Py_UCS4
  

cpdef void findvalue(real[:] data,Py_ssize_t[:,:] iterray,Py_ssize_t[:] resultarray, real[:] val ):
    cdef cython.Py_ssize_t lenofarray = iterray.shape[0]
    cdef cython.Py_ssize_t q,abs_index
    cdef bint isright
    for q in prange(lenofarray,nogil=True):
        abs_index=iterray[q][0]
        isright = data[abs_index]==val[0]
        if isright:
            resultarray[q]=abs_index+1

cpdef void findmultivalues(real[:] data,Py_ssize_t[:,:] iterray,Py_ssize_t[:] resultarray, real[:] val,Py_ssize_t[:] resultarrayvals ):
    cdef cython.Py_ssize_t lenofarray = iterray.shape[0]
    cdef cython.Py_ssize_t q,abs_index,v
    cdef cython.Py_ssize_t vallen=val.shape[0]

    cdef bint isright
    for q in prange(lenofarray,nogil=True):
        abs_index=iterray[q][0]
        for v in range(vallen):
            isright = data[abs_index]==val[v]
            if isright:
                resultarray[q]=abs_index+1
                resultarrayvals[q]=v+1
                break
cpdef void valueisinlastdim(real[:] data,Py_ssize_t[:,:] iterray,Py_ssize_t[:] resultarray, real[:] val, Py_ssize_t steps):
    cdef cython.Py_ssize_t lenofarray = iterray.shape[0]
    cdef cython.Py_ssize_t q
    cdef Py_ssize_t ste,abs_index
    for q in prange(lenofarray,nogil=True):
        abs_index=iterray[q][0]
        for ste in range(steps):
            if data[abs_index+ste:abs_index+ste+1][0]==val[0]:
                resultarray[abs_index]=abs_index+ste+1
                break

cpdef void valuesareinlastdim(real[:] data,Py_ssize_t[:,:] iterray,Py_ssize_t[:] resultarray, real[:] val, Py_ssize_t steps):
    cdef cython.Py_ssize_t lenofarray = iterray.shape[0]
    cdef cython.Py_ssize_t q
    cdef Py_ssize_t ste,abs_index,v
    cdef Py_ssize_t valuearrayloop=len(val)
    cdef Py_ssize_t datalen=len(data)
    cdef Py_ssize_t indexcheck
    for q in prange(lenofarray,nogil=True):
        abs_index=iterray[q][0]
        for ste in range(steps):
            for v in range(valuearrayloop):
                if abs_index+ste+v+1>=datalen:
                    break
                if not data[abs_index+ste+v:abs_index+ste+v+1][0]==val[v]:
                    break
            else:
                resultarray[abs_index]=abs_index+ste+1
                #break
                
"""
    pyxfile = f"cythoniterarray.pyx"
    pyxfilesetup = f"cythoniterarraycompiled_setup.py"

    dirname = os.path.abspath(os.path.dirname(__file__))
    pyxfile_complete_path = os.path.join(dirname, pyxfile)
    pyxfile_setup_complete_path = os.path.join(dirname, pyxfilesetup)

    if os.path.exists(pyxfile_complete_path):
        os.remove(pyxfile_complete_path)
    if os.path.exists(pyxfile_setup_complete_path):
        os.remove(pyxfile_setup_complete_path)
    with open(pyxfile_complete_path, mode="w", encoding="utf-8") as f:
        f.write(cstring)
    numpyincludefolder = np.get_include()
    compilefile = (
            """
	from setuptools import Extension, setup
	from Cython.Build import cythonize
	ext_modules = Extension(**{'py_limited_api': False, 'name': 'cythoniterarray', 'sources': ['cythoniterarray.pyx'], 'include_dirs': [\'"""
            + numpyincludefolder
            + """\'], 'define_macros': [], 'undef_macros': [], 'library_dirs': [], 'libraries': [], 'runtime_library_dirs': [], 'extra_objects': [], 'extra_compile_args': [], 'extra_link_args': [], 'export_symbols': [], 'swig_opts': [], 'depends': [], 'language': None, 'optional': None})

	setup(
		name='cythoniterarray',
		ext_modules=cythonize(ext_modules),
	)
			"""
    )
    with open(pyxfile_setup_complete_path, mode="w", encoding="utf-8") as f:
        f.write(
            "\n".join(
                [x.lstrip().replace(os.sep, "/") for x in compilefile.splitlines()]
            )
        )
    subprocess.run(
        [sys.executable, pyxfile_setup_complete_path, "build_ext", "--inplace"],
        cwd=dirname,
        shell=True,
        env=os.environ.copy(),
    )
    try:
        from .cythoniterarray import findvalue, findmultivalues, valuesareinlastdim, valueisinlastdim


    except Exception as fe:
        sys.stderr.write(f'{fe}')
        sys.stderr.flush()


def search_single_value_in_array(data, value, unordered=True, iterray=()):
    if isinstance(iterray, tuple):
        iterray = get_iterarray(data, unordered=unordered)
    val = np.array([value], dtype=data.dtype)
    pointerdata = get_pointer_array(data)
    resultarray = np.zeros(len(pointerdata), dtype=np.int64)
    findvalue(pointerdata, iterray, resultarray, val)
    return iterray[np.where((resultarray - 1) >= 0)]


def search_multiple_values_in_array(data, values, unordered=True, iterray=()):
    if isinstance(iterray, tuple):
        iterray = get_iterarray(data, unordered=unordered)
    pointerdata = get_pointer_array(data)
    val = np.array(values, dtype=pointerdata.dtype)
    resultarrayvals = np.zeros(len(pointerdata), dtype=np.int64)
    resultarray = np.zeros(len(pointerdata), dtype=np.int64)
    findmultivalues(pointerdata, iterray, resultarray, val, resultarrayvals)
    nonz = np.where((resultarray - 1) >= 0)
    return iterray[nonz], val[resultarrayvals[nonz] - 1]


def value_is_in_dimension(data, value, last_dim, dtype=np.int64, unordered=True, iterray=()):
    if isinstance(iterray, tuple):
        iterray = get_iterarray(data, dtype=dtype, unordered=unordered)

    rax = get_iterarray_shape(iterray, last_dim)

    steps = np.product(data.shape[last_dim - 1:])
    pointerdata = get_pointer_array(data)
    resultarray = np.zeros(len(pointerdata), dtype=np.int64)
    val = np.array([value], dtype=data.dtype)
    valueisinlastdim(pointerdata, rax, resultarray, val, steps)
    return rax[np.isin(rax[..., 0], np.nonzero(resultarray)[0])]


def sequence_is_in_dimension(data, seq, last_dim, dtype=np.int64, unordered=True, iterray=()):
    if isinstance(iterray, tuple):
        iterray = get_iterarray(data, dtype=dtype, unordered=unordered)

    rax = get_iterarray_shape(iterray, last_dim)
    steps = np.product(data.shape[last_dim - 1:])
    pointerdata = get_pointer_array(data)
    resultarray = np.zeros(len(pointerdata), dtype=np.int64)
    val = np.array(seq, dtype=data.dtype)
    valuesareinlastdim(pointerdata, rax, resultarray, val, steps)
    return rax[np.isin(rax[..., 0], np.nonzero(resultarray)[0])]


class FlatIterArray:
    r"""
    FlatIterArray is a utility class for efficiently searching and manipulating multi-dimensional array data.

    Parameters:
    - a (numpy.ndarray): The input array.
    - dtype (numpy.dtype, optional): The data type of the array. Defaults to np.int64.
    - unordered (bool, optional): Flag indicating whether to use unordered iterations. Defaults to True.

    Attributes:
    - a (numpy.ndarray): The input array.
    - dtype (numpy.dtype): The data type of the array.
    - unordered (bool): Flag indicating whether to use unordered (multi processing) iterations.
    - iterray (numpy.ndarray): The flattened iteration array.

    Methods:
    - update_iterarray(dtype=np.int64, unordered=True): Updates the iterray attribute with new parameters.
    - get_flat_pointer_array_from_orig_data(): Returns a flat pointer array from the original data.
    - search_single_value_in_array(value): Searches for a single value in the array and returns indices.
    - search_multiple_values_in_array(values): Searches for multiple values in the array and returns indices and values.
    - value_is_in_dimension(value, last_dim): Finds occurrences of a value in the specified dimension.
    - sequence_is_in_dimension(seq, last_dim): Finds sequences in the specified dimension.

    """

    def __init__(self, a, dtype=np.int64, unordered=True):
        r"""
        Initializes a FlatIterArray instance.

        Parameters:
        - a (numpy.ndarray): The input array.
        - dtype (numpy.dtype, optional): The data type of the index array that will be created. Defaults to np.int64. (It's better not to change that, because it corresponds to cython.Py_ssize_t)
        - unordered (bool, optional): Flag indicating whether to use unordered iterations. Defaults to True. (index array will be created using multiprocessing)
        """
        self.a = a
        self.dtype = dtype
        self.unordered = unordered
        self.iterray = get_iterarray(self.a, dtype=self.dtype, unordered=self.unordered)

    def update_iterarray(self, dtype=np.int64, unordered=True):
        r"""
        Updates the iterray attribute with new parameters.

        Parameters:
        - dtype (numpy.dtype, optional): The data type of the index array that will be created. Defaults to np.int64.
        - unordered (bool, optional): Flag indicating whether to use unordered iterations. Defaults to True. (index array will be created using multiprocessing)
        """
        self.iterray = get_iterarray(self.a, dtype=dtype, unordered=unordered)

    def get_flat_pointer_array_from_orig_data(self, ):
        r"""
        Returns a flat pointer array from the original data.
        If you change data here, it changes also in the original array

        Returns:
        - numpy.ndarray: Flat pointer array.
        """
        return get_pointer_array(self.a)

    def search_single_value_in_array(self, value):
        r"""
        Searches for a single value in the array and returns indices.

        Parameters:
        - value: The value to search for.

        Returns:
        - numpy.ndarray: Array of indices.
        """
        return search_single_value_in_array(data=self.a, value=value, unordered=self.unordered, iterray=self.iterray)

    def search_multiple_values_in_array(self, values, ):
        r"""
        Searches for multiple values in the array and returns indices and values.

        Parameters:
        - values (list or numpy.ndarray): List of values to search for.

        Returns:
        - tuple: Array of indices and array of found values.
        """
        return search_multiple_values_in_array(data=self.a, values=values, unordered=self.unordered,
                                               iterray=self.iterray)

    def value_is_in_dimension(self, value, last_dim, ):
        r"""
        Checks if a value is in a dimension

        Parameters:
        - value: The value to search for.
        - last_dim: The dimension to search in.

        Returns:
        - numpy.ndarray: Array of indices.
        """
        return value_is_in_dimension(data=self.a, value=value, last_dim=last_dim, dtype=self.dtype,
                                     unordered=self.unordered,
                                     iterray=self.iterray)

    def sequence_is_in_dimension(self, seq, last_dim, ):
        r"""
        Checks if a sequence is in a dimension

        Parameters:
        - seq (list or numpy.ndarray): List of values representing the sequence.
        - last_dim: The dimension to search in.

        Returns:
        - numpy.ndarray: Array of indices.
        """
        return sequence_is_in_dimension(data=self.a, seq=seq, last_dim=last_dim, dtype=np.int64,
                                        unordered=self.unordered,
                                        iterray=self.iterray)
