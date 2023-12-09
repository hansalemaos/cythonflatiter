# Locates values in NumPy Arrays with Cython


## pip install cythonflatiter

### Tested against Windows / Python 3.11 / Anaconda


## Cython (and a C/C++ compiler) must be installed

FlatIterArray is a utility class for efficiently searching multi-dimensional array data.

```python
 |  __init__(self, a, dtype=<class 'numpy.int64'>, unordered=True)
 |      Initializes a FlatIterArray instance.
 |      
 |      Parameters:
 |      - a (numpy.ndarray): The input array.
 |      - dtype (numpy.dtype, optional): The data type of the index array that will be created. Defaults to np.int64. (It's better not to change that, because it corresponds to cython.Py_ssize_t)
 |      - unordered (bool, optional): Flag indicating whether to use unordered iterations. Defaults to True. (index array will be created using multiprocessing)
 |  
 |  get_flat_pointer_array_from_orig_data(self)
 |      Returns a flat pointer array from the original data.
 |      If you change data here, it changes also in the original array
 |      
 |      Returns:
 |      - numpy.ndarray: Flat pointer array.
 |  
 |  search_multiple_values_in_array(self, values)
 |      Searches for multiple values in the array and returns indices and values.
 |      
 |      Parameters:
 |      - values (list or numpy.ndarray): List of values to search for.
 |      
 |      Returns:
 |      - tuple: Array of indices and array of found values.
 |  
 |  search_single_value_in_array(self, value)
 |      Searches for a single value in the array and returns indices.
 |      
 |      Parameters:
 |      - value: The value to search for.
 |      
 |      Returns:
 |      - numpy.ndarray: Array of indices.
 |  
 |  sequence_is_in_dimension(self, seq, last_dim)
 |      Checks if a sequence is in a dimension
 |      
 |      Parameters:
 |      - seq (list or numpy.ndarray): List of values representing the sequence.
 |      - last_dim: The dimension to search in.
 |      
 |      Returns:
 |      - numpy.ndarray: Array of indices.
 |  
 |  update_iterarray(self, dtype=<class 'numpy.int64'>, unordered=True)
 |      Updates the iterray attribute with new parameters.
 |      
 |      Parameters:
 |      - dtype (numpy.dtype, optional): The data type of the index array that will be created. Defaults to np.int64.
 |      - unordered (bool, optional): Flag indicating whether to use unordered iterations. Defaults to True. (index array will be created using multiprocessing)
 |  
 |  value_is_in_dimension(self, value, last_dim)
 |      Checks if a value is in a dimension
 |      
 |      Parameters:
 |      - value: The value to search for.
 |      - last_dim: The dimension to search in.
 |      
 |      Returns:
 |      - numpy.ndarray: Array of indices.
 
 import numpy as np
import cv2
from cythonflatiter import FlatIterArray

data = cv2.imread(r"C:\Users\hansc\Desktop\2023-08-29xx16_07_30-Window.png")
f = FlatIterArray(data, dtype=np.int64, unordered=True)
results255inarray = f.search_single_value_in_array(255)
# results255inarray
# Out[6]:
# array([[195330,     34,      0,      0],
#        [201075,     35,      0,      0],
#        [206820,     36,      0,      0],
#        ...,
#        [488324,     84,   1914,      2],
#        [494069,     85,   1914,      2],
#        [499814,     86,   1914,      2]], dtype=int64)
# print(data[84,1914,2])
# print(data[34,0,0])
# 255
# 255
indices, found_values = f.search_multiple_values_in_array(values=[255, 11, 0])
# indices,found_values
# Out[12]:
# (array([[195330,     34,      0,      0],
#         [201075,     35,      0,      0],
#         [206820,     36,      0,      0],
#         ...,
#         [488324,     84,   1914,      2],
#         [494069,     85,   1914,      2],
#         [499814,     86,   1914,      2]], dtype=int64),
#  array([255, 255, 255, ..., 255, 255, 255], dtype=uint8))
concat_values = np.hstack([indices, found_values.reshape((-1, 1))])
# print(concat_values)
# [[195330     34      0      0    255]
#  [201075     35      0      0    255]
#  [206820     36      0      0    255]
#  ...
#  [488324     84   1914      2    255]
#  [494069     85   1914      2    255]
#  [499814     86   1914      2    255]]
# print(data[85,1914,2])
# print(data[36,0,0])
lastdimwith255 = f.value_is_in_dimension(255, 3)
# lastdimwith255
# Out[31]:
# array([[195330,     34,      0],
#        [201075,     35,      0],
#        [206820,     36,      0],
#        ...,
#        [488322,     84,   1914],
#        [494067,     85,   1914],
#        [499812,     86,   1914]], dtype=int64)
# print(data[86,1914])
# print(data[34,0])
# [255 255 255]
# [255 255 255]
penultimatedimwith255255 = f.sequence_is_in_dimension([255, 255, 255], 2)
# penultimatedimwith255255
# Out[38]:
# array([[      0,       0],
#        [   5745,       1],
#        [  11490,       2],
#        ...,
#        [5538180,     964],
#        [5543925,     965],
#        [5549670,     966]], dtype=int64)
# data[964]
# Out[40]:
# array([[255, 255, 255],
#        [255, 255, 255],
#        [255, 255, 255],
#        ...,
#        [ 43,  43,  43],
#        [ 43,  43,  43],
#        [ 43,  43,  43]], dtype=uint8)

ultimatedimwith255255255 = f.sequence_is_in_dimension([255, 255, 255], 3)
# ultimatedimwith255255255
# Out[42]:
# array([[195330,     34,      0],
#        [201075,     35,      0],
#        [206820,     36,      0],
#        ...,
#        [488322,     84,   1914],
#        [494067,     85,   1914],
#        [499812,     86,   1914]], dtype=int64)

# [750 433   0]
# [751 433   0]
# [752 433   0]
# [753 433   0]
# [754 433   0]
# [755 433   0]
# [756 433   0]
# [757 433   0]
# [758 433   0]

```