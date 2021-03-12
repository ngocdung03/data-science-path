##### NumPy
- Documentation: https://docs.scipy.org/doc/numpy-1.14.0/reference/arrays.ndarray.html#calculation
- The NumPy library takes advantage of a processor feature called Single Instruction Multiple Data (SIMD) to process data faster. SIMD allows a processor to perform the same operation, on multiple data points, in a single processor cycle
- The concept of replacing for loops with operations applied to multiple data points at once is called *vectorization* and ndarrays make vectorization possible.
- Tuples are very similar to Python lists, but can't be modified.
- Functions act as stand alone segments of code that usually take an input, perform some processing, and return some output
- In contrast, methods are special functions that belong to a specific type of object.
- In NumPy, sometimes there are operations that are implemented as both methods and functions:
```
np.min(trip_mph)
trip_mph.min()
```
- To remember the right terminology, anything that starts with np (e.g. np.mean()) is a function and anything expressed with an object (or variable) name first (e.g. trip_mph.mean()) is a method. When both exist, it's up to you to decide which to use, but it's much more common to use the method approach.
- Read file into NumPy ndarrays: `np.genfromtxt(filepath, delimiter=None, skip_header=0)`
- NumPy ndarrays can contain only one datatype.
- ndarray.dtype: see the internal datatype that has been used. (eg float64)
- NaN: similar to Python's None constant

##### Boolean indexing with NumPy
- Operation between a ndarray and a single value results in a new ndarray: `print(np.array([2,4,6,8]) + 10)   #[12 14 16 18]`
- To index using our new boolean array, we simply insert it in the square brackets
- When working with 2D ndarrays,  boolean array must have the same length as the dimension you're indexing
- Modify values within an ndarray: 
```
np.genfromtxt(filename, delimiter=None)
ndarray[location_of_values] = new_value`
```
- Shortcut in one line
```
array[array[:, column_for_comparison] == value_for_comparison, column_for_assignment] = new_value
```
