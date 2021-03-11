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