"""Tensor for torchey."""

import numpy as np

rng = np.random.default_rng(42)  # The answer to life the universe and everything

# Constants for memory calculations
BYTES_PER_FLOAT32 = np.dtype(np.float32).itemsize  # Size of NumPy's float32
KB_TO_BYTES = 1024
MB_TO_BYTES = 1048576

class Tensor:
    """torchey's tensor
    
    Wraps NumPy for numerical computation.
    """

    def __init__(self, data):
        """Create a tensor from input data.
        
        Wraps input data in NumPy array and sets the attributes from NumPy
        array's attributes.

        If a list or tuple of tensors is used to create another tensor, then
        the input tensors' data is extracted as NumPy arrays and stacked using np.stack
        and new tensor is created from it.
        """
        if isinstance(data, (list, tuple)) and len(data) > 0 and isinstance(data[0], Tensor):
            arrays = []
            for t in data:
                arrays.append(t.data)
            data = np.stack(arrays)
        self.data = np.array(data, dtype=np.float32)
        self.shape = self.data.shape
        self.size = self.data.size
        self.dtype = self.data.dtype

    def __repr__(self):
        """String representation of tensor."""
        return f"Tensor(data={self.data}, shape={self.shape})"
    
    def __str__(self):
        """Human readable representation of tensor."""
        return f"Tensor({self.data})"

    def numpy(self):
        """Return the underlying NumPy array."""
        return self.data
    
    def memory_footprint(self):
        """Return the memory usage in bytes."""
        return self.size * BYTES_PER_FLOAT32

    @property
    def ndim(self):
        """Return number of dimensions of the tensor.
        
        @property decorator is used to make it behave like an attribute,
        just how it is done in NumPy and PyTorch. `ndim()` is reserved as
        a top-level method for finding the dimensions of a native Python
        structure.
        """
        return self.data.ndim
    
    def numel(self):
        """Return number of elements."""
        return self.data.size
    
    def contiguous(self):
        """Return a contiguous copy of the tensor data."""
        return np.ascontiguousarray(self.data)

    def view(self, *shape):
        """Alias for reshape()"""
        return self.reshape(*shape)
    
    def masked_fill(self, mask, value):
        """Fill positions where mask is True with value.
        
        Parameters
        ----------
        mask : numpy.array, or Tensor
            Boolean NumPy array or Tensor.
        value : int
            Scalar value to be filled.

        Return
        ------
        result : Tensor
            Masked Tensor
        """
        if isinstance(mask, Tensor):
            mask = mask.data
        mask_array = np.asarray(mask, dtype=bool)
        result = self.data.copy()
        result[mask_array] = value  # Boolean array indexing
        return result

    def __add__(self, other):
        """Add two tensors elementwise with broadcasting support."""
        if isinstance(other, Tensor):
            return Tensor(self.data + other.data)
        else:
            return Tensor(self.data + other)  # NumPy handles broadcasting

    def __sub__(self, other):
        """Subtract two tensors elementwise with broadcasting support."""
        if isinstance(other, Tensor):
            return Tensor(self.data - other.data)
        else:
            return Tensor(self.data - other)
        
    def __mul__(self, other):
        """Multiply two tensors elementwise with broadcasting support."""
        if isinstance(other, Tensor):
            return Tensor(self.data * other.data)
        else:
            return Tensor(self.data * other)
        
    def __truediv__(self, other):
        """Divide two tensors elementwise with broadcasting support."""
        if isinstance(other, Tensor):
            return Tensor(self.data / other.data)
        else:
            return Tensor(self.data / other)
