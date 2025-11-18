import json
import numpy as np
from ast import literal_eval
from typedmapping import TypedMapping, TypedMappingMeta
from formatting import pretty_json_string

class POPxfPolynomialMeta(TypedMappingMeta):
    """
    Base metaclass for the `POPxfPolynomial` class.
    Implements additional custom exception classes on top of TypedMappingMeta.
    """
    def __init__(self, classname, baseclasses, attrs):

        # attrs['key_order_error'] = type(
        self.key_order_error = type(
          classname+'KeyOrderError', 
          (attrs['base_error'],),
          {}
        )

        self.length_error = type(
          classname+'LengthError', 
          (attrs['base_error'],),
          {}
        )

        self.shape_error = type(
          classname+'ShapeError', 
          (attrs['base_error'],),
          {}
        )

        super().__init__(classname, baseclasses, attrs)


class POPxfPolynomial(TypedMapping, metaclass=POPxfPolynomialMeta):
    """
    Class to store polynomial data in POPxf JSON files.

    This class represents polynomial expressions where coefficients are stored as 
    numpy arrays. Keys represent polynomial terms (monomials of parameters),
    and values are the corresponding coefficient arrays. The class supports 
    polynomials of arbitrary degree with real and imaginary components.

    Parameters
    ----------
    data : dict
        Dictionary containing polynomial data. Keys must be tuples representing
        polynomial terms, and values must be float arrays or lists convertible
        to numpy arrays.
    degree : int, optional
        Degree of the polynomial. Default is 2.
    length : int, optional
        Length of coefficient arrays. If not provided, inferred from the first
        entry in `data`.

    Attributes
    ----------
    degree : int
        Degree of the polynomial.
    length : int
        Length of the coefficient arrays.
    shape : tuple
        Shape of coefficient arrays (always 1D).
    parameters : tuple
        Tuple of parameter names that appear in the polynomial, sorted alphabetically.
    key_types : tuple
        Allowed types for keys (tuple only).
    value_types : tuple
        Allowed types for values (float and numpy.ndarray).

    Raises
    ------
    POPxfPolynomialInitError
        If initialization fails due to invalid length, shape, or data format.
    POPxfPolynomialKeyError
        If keys are improperly formatted or violate ordering constraints.
    POPxfPolynomialValueError
        If values cannot be converted to proper numpy arrays.
    POPxfPolynomialLengthError
        If array lengths don't match the expected length.
    POPxfPolynomialShapeError
        If arrays have incorrect dimensionality.

    Notes
    -----
    Keys must be tuples where:
    - The first n elements are parameter names (strings)
    - The last element (optional) is a real/imaginary specifier string of 'R' and 'I' characters
    - Parameters must be ordered alphabetically
    - Empty strings represent the constant term
    
    Examples
    --------
    >>> data = {
    ...     ('',''): [1.0, 2.0, 3.0],
    ...     ('C1', ''): [0.1, 0.2, 0.3],
    ...     ('C1', 'C2'): [0.01, 0.02, 0.03]
    ... }
    >>> poly = POPxfPolynomial(data, degree=2, length=3)
    >>> print(poly.parameters)
    ('C1', 'C2')
    
    See Also
    --------
    POPxfPolynomialUncertainty : Subclass for handling polynomial uncertainties
    TypedMapping : Base class providing typed dictionary functionality
    """

    # allowed key and value types
    key_types = (tuple,)
    value_types = (float, np.ndarray)

    def __init__(self, data, degree=2, length=None):
        
        # polynomial degree
        self.degree = degree

        # specify length if given
        if length is not None:
            if isinstance(length, int):
                self.length = length
            else:
                raise self.init_error(
                  f'Invalid length "{length}": must be an int.'
                )
        # otherwise infer shape from first datum
        elif data:
            _, first_val = next(iter(data.items()))
            if isinstance(first_val, list):
                try:
                    parsed_val = np.asarray(first_val)
                except Exception as e:
                    raise self.init_error(
                      f'Could not convert "{first_val}" to numpy.ndarray when '
                      f'inferring length.'
                    ) from e
                self.shape = parsed_val.shape

            elif isinstance(first_val, np.ndarray):
                self.shape = first_val.shape
                
            else:
                raise self.init_error(
                  f'Could not convert infer length from "{first_val}".'
                ) 
            
            if len(self.shape)!=1:
                # forbid ndarrays with more than one dimension
                raise self.init_error(
                  f'Encoutered bad shape when inferring length from '
                  f'"{first_val}". Only one-dimensional arrays are allowed as '
                  'values.'
                )
            else:
                self.length = self.shape[0] 
        else:
            raise self.init_error(
              'Length must be specified if data is empty.'
            )

        super().__init__(data)

        # store the raw input data
        self._raw_data = data

        # tuple of parameters on which the polynomial depends
        self.parameters = tuple(sorted(list(set([ 
          x for y in
          [list(i) if len(i)==self.degree else list(i[:-1]) for i in self.keys()]
          for x in y if x 
        ]))))
    
    @staticmethod
    def is_RI(element):
        """
        Check if element is a real/imaginary specifier string.

        A valid real/imaginary specifier string consists only of 'R' and 'I' 
        characters, where 'R' denotes real component and 'I' denotes imaginary 
        component.

        Parameters
        ----------
        element : str or None
            String to check for real/imaginary specifier format.

        Returns
        -------
        bool
            True if element is a non-empty string containing only 'R' and 'I' 
            characters, False otherwise.
        """
        return ( element and all([x in ('R', 'I') for x in element]) )

    def _parse_key(self, key):
        """
        Parse key, checking it is valid and return as a tuple.

        `key` must be a tuple of strings. The first n elements represent  
        polynomial parameters. Optionally the last element can be a length-n 
        string where each element can only be 'R' or 'I', stating whether the
        real or imaginary component of each coefficient is being referenced. 
        The constant term in the polynomial is specified by a tuple of empty 
        strings. Elements should be ordered alphabetically by convention. 
        If the real/imaginary specifier is not present, parameters are assumed 
        to be real.

        If the key is a string, it is converted to a tuple via `ast.literal_eval()`.

        Parameters
        ----------
        key : tuple
            Key to parse, checking validity as described above.
        
        Returns
        -------
        tuple
            Parsed key as a tuple of three strings.
        
        Raises
        ------
        self.key_error
            If the key is not a valid tuple as described above.
        self.key_order_error
            If the parameters in the key are not ordered alphabetically.

        """
        
        tuplekey = literal_eval(key) if isinstance(key, str) else key

        # Handle optional real/imaginary specifier
        if len(tuplekey) == self.degree:
            params, RI_str = tuplekey, None
        elif len(tuplekey) == self.degree + 1:
            params, RI_str = tuplekey[:-1], tuplekey[-1]
        else:
            raise self.key_error(
              f'Invalid key {key}: number of elements must be either equal to '
              f'the polynomial degree ({self.degree}) or one more than the '
              'polynomial degree (including real/imaginary specifier).'
            )
        
        if sorted(params) != list(params):
            raise self.key_order_error(
              f'Parameters {params} must be ordered alphabetically in '
              f'{tuplekey}.'
            )
        else:
            return super()._parse_key(tuplekey)

    def _parse_key_old(self, key):
        """
        Parse key, checking it is valid and return as a tuple.

        `key` must be a tuple of strings. The first n elements represent  
        polynomial parameters. Optionally the last element can be a length-n 
        string where each element can only be 'R' or 'I', stating whether the
        real or imaginary component of each coefficient is being referenced. 
        The constant term in the polynomial is specified by a tuple of empty 
        strings. Elements should be ordered alphabetically by convention. 
        If the real/imaginary specifier is not present, parameters are assumed 
        to be real.

        If the key is a string, it is converted to a tuple via `ast.literal_eval()`.

        Parameters
        ----------
        key : tuple
            Key to parse, checking validity as described above.
        
        Returns
        -------
        tuple
            Parsed key as a tuple of three strings.
        
        Raises
        ------
        self.key_error
            If the key is not a valid tuple as described above.
        self.key_order_error
            If the parameters in the key are not ordered alphabetically.

        Notes
        -----
        Most of the checks should automatically be checked by the JSON schema 
        validation step, so they could be skipped in principle. Only the 
        ordering check on the parameter names is not covered by the schema. 
        """
        
        tuplekey = literal_eval(key) if isinstance(key, str) else key

        if not isinstance(tuplekey, tuple) or not tuplekey:
            raise self.key_error(
              f'Invalid key {key}: must be a non-empty tuple or string that '
               'returns a non empty-tuple via literal_eval().'
            ) from None
        
        # Handle optional real/imaginary specifier
        if len(tuplekey) == self.degree:
            params, RI_str = tuplekey, None
        elif len(tuplekey) == self.degree + 1:
            params, RI_str = tuplekey[:-1], tuplekey[-1]
        else:
            raise self.key_error(
              f'Invalid key {key}: number of elements must be either equal to '
              f'the polynomial degree ({self.degree}) or one more than the '
              'polynomial degree (including real/imaginary specifier).'
            )

        # ensure formatting of key conforms to specification
        if not all( isinstance(x, str) for x in params ):
            raise self.key_error(
              f'Invalid key {key}: all elements must be strings.'
            )
        elif len(params) != self.degree:
            raise self.key_error(
              f'Invalid key {key}: number of parameters ({len(params)}) '
              f'must match polynomial degree ({self.degree}).'
            )
        elif RI_str is not None and len(params)!=len(RI_str):
            raise self.key_error(
              f'Invalid key {key}: Length of real/imaginary specifier string ' 
              f'(currently {len(RI_str)}) must match the specified degree of '
              f'the polynomial ({self.degree}).'
            )
        elif sorted(params) != list(params):
            raise self.key_order_error(
              f'Parameters {params} must be ordered alphabetically in '
              f'{tuplekey}.'
            )
        else:
            return super()._parse_key(tuplekey)

    def _parse_value(self, value):
        """
        Parse value, checking it is valid and return it, converting lists to 
        `numpy.ndarray`

        `value` must be a `numpy.ndarray` or a list of  floats that is 
        convertible to `numpy.ndarray`. Lists will be converted 
        to `numpy.ndarray`. 

        Parameters
        ----------
        value : list or numpy.ndarray
            Value to parse, checking validity as described above.
        
        Returns
        -------
        numpy.ndarray
            Parsed value.
        
        Raises
        ------
        self.value_error
            If the value is not a valid `numpy.ndarray` or list as described above.
        self.length_error
            If the length of the array does not match `self.length`.
        self.shape_error
            If the array is not one-dimensional.
        
        """

        if isinstance(value, np.ndarray):
            if len(value.shape) != 1:
                raise self.shape_error(
                f'Invalid shape "{value}": array must be 1D.'
                )
            elif value.shape[0] != self.length:
                raise self.length_error(
                f'Invalid value "{value}": array must have length {self.length}.'
                )        
            try:
                newvalue = value.astype(float)
            except Exception as e:
                raise self.value_error(
                    f'Could not convert "{value}" to numpy.ndarray with type float.'
                ) from e

            return super()._parse_value(newvalue)
        # convert to np.ndarray if list
        if isinstance(value, list):
            try:
                newvalue = np.asarray(value)
            except Exception as e:
                raise self.value_error(
                  f'Could not convert "{value}" to numpy.ndarray.'
                ) from e
            
            return self._parse_value(newvalue)
        else:
            raise self.value_error(
              f'Invalid value "{value}": must be a 1D numpy.ndarray or '
               'list of floats convertible to numpy.ndarray.'
            )

    def to_str_dict(self, suppress_RI=False, use_raw=False):
        """
        Convert the polynomial dictionary to a dictionary with keys as strings.

        This method serializes the polynomial data structure into a plain Python 
        dictionary with string keys and list values, suitable for JSON serialization.

        Parameters
        ----------
        suppress_RI : bool, optional
            If True, remove the real/imaginary specifier string from keys.
            Default is False.
        use_raw : bool, optional
            If True, use the raw input data instead of the parsed internal 
            representation. Default is False.

        Returns
        -------
        dict
            Dictionary with string representations of keys and values converted 
            from numpy arrays to lists.

        """

        result = dict()

        source = self if not use_raw else self._raw_data
        for key, val in source.items():
            if suppress_RI and self.is_RI(key[-1]):
                key = key[:-1]
            result[str(key)] = (
              val.tolist() if isinstance(val, np.ndarray) else val
            )
        return result

    def to_jstr(self, suppress_RI=False, use_raw=False):
        """
        Convert polynomial to a formatted JSON string representation.

        This method produces a pretty-printed JSON string of the polynomial data,
        suitable for writing to POPxf JSON files.

        Parameters
        ----------
        suppress_RI : bool, optional
            If True, remove the real/imaginary specifier string from keys.
            Default is False.
        use_raw : bool, optional
            If True, use the raw input data instead of the parsed internal 
            representation. Default is False.

        Returns
        -------
        str
            Formatted JSON string representation of the polynomial dictionary.

        See Also
        --------
        to_str_dict : Convert to plain dictionary
        to_dict : Convert to dictionary via JSON round-trip

        """

        result = json.dumps(
          self.to_str_dict(suppress_RI=suppress_RI, use_raw=use_raw), 
          default = lambda x: x.to_str_dict(suppress_RI=suppress_RI),
          indent=2
        )

        return pretty_json_string(result)

    def to_dict(self):
        """
        Convert polynomial to a plain Python dictionary.

        This method performs a JSON round-trip conversion (to JSON string and back)
        to ensure all values are standard Python types.

        Returns
        -------
        dict
            Plain Python dictionary representation of the polynomial.

        See Also
        --------
        to_jstr : Convert to JSON string
        to_str_dict : Convert to string-keyed dictionary

        Notes
        -----
        This method uses JSON serialization internally, so all numpy arrays
        are converted to lists.
        """
        return json.loads(self.to_jstr())
    
    # def evaluate(self, *args, **kwargs):
    #     """
    #     Evaluate the polynomial at given parameter values.

    #     Parameters
    #     ----------
    #     *args : float
    #         Parameter values in the order specified by `self.parameters`.
        
    #     **kwargs : float
    #         Parameter values specified by name.
        
    #     Returns
    #     -------
    #     numpy.ndarray
    #         Evaluated polynomial as a numpy array.
        
    #     """

    #     if args and kwargs:
    #         raise ValueError(
    #           'Cannot specify both positional and keyword arguments when '
    #           'evaluating polynomial.'
    #         )

    #     if args:
    #         if len(args) != len(self.parameters):
    #             raise ValueError(
    #               f'Invalid number of positional arguments ({len(args)}): '
    #               f'must match number of parameters ({len(self.parameters)}).'
    #             )
    #         param_values = dict(zip(self.parameters, args))
    #     else:
    #         param_values = kwargs

    #     result = np.zeros(self.length)
    #     # unfinished
    #     for key, coeff in self.items():
    #         # assume real coefficients if no RI specifier given
    #         if not self.is_RI(key[-1]):
    #             key = tuple(*key,'R'*len(key))

    #         term_value = coeff
    #         for i, param in enumerate(key[:-1]):
    #             ri_flag = key[-1][i] if self.is_RI(key[-1]) else 'R'
    #             param_value = param_values.get(param, 0.0)
    #             if ri_flag == 'I':
    #                 param_value = 0.0  # Imaginary part is zero for real inputs
    #             term_value *= param_value
    #         result += term_value

    #     return result
    
class POPxfPolynomialUncertainty(POPxfPolynomial):
    """
    Class to store polynomial uncertainty data in POPxf JSON files.

    This class extends `POPxfPolynomial` to handle uncertainty specifications that 
    may be parameter-independent or parameter-dependent. When uncertainty is 
    parameter-independent (constant across all parameter values), it can be 
    specified as a simple array rather than a full polynomial dictionary.

    Parameters
    ----------
    data : dict, list, or numpy.ndarray
        Uncertainty data. Can be either:
        - A dictionary with the same structure as `POPxfPolynomial` for 
          parameter-dependent uncertainties
        - A 1D list or numpy array for parameter-independent (constant) 
          uncertainties, which will be internally converted to a polynomial
          with a single constant term
    degree : int, optional
        Degree of the polynomial. Default is 2.
    length : int, optional
        Length of coefficient arrays. If not provided, inferred from the first
        entry in `data`.
    precision : float, optional
        Numerical precision for operations (currently unused).

    Attributes
    ----------
    degree : int
        Degree of the polynomial (inherited from parent).
    length : int
        Length of the coefficient arrays (inherited from parent).
    parameters : tuple
        Tuple of parameter names that appear in the polynomial (inherited from parent).

    Raises
    ------
    POPxfPolynomialInitError
        If data is a list/array but not 1D, or if other initialization constraints
        from the parent class are violated.

    Notes
    -----
    When `data` is provided as a list or numpy array, it represents a parameter-
    independent uncertainty. This is internally converted to a polynomial dictionary
    with a single entry: the constant term represented by a tuple of empty strings.
    
    The `_raw_data` attribute stores the original input data format, which is used
    by serialization methods to preserve the simplified representation for 
    parameter-independent uncertainties.
    
    See Also
    --------
    POPxfPolynomial : Parent class for polynomial data storage
    """

    def __init__(self, data, degree=2, length=None):
        # handle special case of parameter independent uncertainty
        if isinstance(data, (list, np.ndarray)):
            if len(np.asarray(data).shape)!=1:
                raise self.init_error(
                  f'Invalid data "{data}": must be a 1D list or numpy.ndarray '
                  'for parameter independent uncertainty.'
                )
            newdata = { ('',)*degree : data }
        else:
            newdata = data

        super().__init__(newdata, degree=degree, length=length)

        self._raw_data = data

    def evaluate(self):
        return NotImplementedError(
          'Evaluation of polynomial uncertainties is not implemented.'
        )
    
    def to_str_dict(self, suppress_RI=False, use_raw=True):
        """
        Convert the uncertainty polynomial to a dictionary with keys as strings.

        This method overrides the parent class method to handle the special case
        of parameter-independent uncertainties, which can be serialized as a 
        simple list string rather than a full dictionary.

        Parameters
        ----------
        suppress_RI : bool, optional
            If True, remove the real/imaginary specifier string from keys.
            Default is False.
        use_raw : bool, optional
            If True, use the raw input data instead of the parsed internal 
            representation. Default is True (unlike parent class default).

        Returns
        -------
        dict or str
            If the raw data is a list or numpy array (parameter-independent 
            uncertainty), returns a string representation of the array.
            Otherwise, returns a dictionary with string keys and list values
            as in the parent class.

        Notes
        -----
        The default value of `use_raw=True` differs from the parent class to
        preserve the simplified representation of parameter-independent 
        uncertainties in serialized output.

        See Also
        --------
        POPxfPolynomial.to_str_dict : Parent class method
        """

        source = self if not use_raw else self._raw_data
        # handle special case of parameter independent uncertainty
    
        if isinstance(source, list):
            return str(source)
        elif isinstance(source, np.ndarray):
            return str(source.tolist())
        else:
            return super().to_str_dict(suppress_RI=suppress_RI, use_raw=use_raw)

    def to_jstr(self, suppress_RI=False, use_raw=True):
        """
        Convert uncertainty polynomial to a formatted JSON string representation.

        This method calls the parent class method with `use_raw=True` by default
        to preserve the simplified representation of parameter-independent 
        uncertainties.

        Parameters
        ----------
        suppress_RI : bool, optional
            If True, remove the real/imaginary specifier string from keys.
            Default is False.
        use_raw : bool, optional
            If True, use the raw input data instead of the parsed internal 
            representation. Default is True (unlike parent class default).

        Returns
        -------
        str
            Formatted JSON string representation of the uncertainty data.

        Notes
        -----
        The default value of `use_raw=True` differs from the parent class to
        preserve the simplified representation of parameter-independent 
        uncertainties in JSON output.

        See Also
        --------
        POPxfPolynomial.to_jstr : Parent class method
        to_str_dict : Convert to string-keyed dictionary
        """
        return super().to_jstr(suppress_RI=suppress_RI, use_raw=use_raw)
