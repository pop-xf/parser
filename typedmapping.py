from abc import ABCMeta
from collections.abc import MutableMapping
from warnings import warn

class TypedMappingMeta(ABCMeta):
    """
    Base metaclass for the `TypedMapping` class.
    Implement the generation of custom exception classes. Defines the base 
    exception and base warning classes using class name as prefix. 
    Implements derived exception classes analogous to `KeyError` and 
    `ValueError` with the same prefix. Also defines a warning class for 
    duplicate keys.
    """

    def __new__(metaname, classname, baseclasses, attrs):
        base_exception = type(classname+'Error', (Exception,),{})
        attrs['base_error'] = base_exception

        base_warning = type(classname+'Warning', (UserWarning,),{})
        attrs['base_warning'] = base_warning

        attrs['init_error'] = type(
          classname+'InitError', 
          (base_exception, ),
          {}
        )

        attrs['key_error'] = type(
          classname+'KeyError', 
          (base_exception, KeyError),
          {}
        )

        attrs['value_error'] = type(    
          classname+'ValueError', 
          (base_exception, ValueError),
          {}
        )
        
        attrs['duplicate_key_warning'] = type(
            classname+'DuplicateKeyWarning',
            (base_warning,),
            {}
        )

        return super().__new__(metaname, classname, baseclasses, attrs)

class TypedMapping(MutableMapping, metaclass=TypedMappingMeta):
    """
    Dict-like class with restrictions on types for keys and values. 
    Warns you when you are setting a duplicate key in constructor.

    Attributes
    ----------
    key_types : tuple
        Tuple of acceptable types for keys.
    value_types : tuple
        Tuple of acceptable types for values.

    Default Exceptions
    ------------------
    note: automatically generated and named by the metaclass, see 
    StrictDictMeta documentation for details.

    init_error 
        Custom Exception class for errors during initialisation.
    key_error
        Custom Exception class for key errors.
    value_error
        Custom Exception class for value errors.
    duplicate_key_warning
        Custom Warning class for duplicate key warnings.
    
    ABC Methods
    -----------
    Required methods to implement ABC & achieve dict-like behaviour.

    __getitem__(key)
        Retrieves the value associated with the given key.
    __setitem__(key, value)
        Sets the value for the given key.
    __delitem__(key)
        Deletes the item associated with the given key.
    __iter__()
        Returns an iterator over the keys of the dictionary.
    __len__()
        Returns the number of items in the dictionary.

    Methods
    -------
    __repr__()
        Returns the string representation of the dictionary.
    update(data, warn_duplicates=False)
        Updates the dictionary with the given data.
    _parse_key(key)
        Validates and parses the key. Should be overloaded by subclasses.
    _parse_value(value)
        Validates and parses the value. Should be overloaded by subclasses.
    init_from_json(datafile)
        Class method to initialize the dictionary from a JSON file.
    to_str_dict()
        Converts the dictionary to a dictionary with keys as strings.
    to_jstr()
        Converts the dictionary to a JSON string representation.
    to_dict()
        Converts the dictionary to a standard Python dictionary.
    
    """

    key_types = (object,)
    value_types = (object,)

    def __init__(self, data):

        if not (isinstance(data, dict) or isinstance(data, MutableMapping)):
            raise self.init_error(
              f'Invalid data "{data}": must be a dictionary or dict-like.'
            )
        
        self._data = dict()

        try:
            self.update(data, warn_duplicates=True)
        except Exception as e:
            raise self.init_error(
              f'Error initialising with data {data}'
            ) from e

    def __getitem__(self, key):
        try:
            return self._data[self._parse_key(key)]
        except (self.key_error, KeyError) as e:
            raise self.key_error(f'Key {key} not found.') from None

    def __setitem__(self, key, value):
        try:
            self._data[self._parse_key(key)] = self._parse_value(value)
        except Exception as e:
            raise self.value_error(
              f'Error setting key "{key}" to value "{value}".'
            ) from e
        
    def __delitem__(self, key):
        try:
            del self._data[self._parse_key(key)]
        except Exception as e:
            raise self.key_error(f'Key {key} not found.') from e
    
    def __iter__(self):
        return iter(self._data)
    
    def __len__(self):
        return len(self._data)
    
    def __repr__(self):
        return repr(self._data)
    
    def update(self, data, warn_duplicates=False):
        """
        Update the dictionary with the key-value pairs from the given data.
        Parameters
        ----------
        data : dict
            A dictionary containing key-value pairs to update the current 
            dictionary.
        warn_duplicates : bool, optional
            If True, a warning will be issued when a duplicate key is found 
            (default is False).
        Raises
        ------
        UserWarning
            If warn_duplicates is True and a duplicate key is found, a warning 
            will be issued.
        """

        for key, val in data.items():
            if warn_duplicates and key in self:
                warn(
                  f'Duplicate key "{key}" found. Overwriting with new value.',
                  self.duplicate_key_warning
                )
            self[key] = val

    def _parse_key(self, key):
        """
        Parse and validate the given key.

        Checks that the key is of a valid type specified in self.key_types.

        Parameters
        ----------
        key : any
            The key to be validated.

        Returns
        -------
        key : any
            The validated key if it is of a valid type.

        Raises
        ------
        self.key_error
            If the key is not of a valid type specified in self.key_types.
        """

        if not any([ isinstance(key, t) for t in self.key_types]):
            raise self.key_error(
              f'Invalid key "{key}" for {self.__class__}: must be one of '
              f'types {self.key_types}.'
            )
        return key
    
    def _parse_value(self, value):
        """
        Parse and validate the given value.

        Checks that the value is of a valid type specified in self.value_types.

        Parameters
        ----------
        value : any
            The value to be parsed and validated.

        Returns
        -------
        any
            The validated value if it matches one of the allowed types.
        
        Raises
        ------
        ValueError
            If the value does not match any of the allowed types specified in 
            `self.value_types`.
        """

        if not any([ isinstance(value, t) for t in self.value_types]):
            raise self.value_error(
              f'Invalid value "{value}" for {self.__class__}: must be one of '
              f'types {self.value_types}.'
            )
        return value
