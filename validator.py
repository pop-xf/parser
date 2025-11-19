import re
import sys
import numpy as np
import json
import jsonschema
from io import StringIO
from ast import literal_eval
from pathlib import Path
from jsonschema.exceptions import ValidationError
from schemas import schemas, validators

# TODO:
# implement serialization back to JSON

class POPxfValidator(object):
    """
    Validator for POPxf JSON files.

    This class validates POPxf (Polynomial Observable Prediction eXchange 
    Format) JSON files, which store a data representation for polynomials in 
    model parameters. It supports two modes: Single-Polynomial (SP) mode for 
    direct observable predictions, and Function-Of-Polynomials (FOP) mode for 
    predictions expressed as functions of auxiliary polynomials.

    Parameters
    ----------
    json_data : dict
        Dictionary containing POPxf JSON data. Must include a '$schema' field
        specifying the schema version, a 'metadata' field with observable and
        parameter information, and a 'data' field with polynomial coefficients.

    Attributes
    ----------
    json : dict
        The input JSON data.
    schema_version : str
        Version identifier extracted from the '$schema' field.
    json_schema : dict
        The JSON schema definition for validation.
    validator : jsonschema.Validator
        JSON schema validator instance.
    metadata : dict
        Metadata section from the JSON, containing observable names, parameters,
        scale information, and mode-specific fields.
    data : dict
        Data section from the JSON, containing polynomial coefficients.
    polynomial_degree : int
        Degree of the polynomial expansion (default 2 if not specified).
    mode : str
        Operating mode, either 'SP' (Single-Polynomial) or 'FOP' 
        (Function-Of-Polynomials).
    length_observable_names : int
        Number of observables (length of metadata['observable_names']).
    length_polynomial_names : int
        Number of polynomials (length of metadata['polynomial_names'] in FOP 
        mode, set to 0 in SP mode).
    parameters : list
        List of parameters appearing in the polynomials.
    observable_central : POPxfPolynomial, optional
        Central values for observables in SP mode. Reguired in SP mode.
    polynomial_central : POPxfPolynomial, optional
        Central values for auxiliary polynomials in FOP mode. Present only in 
        FOP mode.
    observable_uncertainties : dict of POPxfPolynomialUncertainty, optional
        Dictionary mapping uncertainty source names to uncertainty polynomials.
        Present only if 'observable_uncertainties' exists in data.

    Static Methods
    --------------
    open_json(jsonfile)
        Open and load a JSON file.
        
    Class Methods
    -------------
    from_json(cls, jsonfile)
        Create a POPxfValidator instance from a JSON file.

    Raises
    ------
    POPxfValidatorError
    POPxfValidationError
        - If the '$schema' field is missing or specifies an unrecognized version.
        - If the JSON data fails schema validation, has inconsistent field lengths,
        contains unrecognized parameters, or violates mode-specific constraints.

    Notes
    -----
    The parser performs multi-level validation:
    
    1. **Schema validation**: Ensures JSON structure conforms to the POPxf schema
    2. **Mode detection**: Determines SP or FOP mode based on present fields
    3. **Scale validation**: Checks consistency of scale field with mode and data
    4. **Polynomial validation**: Validates polynomial keys, values, and parameters
    5. **Length validation**: Ensures array lengths match metadata specifications
    
    **SP Mode**: Observable values are directly represented as polynomials in the
    'observable_central' field.
    
    **FOP Mode**: Observable values are computed from auxiliary polynomials in the
    'polynomial_central' field using expressions in metadata['observable_expressions'].
    
    See Also
    --------
    POPxfPolynomial : Class for storing polynomial data
    POPxfPolynomialUncertainty : Class for storing uncertainty polynomials
    POPxfValidatorError : Base exception class
    POPxfValidationError : Validation error exception class
    """
    
    def __init__(self, json_data):

        self.json = json_data

        # determine schema version from JSON data
        try:
            self.schema_version = json_data["$schema"].split('/')[-1]
        except KeyError as e:
            raise POPxfValidationError(
              "POPxf JSON data is missing required '$schema' field."
            ) from None
        
        # load schema from map defined in schemas.py
        try:
            self.json_schema = schemas[self.schema_version]
        except KeyError:
            allowed_schemas = ', '.join([ f"'{v}'" for v in schemas.keys() ])
            raise POPxfValidationError(
              f"POPxf JSON schema version '{self.schema_version}' is not "
              "recognized. Available versions are: "
              f"{allowed_schemas}"
            ) from None

        # load validator from map defined in schemas.py
        try:
            self.validator = validators[self.schema_version](self.json_schema)
        except KeyError:
            allowed_validators = ', '.join([ f"'{v}'" for v in validators.keys() ])
            raise POPxfValidationError(
              f"Validator for schema version '{self.schema_version}' is not "
              "recognized. Available versions are: "
              f"{allowed_validators}"
            ) from None 

        # validate against schema
        self.validate_schema()

        # get metadata and data fields
        self.metadata = self.json["metadata"]
        self.data = self.json["data"]
        # set other useful attributes
        self.polynomial_degree = self.metadata.get("polynomial_degree", 2)
        ## determine mode
        self.mode = "FOP" if "polynomial_names" in self.metadata else "SP"
        ## length of observable
        self.length_observable_names = len(self.metadata["observable_names"])
        self.length_polynomial_names = len(self.metadata.get("polynomial_names", []))
        ## parameters
        self.parameters = self.metadata["parameters"]

        # validate other fields beyond schema
        self.validate_other()

        # set polynomial data, performs additional validation
        self.set_poly_data()

    def validate_schema(self):
        """
        Validate JSON data against the POPxf schema.

        Uses the jsonschema validator to check that the input JSON conforms to
        the POPxf schema specification for the detected schema version.

        Raises
        ------
        POPxfValidationError
            If the JSON data fails schema validation. The error message includes
            detailed information about the validation failure, including the path
            to the problematic field.

        Notes
        -----
        This method is called automatically during initialization. Validation
        errors are converted from jsonschema.ValidationError to POPxfValidationError
        with enhanced error messages generated by `get_validation_error_message`.

        See Also
        --------
        get_validation_error_message : Generate detailed error messages
        """
        try:
            self.validator.validate(
              instance=self.json
            )
        except ValidationError as e:
            error_message = self.get_validation_error_message(e)

            raise POPxfSchemaValidationError(
              "POPxf JSON data does not conform to schema version "
             f"{self.schema_version}:\n{error_message}"
            ) from None

    @classmethod
    def get_validation_error_message(cls, error):
        """
        Generate a detailed error message from a jsonschema.ValidationError.

        This class method recursively processes validation errors, including 
        nested errors in subschemas, to produce human-readable error messages
        with full path information to the problematic field.

        Parameters
        ----------
        error : jsonschema.exceptions.ValidationError
            The validation error object from JSON schema validation.

        Returns
        -------
        str
            Formatted error message with path information and description of
            the validation failure. For errors with nested subschemas (e.g.,
            'oneOf' constraints), combines suberror messages with appropriate
            logical operators.

        Notes
        -----
        The method handles several types of validation errors:
        
        - 'required': Missing required fields
        - 'pattern': Pattern mismatch for string values
        - 'oneOf': Errors from oneOf schema constraints (combined with 'AND')
        - Other errors: Generic validation failures
        
        For nested schema errors, suberrors are combined with 'OR' (default)
        or 'AND' (for oneOf constraints).

        """
        # recursively deal with errors in subschemas
        if not error.context:
            rel_path = list(error.absolute_path)
            # build path string
            # empty path means root object
            if not rel_path:
                rel_path.append('the root JSON object')
        
            path = ''.join(
              [rel_path[0]]+[
                f"[{str(x)}]" if isinstance(x, int) else f'["{x}"]'
                for x in rel_path[1:]
              ]
            )

            if error.validator == 'required':
                # missing required field
                msg_suffix = f" of '{path}'"
            elif error.validator == 'pattern':
                # pattern mismatch
                msg_suffix = f" pattern for '{path}'"
            elif error.validator == 'not':
                # forbidden field present
                if 'required' in error.validator_value:
                    forbidden = ', '.join(error.validator_value['required'])
                    msg_suffix = f".\n  '{forbidden}' forbidden in '{path}'"
                elif '$ref' in error.validator_value:
                    if error.validator_value['$ref']=='#/$defs/stringifiedTuplePattern':
                        msg_suffix = (
                          f".\n  'stringified tuple fields are forbidden in '{path}'"
                        )
                    else:
                        msg_suffix = f"{error.instance} forbidden in '{path}'"
                else:
                    msg_suffix = f" in '{path}'"
            else:
                msg_suffix = f" in '{path}'"
            
            return f"  {error.message}{msg_suffix}"
        else:
            sep = 'AND' if error.validator == 'oneOf' else 'OR'
            suberror_messages = [
              cls.get_validation_error_message(suberror)
              for suberror in error.context
            ]
            # remove possible duplicates before concatenating and returning
            return f"\n  {sep}\n".join(list(set(suberror_messages)))
        
    def validate_other(self):
        """
        Perform additional validations beyond schema checks. Assumes that the 
        JSON validation check has suceeded and the following attributes have 
        been set:

        - self.metadata
        - self.data
        - self.polynomial_degree
        - self.parameters
        - self.length_observable_names
        - self.mode

        Checks performed:
        1. self.validate_scale(): Scale field consistency with mode and metadata/data
        2. self.validate_data(): Polynomial data consistency with mode and metadata

        Raises
        ------
        POPxfValidationError
            If any validations fail.

        """

        # validate scale field
        self.validate_scale()
        # validate FOP-mode observable expressions lengths
        self.validate_expressions()
        # validate data polynomials
        self.validate_data()

    def validate_scale(self):
        """
        Validate the 'scale' field in metadata for consistency with mode.
        - Specifying the scale as a single number requires no further checks.
        - Array-valued scales must have lengths matching observable_names
          in single-polynomial (SP) mode or polynomial_names in 
          function-of-polynomials (FOP) mode.
        
        Raises
        ------
        POPxfValidationError
            If the scale field is inconsistent with mode and metadata.

        """
        scale = self.metadata["scale"]
        # could be a fancier numeric type check
        if isinstance(scale, (int, float)):
            # scale is just a number
            return None
        elif self.mode == 'SP':
            # length matches observable_names
            if len(scale) != len(self.metadata["observable_names"]):
                raise POPxfScaleError(
                    "Lengths of array-valued 'scale' and "
                    "'observable_names' metadata fields must match in "
                    "single-polynomial (SP) mode."
                )
        elif self.mode == 'FOP':
            # length matches polynomial_names
            if len(scale) != len(self.metadata["polynomial_names"]):
                raise POPxfScaleError(
                    "Lengths of array-valued 'scale' and "
                    "'polynomial_names' metadata fields must match in "
                    "function-of-polynomials (FOP) mode."
                )

    def validate_data(self):
        """
        Validate polynomial data for consistency with mode and metadata.
        Checks that polynomial value lengths match expected lengths depending 
        on mode using self.validate_polynomial().

        - In SP mode: Lengths of polynomial values in 'observable_central' must 
          match the length of `observable_names` in metadata.
        - In FOP mode: Lengths of polynomial values in 'polynomial_central' 
          must match the length of `polynomial_names` in metadata.
        - If present, lengths of polynomial values in 'observable_uncertainties'
          must match the length of `observable_names` in metadata. Accounts for 
          the special case of specifying parameter-independent uncertainties by 
          providing an array instead of a JSON object (dictionary).

        """
        if self.mode == 'SP':
            # single-polynomial mode
            # validate observable_central
            self.validate_polynomial(
              self.data["observable_central"], 
              self.length_observable_names,
              'data["observable_central"]'
            )
            
        elif self.mode == 'FOP':
            # function-of-polynomials mode
            # validate polynomial_central
            self.validate_polynomial(
              self.data["polynomial_central"],
              self.length_polynomial_names,
              'data["polynomial_central"]'
            )

        # validate observable_uncertainties if present
        if "observable_uncertainties" in self.data:
            for k,v in self.data["observable_uncertainties"].items():
                if isinstance(v, dict):
                    self.validate_polynomial(
                      v, 
                      self.length_observable_names,
                      f'data["observable_uncertainties"][{k}]'
                    )
                elif isinstance(v, list):
                    self.validate_polynomial(
                      { "('','')": v }, 
                      self.length_observable_names,
                      f'data["observable_uncertainties"][{k}]'
                    )

    def validate_expressions(self):
        """
        Validate observable expressions in FOP mode.

        Checks that the number of observable expressions matches the number of
        observable names in metadata when operating in function-of-polynomials
        (FOP) mode.

        Raises
        ------
        POPxfValidationError
            If the number of observable expressions does not match the number
            of observable names in metadata.

        """
        if self.mode == 'FOP':
            num_expressions = len(self.metadata['observable_expressions'])
            if num_expressions != self.length_observable_names:
                raise POPxfFOPExpressionError(
                  "In function-of-polynomials (FOP) mode, the number of "
                  "observable expressions in 'metadata.observable_expressions' "
                  "must match the number of observable names in "
                  "'metadata.observable_names'."
                )

    def validate_polynomial(self, poly, expected_length, poly_name):
        """
        Validate a polynomial's keys and values for consistency. 
        Checks that the value lengths match `expected_length` and that the keys 
        are alphabetically ordered accounting for the optional real/imaginary 
        specifier.

        Checks that the set of parameters in the polynomial is a subset of 
        `self.parameters`.

        Assumes that the input JSON file has already passed schema validation.

        Parameters
        ----------
        poly : dict
            Polynomial data as a dictionary mapping keys to coefficient arrays.
        expected_length : int
            Expected length of the polynomial coefficient arrays.
        poly_name : str
            Name/path of the polynomial field being checked (for error messages).
        
        Raises
        ------
        POPxfValidationError
            If the polynomial data is inconsistent, such as:
            - Keys not ordered alphabetically
            - Value lengths not matching expected_length
            - Parameters not listed in metadata.parameters
        """
        poly_params = []
        for k,v in poly.items():
            # check key format
            tuplekey = literal_eval(k)
            ## handle optional RI specifier
            if len(tuplekey) == self.polynomial_degree:
                params, RI_str = tuplekey, None
            elif len(tuplekey) == self.polynomial_degree + 1:
                params, RI_str = tuplekey[:-1], tuplekey[-1]
            else:
                ## invalid key length, never raised assuming schema validation passed
                raise POPxfKeyLengthError(
                 f"Error initialising '{poly_name}' polynomial data:\n"
                 f"  Invalid key {k}: number of elements must be either equal to "
                 f"or one greater that the polynomial degree ({self.polynomial_degree})."
                )
            
            # check key ordering
            if sorted(params) != list(params):
                raise POPxfKeyOrderError(
                  f"Error initialising '{poly_name}' polynomial data:\n"
                  f"  Parameters in {params} must be ordered alphabetically in "
                  f"{tuplekey}."
                )
            
            # check value length
            if len(v) != expected_length:
                raise POPxfDataLengthError(
                  f"Error initialising '{poly_name}' polynomial data:\n"
                  f'  Polynomial values for key "{k}" must have length '
                  f"{expected_length}, got {len(v)}."
                )
            
            poly_params += [ x for x in params ]

        extra_parameters = (set(poly_params) - {''}) - set(self.parameters)
        if extra_parameters:
            raise POPxfParameterSetError(
              f"Error initialising '{poly_name}' polynomial data:\n"
              f"'{poly_name}' contains unrecognized parameters {extra_parameters} "
              "not listed in metadata.parameters."
            )
        
    @staticmethod
    def get_poly_params(poly, degree):
        """
        Get the set of parameters used in a polynomial.

        Parameters
        ----------
        poly : dict
            Polynomial data as a dictionary mapping keys to coefficient arrays.

        Returns
        -------
        set
            Set of unique parameters used in the polynomial keys.
        """
        poly_params = set()
        for k in poly.keys():
            tuplekey = literal_eval(k)
            # handle optional RI specifier
            if len(tuplekey) ==  degree + 1:
                params = tuplekey[:-1]
            else:
                params = tuplekey
            poly_params.update([ x for x in params if x != '' ])
        
        return sorted(poly_params)

    def info(self, verbose=False, show_data=False, show_uncertainties=False):
        """
        Generate a summary string of the POPxfValidator object's properties.

        Returns
        -------
        str
            Formatted string summarizing the parser's properties, including
            schema version, mode, polynomial order, number of observables,
            parameters, metadata keys, and details about central values and
            uncertainties.
        """

        result = StringIO()

        result.write("=" * 70 + "\n")
        result.write(f"{self.__class__.__name__} Object Properties\n")
        result.write("=" * 70 + "\n")
        
        result.write(f"\nSchema version: {self.schema_version}\n")

        if verbose:
            result.write("\nmetadata keys:\n")
            for key in self.metadata.keys():
                result.write(f"  - {key}\n")
            
            result.write("\ndata keys:\n")
            for key in self.data.keys():
                result.write(f"  - {key}\n")

        if self.mode == 'SP':
            result.write("\nMode: Single-Polynomial (SP)\n")
        else:
            result.write("\nMode: Function-Of-Polynomials (FOP)\n")

        result.write(f"Polynomial Degree: {self.polynomial_degree}\n")
        result.write(f"Length (number of observables): {self.length_observable_names}\n")        
        result.write(f"Observable Names: {self.metadata['observable_names']}\n")
        result.write(f"Parameters: {self.parameters}\n")
        result.write(f"Scale: {self.metadata['scale']} [GeV]\n")

        if verbose and self.mode == 'FOP':
            poly_names = self.metadata['polynomial_names']
            poly = self.data['polynomial_central']
            params = self.get_poly_params(poly, self.polynomial_degree)
            result.write("\nFOP data:\n")
            result.write(f"  - Number of polynomials: {len(poly_names)}\n")
            result.write("  - Polynomial names: " + ", ".join(poly_names) + "\n")
            result.write(f"\nPolynomial Central:\n")
            result.write(f"  - Number of polynomial terms: {len(poly)}\n")
            result.write(f"  - Parameters: {params}\n")

            result.write(f"\nObservable Expressions:\n")
            for i, (obs, expr) in enumerate(zip( 
              self.metadata['observable_names'], 
              self.metadata['observable_expressions']
            )):
                result.write(f"  [{obs}] {expr}\n")

        if verbose and 'observable_central' in self.data:
            poly = self.data['observable_central']
            params = self.get_poly_params(poly, self.polynomial_degree)
            result.write("\nObservable Central:\n")
            result.write(f"  - Parameters: {params}\n")
            result.write(f"  - Number of polynomial terms: {len(poly)}\n")
            result.write(f"  - Polynomial keys: {list(poly.keys())[:5]}{'...' if len(poly) > 5 else ''}\n")

        result.write(f"\n{'='*70}")
        # Show polynomial data if requested
        if show_data:
            if self.mode == 'SP':
                poly = self.data['observable_central']
                path ='data["observable_central"]'
            else:
                poly = self.data['polynomial_central']
                path ='data["polynomial_central"]'

            result.write(f"\nPolynomial Data ({path}):\n")
            result.write(f"{'='*70}\n")
            for key, value in poly.items():
                result.write(f"  {key}: {value}\n")
            result.write(f"\n{'='*70}")

        # Show detailed uncertainty information if requested
        if show_uncertainties and 'observable_uncertainties' in self.data:
            uncs = self.data['observable_uncertainties']
            result.write(f"\nObservable Uncertainties:\n")
            result.write(f"{'='*70}\n")
            result.write(f"  Number of uncertainty sources: {len(uncs)}\n")
            for unc_name, unc_obj in uncs.items():
                result.write(f"  Source '{unc_name}':\n")
                if isinstance(unc_obj, dict):
                    params = self.get_poly_params(unc_obj, self.polynomial_degree)
                    result.write(f"    Parameters: {params}\n")
                    result.write(f"    Number of polynomial terms: {len(unc_obj)}\n")
                    if show_data:
                        result.write(f"    Polynomial data:\n")
                        for key, value in unc_obj.items():
                            result.write(f"      {key}: {value}\n")
                else:
                    result.write(f"    Parameter-independent uncertainty\n")
                    if show_data:
                        result.write(f"    Polynomial data:\n")
                        result.write(f"      {unc_obj}\n")
            result.write(f"\n{'='*70}")
        else:
            result.write("\nNo uncertainty information available.\n")
            result.write(f"{'='*70}")

         # Show reproducibility information if verbose
        if verbose and 'reproducibility' in self.metadata:
            result.write(f"\nReproducibility Information:\n")
            result.write(f"{'='*70}\n")
            repro = self.metadata['reproducibility']
            for i, item in enumerate(repro, start=1):
                result.write(f"  Step {i}:\n")
                for key, value in item.items():
                    result.write(f"  - {key}: {value}\n")
            result.write(f"\n{'='*70}")

        return result.getvalue()
        
    @staticmethod
    def open_json(jsonfile):
        """
        Open and load a POPxf JSON file.

        Parameters
        ----------
        jsonfile : str or Path
            Path to the POPxf JSON file.

        Returns
        -------
        dict
            Loaded JSON data as a dictionary.

        Raises
        ------
        POPxfValidatorIOError
            If there are issues reading the file (e.g., file not found, 
            permission denied).
        POPxfValidatorJSONError
            If the file content is not valid JSON.

        """
        try:
            input_path = Path(jsonfile)
            
            if not input_path.exists():
                raise POPxfIOError(f"File not found: {jsonfile}")
            
            if not input_path.is_file():
                raise POPxfIOError(f"Not a file: {jsonfile}")
            
            with open(input_path, 'r') as f:
                json_data = json.load(f)
        
        except IOError as e:
            raise POPxfIOError(
              f"Failed to read file: {jsonfile}"
            ) from e

        except json.JSONDecodeError as e:
            raise POPxfJSONError(
              f"Invalid JSON format in file: {jsonfile}\n"
              f"  Line {e.lineno}, Column {e.colno}: {e.msg}"
            ) from e

        return json_data

    @classmethod
    def from_json(cls, jsonfile):
        """
        Create a POPxfValidator instance from a JSON file.

        Parameters
        ----------
        jsonfile : str or Path
            Path to the POPxf JSON file.

        Returns
        -------
        POPxfValidator
            An instance of POPxfValidator initialized with the JSON data.
        """
        json_data = cls.open_json(jsonfile)

        return cls(json_data)
        

"""Exceptions for POPxf JSON validation errors."""

class POPxfValidationError(Exception):
    """
    Base exception class for POPxf JSON validation errors.

    Notes
    -----
    Specific error types (e.g., validation errors) are implemented as subclasses
    of this base exception. Catching POPxfValidationError will catch all POPxf-related
    errors. Validation errors include:
    
    - Schema conformance failures (missing fields, wrong types, pattern mismatches)
    - Mode detection and mode-specific field requirements
    - Array length inconsistencies between data and metadata
    - Unrecognized parameters not listed in metadata.parameters
    - Invalid polynomial keys or values
    - Scale field inconsistencies

    See Also
    --------
    POPxfValidator.validate_schema : Schema validation method
    POPxfValidator.get_validation_error_message : Error message formatting
    POPxfValidator.validate_other : Beyond-schema validation method
    """

class POPxfSchemaValidationError(POPxfValidationError):
    """
    Exception class for POPxf JSON schema validation errors.

    Notes
    -----
    Raised when the input JSON data fails to conform to the POPxf schema
    specification for the detected schema version. This includes missing required
    fields, incorrect data types, pattern mismatches, and other structural issues.

    See Also
    --------
    POPxfValidator.validate_schema : Schema validation method
    POPxfValidator.get_validation_error_message : Error message formatting
    """

class POPxfScaleError(POPxfValidationError):
    """
    Exception class for POPxf JSON scale field validation errors.

    Notes
    -----
    Raised when the 'scale' field in metadata is inconsistent with the operating
    mode (SP or FOP) and the lengths of observable_names or polynomial_names.

    See Also
    --------
    POPxfValidator.validate_scale : Scale validation method
    """

class POPxfFOPExpressionError(POPxfValidationError):
    """
    Exception class for POPxf JSON FOP expression validation errors.

    Notes
    -----
    Raised when the number of observable expressions in metadata does not match
    the number of observable names when operating in function-of-polynomials (FOP)
    mode.

    See Also
    --------
    POPxfValidator.validate_expressions : FOP expression validation method
    """

class POPxfKeyLengthError(POPxfValidationError):
    """
    Exception class for POPxf JSON data length validation errors.

    Notes
    -----
    Raised when polynomial value lengths in the data section do not match the
    expected lengths based on metadata specifications.

    See Also
    --------
    POPxfValidator.validate_polynomial : Polynomial length validation method
    """

class POPxfKeyOrderError(POPxfValidationError):
    """
    Exception class for POPxf JSON data key validation errors.

    Notes
    -----
    Raised when polynomial keys in the data section are not ordered 
    alphabetically according to python sort().

    See Also
    --------
    POPxfValidator.validate_polynomial : Polynomial key validation method
    """

class POPxfDataLengthError(POPxfValidationError):
    """
    Exception class for POPxf JSON data length validation errors.

    Notes
    -----
    Raised when polynomial value lengths in the data section do not match the
    expected lengths based on metadata specifications.

    See Also
    --------
    POPxfValidator.validate_polynomial : Polynomial length validation method
    """

class POPxfParameterSetError(POPxfValidationError):
    """
    Exception class for POPxf JSON parameter set validation errors.

    Notes
    -----
    Raised when polynomial data contains parameters not listed in 
    metadata.parameters.

    See Also
    --------
    POPxfValidator.validate_polynomial : Polynomial parameter validation method
    """

class POPxfIOError(Exception):
    """
    Exception class for POPxf JSON file I/O errors.

    Notes
    -----
    Raised when there are issues reading or parsing the input POPxf JSON file,
    such as file not found, permission denied, or invalid JSON format.

    See Also
    --------
    POPxfValidator.open_json : JSON file opening method
    """

class POPxfJSONError(POPxfIOError):
    """
    Exception class for POPxf JSON parsing errors.

    Notes
    -----
    Raised when the input POPxf JSON file contains invalid JSON syntax or format.

    See Also
    --------
    POPxfValidator.open_json : JSON file opening method
    """

if __name__ == "__main__":
    pass
    # import sys
    # example = json.load(open('examples/Gam_Wmunum.json'))

    # example = json.load(open('examples/R_W_lilj.json'))
    # example = json.load(open('examples/BR_Bs_mumu_B0_mumu.json'))
    # example = json.load(open('examples/BR_Bs_mumu.json'))
    # example = json.load(open('examples/BR_B0_mumu.json'))
    # example = json.load(open('examples/bad/missing_polynomial_names.json'))
    # example = json.load(open('examples/bad/missing_observable_expressions.json'))
    # example = json.load(open('examples/bad/bad_length_observable_central.json'))
    # example = json.load(open('examples/bad/bad_keys_observable_uncertainties.json'))
    # example = json.load(open('examples/bad/bad_observable_central_scale_array_FOP.json'))
    # example = json.load(open('examples/bad/bad_observable_uncertainties_scale_array_FOP.json'))
   # 
    # from glob import glob
    # bad_files = glob('examples/bad/*.json')
    # guy = POPxfValidator.from_json('examples/bad/bad_observable_uncertainties_scale_array_FOP.json')
    # print(guy.parameters)
    
    # print(guy.info())

    # test_data = {
    #   "('', '')": [0.22729],
    #   "('', 'c3pl1')": [-0.0137796],
    #   "('', 'c3pl2')": [0.0137786],
    #   "('', 'cll')": [0.0137796],
    #   "('c3pl1', 'c3pl1')": [0.000208845],
    #   "('c3pl2', 'c3pl2')": [0.00020885],
    #   "('cll', 'cll')": [0.00020884],
    #   "('c3pl1', 'c3pl2')": [-0.00041769],
    #   "('c3pl1', 'cll')": [-0.00041768],
    #   "('c3pl2', 'cll')": [0.00041768],
    #   "('RR', 'c3pl2')": [0.00041768]
    # }

    # POPxfPolynomial(
    #   test_data,
    #   degree=2,
    #   length=1
    # )
