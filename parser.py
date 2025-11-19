import re
import sys
import numpy as np
import json
import jsonschema
from io import StringIO
from jsonschema.exceptions import ValidationError
from schemas import schemas, validators
from validator import POPxfValidator
from polynomial import POPxfPolynomial, POPxfPolynomialUncertainty
from validator import POPxfValidator
# TODO:
# implement evaluate()
# implement serialization back to JSON
# implement uncertainty treatment to get covariance matrices, etc.
# splitting a file into multiple files

class POPxfParser(POPxfValidator):
    """
    Parser for POPxf JSON files.

    This class validates and parses POPxf (Polynomial Observable Prediction 
    eXchange Format) JSON files, which store a data representation for 
    polynomials in model parameters. It supports two modes: Single-Polynomial 
    (SP) mode for   direct observable predictions, and Function-Of-Polynomials 
    (FOP) mode for predictions expressed as functions of auxiliary polynomials.

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
    length : int
        Number of observables (length of metadata['observable_names']).
    parameters : list
        List of EFT parameters appearing in the polynomials.
    observable_central : POPxfPolynomial, optional
        Central values for observables in SP mode. Reguired in SP mode.
    polynomial_central : POPxfPolynomial, optional
        Central values for auxiliary polynomials in FOP mode. Present only in 
        FOP mode.
    observable_uncertainties : dict of POPxfPolynomialUncertainty, optional
        Dictionary mapping uncertainty source names to uncertainty polynomials.
        Present only if 'observable_uncertainties' exists in data.

    Raises
    ------
    POPxfParserError
        If the '$schema' field is missing or specifies an unrecognized version.
    POPxfValidationError
        If the JSON data fails schema validation, has inconsistent field lengths,
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
    POPxfParserError : Base exception class
    POPxfValidationError : Validation error exception class
    """
    
    def __init__(self, json_data):

        super().__init__(json_data)

        # set polynomial data, no additional validation needed
        self.set_poly_data()

    def set_poly_data(self):
        """
        Parse polynomial data from the JSON data section.

        Converts polynomial data dictionaries into POPxfPolynomial and 
        POPxfPolynomialUncertainty objects.

        Since the parent POPxfValidator class already performs full validation 
        of the input data, no additional validation is needed.

        Raises
        ------
        POPxfParserError
            If polynomial initialization fails due to invalid keys, values, or
            array lengths, or if polynomials contain parameters not declared in
            metadata.parameters.

        Notes
        -----
        **Validation performed:**
        
        1. **Array length consistency**:
           - SP mode: polynomial values must match length of metadata['observable_names']
           - FOP mode: polynomial values must match length of metadata['polynomial_names']
           - Uncertainties: must match length of metadata['observable_names']
        
        2. **Key/value format**: Validated by POPxfPolynomial constructor
           - Keys must be tuples matching polynomial_degree
           - Values must be 1D numerical arrays
           - Keys must be alphabetically ordered
        
        3. **Parameter declarations**: All parameters in polynomials must be
           listed in metadata['parameters']
        
        **Fields parsed by mode:**
        
        - **SP mode**: Sets `self.observable_central` (POPxfPolynomial)
        - **FOP mode**: Sets `self.polynomial_central` (POPxfPolynomial)
        - **Both modes**: Sets `self.observable_uncertainties` (dict of 
          POPxfPolynomialUncertainty) if present in data

        See Also
        --------
        POPxfPolynomial : Class for polynomial data storage
        POPxfPolynomialUncertainty : Class for uncertainty polynomials
        raise_polynomial_error : Error message generation
        check_parameter_subset : Parameter validation
        """
        if self.mode == 'SP':
            # single-polynomial mode
            # validate observable_central
            try:
                observable_central = POPxfPolynomial(
                  self.data["observable_central"],
                  degree=self.polynomial_degree,
                  length=self.length_observable_names
                )
            except (POPxfPolynomial.init_error) as e:
                msg = "Error initialising 'observable_central' polynomial data"
                self.raise_polynomial_error(e, msg)
            
            self.observable_central = observable_central

        elif self.mode == 'FOP':
            # function-of-polynomials mode
            # validate polynomial_central
            try:
                polynomial_central = POPxfPolynomial(
                  self.data["polynomial_central"],
                  degree=self.polynomial_degree,
                  length=self.length_polynomial_names
                )
            except (POPxfPolynomial.init_error) as e:
                msg = "Error initialising data['polynomial_central'] polynomial data"
                self.raise_polynomial_error(e, msg)
            
            self.polynomial_central = polynomial_central

        # validate observable_uncertainties if present
        if "observable_uncertainties" in self.data:

            self.observable_uncertainties = {}
            for k,v in self.data["observable_uncertainties"].items():
                try:
                    observable_uncertainty = POPxfPolynomialUncertainty(
                      v,
                      degree=self.polynomial_degree,
                      length=self.length_observable_names
                    )

                    self.observable_uncertainties[k] = observable_uncertainty

                except (POPxfPolynomialUncertainty.init_error) as e:
                    msg = (
                      f"Error initialising '{k}' entry of "
                      f"data['observable_uncertainties'] polynomial data."
                    )
                    self.raise_polynomial_error(e, msg)

    def raise_polynomial_error(self, exception, msg_prefix):
        """
        Raise a POPxfValidationError with enhanced messaging for polynomial errors.

        Analyzes the exception chain to determine the root cause and generates
        a detailed error message with context-specific guidance for fixing the
        issue.

        Parameters
        ----------
        exception : Exception
            The caught exception from polynomial initialization.
        msg_prefix : str
            Prefix describing which polynomial field caused the error.

        Raises
        ------
        POPxfValidationError
            Always raised with an enhanced error message that includes the
            prefix and a description of the specific validation failure.

        Notes
        -----
        The method identifies the root cause by traversing the exception chain
        and provides specialized messages for:
        
        - **Value/Length/Shape errors**: Issues with polynomial coefficient arrays
          (wrong dimension, incorrect length, non-numeric values)
        - **Key/KeyOrder errors**: Issues with polynomial keys (wrong format,
          incorrect degree, improper ordering, invalid RI specifiers)
        - **Other errors**: Generic error message
        
        The expected array length depends on the mode:
        - SP mode: length of metadata['observable_names']
        - FOP mode: length of metadata['polynomial_names']

        See Also
        --------
        get_poly_data : Method that uses this for error handling
        """
        # find root cause
        causes = [exception.__cause__]
        while causes[-1] is not None:
            causes.append(causes[-1].__cause__)
        last_cause = causes[-2]

        expected_length = (
          self.length if self.mode=='SP' else 
          len(self.metadata["polynomial_names"])
        )
        # specific messaging depending on cause
        if isinstance(
            last_cause, 
            (POPxfPolynomial.value_error,
            POPxfPolynomial.length_error, 
            POPxfPolynomial.shape_error)
        ):
            reason = (
              ":\n Polynomial values should be 1D numerical arrays matching "
             f"the length ({expected_length}) of "
              "metadata.observable_names."
            )
        elif isinstance(
            last_cause, 
            (POPxfPolynomial.key_error, POPxfPolynomial.key_order_error)
        ):
            reason = (
              ":\n Polynomial keys should be stringified, alphabetically "
             f"ordered tuples with length matching metadata.polynomial_degree "
             f"({self.polynomial_degree}) and an optional real/imaginary "
              "specifier string as the last element, of the same length."
            )
        else:
            reason = "."

        raise POPxfValidationError(msg_prefix+reason) from exception

    def check_parameter_subset(self, poly_params, poly_name):
        """
        Check that polynomial parameters are a subset of metadata.parameters.

        Validates that all parameters appearing in a polynomial are declared
        in the metadata.parameters list.

        Parameters
        ----------
        poly_params : tuple or list
            Parameters found in the polynomial (from POPxfPolynomial.parameters).
        poly_name : str
            Name/path of the polynomial field being checked (for error messages).

        Raises
        ------
        POPxfValidationError
            If any parameters in poly_params are not listed in metadata.parameters.

        """
        if not set(poly_params).issubset(self.parameters):
            diff = set(poly_params).difference(self.parameters)
            raise POPxfValidationError(
             f"'{poly_name}' contains unrecognized parameters {diff} "
              "not listed in metadata.parameters."
            )
            
class POPxfParserError(Exception):
    """
    Base exception class for POPxf JSON parsing errors.

    """

if __name__ == "__main__":
    pass
 