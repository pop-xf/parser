import re
import sys
import numpy as np
import json
import jsonschema
from io import StringIO
from ast import literal_eval
from jsonschema.exceptions import ValidationError
from schemas import schemas, validators
from polynomial import POPxfPolynomial, POPxfPolynomialUncertainty
from validator import POPxfValidator
# TODO:
# implement evaluate()
# implement serialization back to JSON
# implement uncertainty treatment to get covariance matrices, etc.
# splitting a file into multiple files

# change to Validator
# modify info() to not use popxf polynomial

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

        # set polynomial data, performs additional validation
        self.set_poly_data()

    def set_poly_data(self):
        """
        Parse and validate polynomial data from the JSON data section.

        Converts polynomial data dictionaries into POPxfPolynomial and 
        POPxfPolynomialUncertainty objects, performing comprehensive validation
        beyond what the JSON schema provides. The specific fields parsed depend
        on the operating mode.

        Raises
        ------
        POPxfValidationError
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
                  length=len(self.metadata["polynomial_names"])
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


            
class POPxfParserError(Exception):
    """
    Base exception class for POPxf JSON parsing errors.

    """

if __name__ == "__main__":
    # pass
    # import sys
    # example = json.load(open('examples/Gam_Wmunum.json'))

    # example = json.load(open('examples/R_W_lilj.json'))
    # example = json.load(open('examples/BR_Bs_mumu_B0_mumu.json'))
    # example = json.load(open('examples/BR_Bs_mumu.json'))
    # example = json.load(open('examples/BR_B0_mumu.json'))
    # example = json.load(open('examples/bad/missing_polynomial_names.json'))
    # example = json.load(open('examples/bad/missing_observable_expressions.json'))
    # example = json.load(open('examples/bad/bad_length_observable_central.json'))
    example = json.load(open('examples/bad/bad_keys_observable_uncertainties.json'))
    example = json.load(open('examples/bad/bad_observable_central_scale_array_FOP.json'))
    example = json.load(open('examples/bad/bad_observable_uncertainties_scale_array_FOP.json'))
    example = json.load(open('examples/bad/missing_tool_name.json'))
   # 
    # from glob import glob
    # bad_files = glob('examples/bad/*.json')
    guy = POPxfParser(example)
    print(guy.parameters)
    
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
