import re
import sys
import numpy as np
import json
import jsonschema
from io import StringIO
from jsonschema.exceptions import ValidationError
from schemas import schemas, validators
from polynomial import POPxfPolynomial, POPxfPolynomialUncertainty

class POPxfParser(object):
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
    polynomial_order : int
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

        self.json = json_data

        try:
            self.schema_version = json_data["$schema"].split('/')[-1]
        except KeyError as e:
            raise POPxfParserError(
              "POPxf JSON data is missing required '$schema' field."
            ) from None
        
        try:
            self.json_schema = schemas[self.schema_version]
        except KeyError:
            allowed_schemas = ', '.join([ f"'{v}'" for v in schemas.keys() ])
            raise POPxfParserError(
                f"POPxf JSON schema version '{self.schema_version}' is not "
                "recognized. Available versions are: "
                f"{allowed_schemas}"
            ) from None

        self.validator = validators[self.schema_version](self.json_schema)

        # validate against schema
        self.validate_schema()

        # get metadata and data fields
        self.metadata = self.json["metadata"]
        self.data = self.json["data"]
        self.polynomial_order = self.metadata.get("polynomial_order", 2)

        # determine mode
        self.mode = self.get_mode()

        # length of observable
        self.length = len(self.metadata["observable_names"])

        # parameters
        self.parameters = self.metadata["parameters"]

        # validate scale field
        self.validate_scale(self.mode)

        # set polynomial data
        self.set_poly_data()

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

            raise POPxfValidationError(
              "POPxf JSON data does not conform to schema version "
             f"{self.schema_version}:\n{error_message}"
            ) from None
    
    def get_mode(self):
        """
        Determine the operating mode of the POPxf JSON data.

        Detects whether the data is in Single-Polynomial (SP) or 
        Function-Of-Polynomials (FOP) mode based on the presence of 
        mode-specific fields. Also validates that all required fields for
        the detected mode are present.

        Returns
        -------
        str
            Operating mode: 'SP' for Single-Polynomial mode or 'FOP' for
            Function-Of-Polynomials mode.

        Raises
        ------
        POPxfValidationError
            If required fields for the detected mode are missing, or if the
            mode cannot be determined from the available fields.

        Notes
        -----
        **FOP mode** is detected if any of these fields are present:
        - metadata['polynomial_names']
        - metadata['observable_expressions']
        - data['polynomial_central']
        
        If FOP mode is detected, all of the above fields must be present.
        
        **SP mode** is assumed if 'observable_central' is present in data
        and no FOP-specific fields are found.
        
        The mode determines how observables are computed:
        - SP: Direct polynomial representation in 'observable_central'
        - FOP: Computed from 'polynomial_central' using expressions in
          'observable_expressions'

        """
        # required fields for FOP mode
        FOP_metadata_fields = [ "polynomial_names", "observable_expressions"]
        FOP_data_fields = [ "polynomial_central" ]

        if (
          any(fld in self.metadata for fld in FOP_metadata_fields) or 
          any(fld in self.data for fld in FOP_data_fields)
        ):
            # any of the required FOP fields are present
            mode = 'FOP'
            # check that all required fields are present
            missing_fields_metadata = [
              fld for fld in FOP_metadata_fields
              if fld not in self.metadata
            ] 
            missing_fields_data = [
              fld for fld in FOP_data_fields
              if fld not in self.data
            ]
            if missing_fields_metadata:
                raise POPxfValidationError(
                  "POPxf JSON data is missing required metadata fields for "
                  "function-of-polynomials (FOP) mode: "
                 f"{', '.join(missing_fields_metadata)}"
                )
            if missing_fields_data:
                raise POPxfValidationError(
                  "POPxf JSON data is missing required data fields for "
                  "function-of-polynomials (FOP) mode: "
                 f"{', '.join(missing_fields_data)}"
                )
        elif "observable_central" in self.data:
            mode = 'SP'
        else:
            raise POPxfValidationError(
              "POPxf JSON data is missing required metadata keys to "
              "determine mode (either SP or FOP). See documentation."
            )
            
        return mode
    
    def validate_scale(self, mode):
        """
        Validate the 'scale' field in metadata for consistency with mode and data.

        The scale field can be either a single number or an array. Array-valued
        scales have different length requirements and compatibility constraints
        depending on the operating mode.

        Parameters
        ----------
        mode : str
            Operating mode ('SP' or 'FOP').

        Raises
        ------
        POPxfValidationError
            If the scale field is inconsistent with the mode or data fields:
            - In SP mode: array length must match observable_central length
            - In FOP mode: array length must match polynomial_names length
            - In FOP mode with array scale: observable_central must be absent
            - In FOP mode with array scale: observable_uncertainties must be
              parameter-independent only

        Notes
        -----
        **Scalar scale**: A single number applies to all observables or polynomials.
        
        **Array scale in SP mode**: Each element corresponds to an observable in
        observable_central.
        
        **Array scale in FOP mode**: Each element corresponds to a polynomial in
        polynomial_names. In this case:
        - Observable_central cannot be present (observables computed from polynomials)
        - Observable_uncertainties can only contain parameter-independent terms
        
        The parameter-independence check examines polynomial keys to ensure they
        only contain empty strings and optional real/imaginary specifiers.

        See Also
        --------
        get_mode : Determine the operating mode
        """
        scale = self.metadata["scale"]

        if isinstance(scale, (int, float)):
            # scale is just a number
            return None
        else:
            # scale is array-valued
            if mode == 'SP':
                # length matches observable_central
                if len(scale) != len(self.data["observable_central"]):
                    raise POPxfValidationError(
                      "Lengths of array-valued 'scale' and "
                      "'observable_central' metadata fields must match in "
                      "single-polynomial (SP) mode."
                    )
            elif mode == 'FOP':
                # length matches polynomial_names
                if len(scale) != len(self.metadata["polynomial_names"]):
                    raise POPxfValidationError(
                      "Lengths of array-valued 'scale' and "
                      "'polynomial_names' metadata fields must match in "
                      "function-of-polynomials (FOP) mode."
                    )
                # observable_central must be absent
                if "observable_central" in self.data:
                    raise POPxfValidationError(
                      "'observable_central' data field must be absent "
                      "when 'scale' is array-valued in function-of-"
                      "polynomials (FOP) mode."
                    )
                # observable_uncertainties must be absent or only specify 
                # parameter-independent uncertainty
                for key in self.data.get( 
                  "observable_uncertainties", 
                  {} # skip if absent
                ).keys():
                    # check if key is parameter independent
                    is_param_indep = set(
                      [x for y in eval(key) for x in y]
                    ).issubset({"I","R"}) 
                    # could technically remove "I" as that would be a 
                    # strange parameter independent specification

                    if not is_param_indep:
                        example = ", ".join(["''"]*self.polynomial_order)
                        example_2 = example + ", " + 'R'*self.polynomial_order
                        raise POPxfValidationError(
                          "'observable_uncertainties' data field must be "
                          "absent or only specify parameter-independent "
                          "uncertainty when 'scale' is array-valued in "
                          "function-of-polynomials (FOP) mode. For example, "
                         f"({example}) or ({example_2})."
                        )
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
              ":\n Polynomial keys should be stringified tuples with "
             f"length matching metadata.polynomial_degree "
             f"({self.polynomial_order}) and an optional real/imaginary "
              "specifier string of the same length as the last element."
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
           - Keys must be tuples matching polynomial_order
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
                  degree=self.polynomial_order,
                  length=self.length
                )
            except (POPxfPolynomial.init_error) as e:
                msg = "Error initialising 'observable_central' polynomial data"
                self.raise_polynomial_error(e, msg)

            self.check_parameter_subset(
              observable_central.parameters, 
              "data['observable_central']"
            )
            
            self.observable_central = observable_central

        elif self.mode == 'FOP':
            # function-of-polynomials mode
            # validate polynomial_central
            try:
                polynomial_central = POPxfPolynomial(
                  self.data["polynomial_central"],
                  degree=self.polynomial_order,
                  length=len(self.metadata["polynomial_names"])
                )
            except (POPxfPolynomial.init_error) as e:
                msg = "Error initialising data['polynomial_central'] polynomial data"
                self.raise_polynomial_error(e, msg)

            self.check_parameter_subset(
              polynomial_central.parameters, 
              "data['polynomial_central']"
            )
            
            self.polynomial_central = polynomial_central

        # validate observable_uncertainties if present
        if "observable_uncertainties" in self.data:

            self.observable_uncertainties = {}
            for k,v in self.data["observable_uncertainties"].items():
                try:
                    observable_uncertainty = POPxfPolynomialUncertainty(
                      v,
                      degree=self.polynomial_order,
                      length=self.length 
                    )
                    
                    self.check_parameter_subset(
                    observable_uncertainty.parameters, 
                    f"data['observable_uncertainties'][{k}]"
                    )

                    self.observable_uncertainties[k] = observable_uncertainty

                except (POPxfPolynomialUncertainty.init_error) as e:
                    msg = (
                      f"Error initialising '{k}' entry of "
                      f"data['observable_uncertainties'] polynomial data."
                    )
                    self.raise_polynomial_error(e, msg)

    def info(self):
        """
        Generate a summary string of the POPxfParser object's properties.

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
        result.write("POPxfParser Object Properties\n")
        result.write("=" * 70 + "\n")
        
        result.write(f"\nSchema version: {self.schema_version}\n")

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

        result.write(f"Polynomial Order: {self.polynomial_order}\n")
        result.write(f"Length (number of observables): {self.length}\n")        
        result.write(f"Observable Names: {self.metadata['observable_names']}\n")
        result.write(f"Parameters: {self.parameters}\n")
        result.write(f"Scale: {self.metadata['scale']} [GeV]\n")
        if self.mode == 'FOP':
            result.write("\nFOP data:\n")
            result.write(f"  - Number of polynomials: {len(self.metadata.get('polynomial_names', []))}\n")
            result.write("  - Polynomial names: " + ", ".join(self.metadata.get('polynomial_names', [])) + "\n")
            result.write(f"\nPolynomial Central (type: {type(self.polynomial_central).__name__}):\n")
            result.write(f"  - Number of polynomial terms: {len(self.polynomial_central)}\n")
            result.write(f"  - Parameters in polynomial: {self.polynomial_central.parameters}\n")

        if hasattr(self, 'observable_central'):
            result.write(f"\nObservable Central (type: {type(self.observable_central).__name__}):\n")
            result.write(f"  - Number of polynomial terms: {len(self.observable_central)}\n")
            result.write(f"  - Parameters in polynomial: {self.observable_central.parameters}\n")
            result.write(f"  - Polynomial keys: {list(self.observable_central.keys())[:5]}{'...' if len(self.observable_central) > 5 else ''}\n")
            
        if hasattr(self, 'observable_uncertainties'):
            result.write(f"\nObservable Uncertainties:\n")
            result.write(f"  - Number of uncertainty sources: {len(self.observable_uncertainties)}\n")
            for unc_name, unc_obj in self.observable_uncertainties.items():
                result.write(f"  - '{unc_name}' (type: {type(unc_obj).__name__})\n")
                result.write(f"    Parameters: {unc_obj.parameters}\n")
                result.write(f"    Number of terms: {len(unc_obj)}\n")
        
        result.write("\n" + "=" * 70)

        return result.getvalue()
            
class POPxfParserError(Exception):
    """
    Base exception class for POPxf JSON parsing errors.

    This is the base class for all exceptions raised during POPxf file parsing.
    It is a direct subclass of the built-in Exception class.

    Notes
    -----
    Specific error types (e.g., validation errors) are implemented as subclasses
    of this base exception. Catching POPxfParserError will catch all POPxf-related
    errors.

    See Also
    --------
    POPxfValidationError : Subclass for validation-specific errors
    """

class POPxfValidationError(POPxfParserError):
    """
    Exception raised for POPxf JSON validation errors.

    This exception is raised when POPxf JSON data fails validation checks,
    including schema validation, mode-specific constraints, array length
    mismatches, parameter consistency, or polynomial format errors.

    Notes
    -----
    This is a subclass of POPxfParserError. Validation errors include:
    
    - Schema conformance failures (missing fields, wrong types, pattern mismatches)
    - Mode detection and mode-specific field requirements
    - Array length inconsistencies between data and metadata
    - Unrecognized parameters not listed in metadata.parameters
    - Invalid polynomial keys or values
    - Scale field inconsistencies

    Error messages are designed to be informative, including the path to the
    problematic field and specific guidance on how to fix the issue.

    See Also
    --------
    POPxfParserError : Base exception class
    POPxfParser.validate_schema : Schema validation method
    POPxfParser.get_validation_error_message : Error message formatting
    """
    pass

if __name__ == "__main__":
    # import sys
    # example = json.load(open('examples/Gam_Wmunum.json'))

    example = json.load(open('examples/R_W_lilj.json'))
    # example = json.load(open('examples/BR_Bs_mumu_B0_mumu.json'))
    # example = json.load(open('examples/BR_Bs_mumu.json'))
    # example = json.load(open('examples/BR_B0_mumu.json'))
    
    # from glob import glob
    # bad_files = glob('examples/bad/*.json')
    guy = POPxfParser(example)
    
    print(guy.info())

