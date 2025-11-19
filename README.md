# POPxf Parser

A lightweight Python parser and validator for POPxf (Polynomial Parameterization eXchange Format) files. POPxf is a JSON-based format for storing predictions as polynomial expansions in parameters.

## Features

- **Schema Validation**: Validates POPxf JSON files against versioned JSON schemas
- **Array Length Validation**: Verifies consistency between data arrays and metadata specifications
- **Type-Safe Polynomials**: Stores polynomial data in validated Python objects with numpy arrays
- **Comprehensive Error Messages**: Provides detailed, actionable error messages for validation failures

## Installation

```bash
# Clone the repository
git clone https://github.com/pop-xf/parser.git
cd parser

# Install dependencies
pip install numpy jsonschema
```

## Architecture

The package provides two main classes with different use cases:

### POPxfValidator
- **Purpose**: Schema validation and metadata access
- **Use when**: You only need to validate file structure and access metadata
- **Advantage**: Faster, lower memory footprint (doesn't create polynomial objects)
- **Example use case**: Batch validation of files, quick metadata extraction

### POPxfParser  
- **Purpose**: Full parsing with polynomial objects (extends POPxfValidator)
- **Use when**: You need to work with polynomial coefficients programmatically
- **Advantage**: Type-safe polynomial access with numpy arrays
- **Example use case**: Computing predictions, analyzing coefficient structure

## Quick Start

### Validation Only

```python
import json
from validator import POPxfValidator

with open('examples/BR_B0_mumu.json') as f:
    validator = POPxfValidator(json.load(f))

print(validator.info())  # Display summary
print(f"Mode: {validator.mode}")
print(f"Parameters: {validator.parameters}")
```

### Full Parsing

```python
import json
from parser import POPxfParser

with open('examples/BR_B0_mumu.json') as f:
    parser = POPxfParser(json.load(f))

# Access polynomial data
poly = parser.observable_central
print(f"Polynomial has {len(poly)} terms")
print(f"Parameters: {poly.parameters}")
```

## Command-Line Tool

The package includes a command-line tool `validate-popxf` for validating and inspecting POPxf files.

### Basic Usage

```bash
# Validate a POPxf file
python bin/validate-popxf examples/BR_B0_mumu.json

# Validate with detailed information
python bin/validate-popxf examples/BR_B0_mumu.json --verbose

# Quiet mode (only show errors)
python bin/validate-popxf examples/BR_B0_mumu.json --quiet

# Display polynomial coefficients
python bin/validate-popxf examples/BR_B0_mumu.json --show-data

# Show detailed uncertainty information
python bin/validate-popxf examples/BR_B0_mumu.json --show-uncertainties

# Combine options
python bin/validate-popxf examples/BR_B0_mumu.json --verbose --show-data --show-uncertainties
```

### Command-Line Options

| Option | Description |
|--------|-------------|
| `input_file` | Path to the POPxf JSON file to parse (required) |
| `-v`, `--verbose` | Print detailed information including reproducibility info and observable expressions |
| `-q`, `--quiet` | Suppress output (only show validation status or errors) |
| `--show-data` | Display full polynomial coefficient data |
| `--show-uncertainties` | Show detailed uncertainty information |
| `-h`, `--help` | Display help message and usage examples |

### Example Output

```bash
$ python bin/validate-popxf examples/BR_B0_mumu.json

======================================================================
Reading JSON file: examples/BR_B0_mumu.json
======================================================================
======================================================================
POPxfParser Object Properties
======================================================================

Schema version: popxf-1.0.json

metadata keys:
  - basis
  - scale
  - parameters
  - observable_names
  - reproducibility
  - misc

data keys:
  - observable_central
  - observable_uncertainties

Mode: Single-Polynomial (SP)
Polynomial Order: 2
Length (number of observables): 1
Observable Names: ['BR(B0->mumu)']
Parameters: ['C10_bdmumu', 'C10p_bdmumu']
Scale: 4.8 [GeV]

Observable Central:
  - Parameters: ('C10_bdmumu', 'C10p_bdmumu')
  - Number of polynomial terms: 9
  - Polynomial keys: [('', '', 'RR'), ('', 'C10_bdmumu', 'RR'), ('', 'C10p_bdmumu', 'RR'), ('C10_bdmumu', 'C10_bdmumu', 'II'), ('C10_bdmumu', 'C10_bdmumu', 'RR')]...

Observable Uncertainties:
  - Number of uncertainty sources: 1
  - 'total':
    Parameters: ('C10_bdmumu', 'C10p_bdmumu')
    Number of polynomial terms: 9

======================================================================

======================================================================
✓ Validation successful
======================================================================
```

### Error Handling

The tool provides detailed error messages for various failure scenarios:

```bash
# Missing file
$ python bin/validate-popxf nonexistent.json
Error: File not found: nonexistent.json

# Invalid JSON
$ python bin/validate-popxf invalid.json
Error: Invalid JSON format:
  Line 1, Column 13: Expecting value

# Schema validation error
$ python bin/validate-popxf examples/bad/missing_schema.json
======================================================================
Reading JSON file: examples/bad/missing_schema.json
======================================================================

======================================================================
POPxf Validation Error (POPxfValidationError)
======================================================================

POPxf JSON data is missing required '$schema' field.

======================================================================
```

### Exit Codes

- `0`: Validation successful
- `1`: Error occurred (file not found, invalid JSON, validation failure, etc.)

## POPxf Modes

### Single-Polynomial (SP) Mode

In SP mode, observable predictions are directly represented as polynomials in parameters

**Required fields:**
- `data.observable_central`: Polynomial coefficients for central values

**Example:**
```python
# Observable is a direct polynomial in parameters
observable = c0 + c1*C1 + c2*C2 + c3*C1*C2 + ...
```

### Function-Of-Polynomials (FOP) Mode

In FOP mode, observables are computed as functions of auxiliary polynomials.

**Required fields:**
- `metadata.polynomial_names`: Names of auxiliary polynomials
- `metadata.observable_expressions`: Expressions relating observables to polynomials
- `data.polynomial_central`: Auxiliary polynomial coefficients

**Example:**
```python
# Observables computed from auxiliary polynomials
poly1 = p0 + p1*C1 + p2*C2 + ...
poly2 = q0 + q1*C1 + q2*C2 + ...
observable = f(poly1, poly2)  # e.g., poly1/poly2
```

## Core Classes

### POPxfValidator

Base validator class that performs schema validation and mode detection without parsing polynomial data.

**Attributes:**
- `mode`: Operating mode ('SP' or 'FOP')
- `schema_version`: POPxf schema version
- `polynomial_degree`: Degree of polynomial expansion (default: 2)
- `parameters`: List of parameters
- `metadata`: Full metadata dictionary
- `data`: Raw data dictionary

**Methods:**
- `validate_schema()`: Validate JSON against POPxf schema
- `get_mode()`: Determine SP or FOP mode
- `validate_scale()`: Validate scale field consistency
- `info()`: Generate formatted summary string

**Use Case:** When you only need validation without the overhead of parsing polynomial objects.

```python
from validator import POPxfValidator

validator = POPxfValidator(json_data)
print(validator.info())  # Summary without polynomial parsing
```

### POPxfParser

Parser class that extends POPxfValidator to parse polynomial data into Python objects.

**Attributes:**
- All attributes from POPxfValidator, plus:
- `observable_central`: Central value polynomial (SP mode, POPxfPolynomial)
- `polynomial_central`: Auxiliary polynomials (FOP mode, POPxfPolynomial)
- `observable_uncertainties`: Dictionary of uncertainty polynomials (POPxfPolynomialUncertainty)

**Methods:**
- All methods from POPxfValidator, plus:
- `get_poly_data()`: Parse and validate polynomial data

**Use Case:** When you need to work with polynomial data programmatically.

```python
from parser import POPxfParser

parser = POPxfParser(json_data)
poly = parser.observable_central  # Access parsed polynomial
print(poly.parameters)  # ('C1', 'C2', ...)
```

### POPxfPolynomial

Container for polynomial data with validated keys and values.

**Key Features:**
- Keys are tuples of parameter names (must be alphabetically ordered)
- Values are 1D lists or numpy arrays of coefficients
- Supports real/imaginary component specifications
- Automatic parameter extraction

**Example:**
```python
poly_data = {
    ('', ''): [1.0, 2.0, 3.0],           # Constant term
    ('C1', ''): [0.1, 0.2, 0.3],       # Linear term
    ('C1', 'C2'): [0.01, 0.02, 0.03] # Quadratic term (ordered!)
}
poly = POPxfPolynomial(poly_data, degree=2)
print(poly.parameters)  # ('C1', 'C2')
```

### POPxfPolynomialUncertainty

Specialized class for uncertainty polynomials, supporting both parameter-dependent and parameter-independent uncertainties.

**Example:**
```python
# Parameter-independent uncertainty (constant)
unc_data = [0.1, 0.2, 0.3]
unc = POPxfPolynomialUncertainty(unc_data, degree=2)

# Parameter-dependent uncertainty
unc_data = {
    ('', ''): [0.1, 0.2, 0.3],
    ('cHq3', ''): [0.01, 0.02, 0.03]
}
unc = POPxfPolynomialUncertainty(unc_data, degree=2)
```

## Validation

The validator/parser performs multiple levels of validation:

1. **Schema Validation**: Ensures JSON structure conforms to POPxf schema
2. **Mode Detection**: Determines SP/FOP mode and checks required fields
3. **Scale Validation**: Checks consistency of scale field with mode
4. **Polynomial Format** (POPxfParser only): Validates polynomial keys (ordering, length) and values (type, shape)
5. **Array Length** (POPxfParser only): Ensures coefficient arrays match metadata specifications
6. **Parameter Consistency** (POPxfParser only): Verifies all polynomial parameters are declared

### Error Handling

```python
from validator import POPxfValidator, POPxfValidationError
from parser import POPxfParser, POPxfParserError

# Validation only
try:
    validator = POPxfValidator(data)
except POPxfValidationError as e:
    print(f"Validation failed: {e}")

# Full parsing
try:
    parser = POPxfParser(data)
except POPxfValidationError as e:
    # Validation error with detailed path and description
    print(f"Validation failed: {e}")
except POPxfParserError as e:
    # Parsing error (e.g., polynomial initialization)
    print(f"Parsing failed: {e}")
```

## Polynomial Key Format

Polynomial keys must follow specific conventions:

```python
# General format: (param1, param2, ..., paramN, [RI_specifier])
# - N parameters matching polynomial degree
# - Alphabetically ordered
# - Optional real/imaginary specifier as last element

# Valid keys (degree 2):
('', '')              # Constant term (assumed real)
('', '', 'RR')        # Constant term (explicitly real)
('C1', '')          # Linear in cHq1
('C1', 'C2')      # Quadratic (alphabetically ordered)
('C1', 'C2', 'RR') # Quadratic, both real components
('C1', 'C2', 'RI') # Quadratic, cHq1 real, cHq3 imaginary

# Invalid keys:
('C2', 'C1')      # Wrong order! Must be alphabetical
('C1', 'C2', 'C3') # Wrong degree for degree=2
```

## Project Structure

```
parser/
├── __init__.py          # Package initialization
├── validator.py         # POPxfValidator class (validation only)
├── parser.py            # POPxfParser class (extends POPxfValidator)
├── polynomial.py        # POPxfPolynomial and POPxfPolynomialUncertainty classes
├── schemas.py           # Schema definitions and validators
├── typedmapping.py      # Base class for type-validated dictionaries
├── formatting.py        # JSON formatting utilities
├── bin/                 # Command-line tools
│   └── validate-popxf   # CLI validator and inspector
├── schemas/             # JSON schema files
│   └── popxf-1.0.json
└── examples/            # Example POPxf files
    ├── BR_B0_mumu.json
    ├── BR_Bs_mumu.json
    ├── BR_Bs_mumu_B0_mumu.json
    ├── Gam_Wmunum.json
    ├── R_W_lilj.json
    └── bad/             # Invalid files for testing
        ├── missing_schema.json
        ├── bad_value_schema.json
        └── ... (other test files)
```

## Examples

### Command-Line Usage

```bash
# Quick validation
python bin/validate-popxf examples/BR_B0_mumu.json

# Detailed inspection
python bin/validate-popxf examples/Gam_Wmunum.json --verbose --show-data

# Batch validation (quiet mode)
for file in examples/*.json; do
    python bin/validate-popxf "$file" --quiet && echo "✓ $file"
done
```

### Python API - Validation Only

Use `POPxfValidator` when you only need to validate the file structure without parsing polynomials:

```python
import json
from validator import POPxfValidator

# Load and validate
with open('examples/BR_B0_mumu.json') as f:
    validator = POPxfValidator(json.load(f))

# Print summary information
print(validator.info())

# Access metadata
print(f"Mode: {validator.mode}")
print(f"Observable: {validator.metadata['observable_names'][0]}")
print(f"Parameters: {validator.parameters}")
```

### Python API - Full Parsing

Use `POPxfParser` when you need to work with polynomial data:

```python
import json
from parser import POPxfParser

# Load and parse
with open('examples/BR_B0_mumu.json') as f:
    parser = POPxfParser(json.load(f))

# Print summary information
print(parser.info())

# Access metadata (inherited from POPxfValidator)
print(f"Observable: {parser.metadata['observable_names'][0]}")
print(f"EFT Basis: {parser.metadata['basis']}")
print(f"Scale: {parser.metadata['scale']} GeV")

# Access polynomial objects (POPxfParser-specific)
poly = parser.observable_central
print(f"Polynomial has {len(poly)} terms")
print(f"Parameters used: {poly.parameters}")

for key, coeffs in poly.items():
    print(f"{key}: {coeffs}")
```

### Working with Uncertainties

```python
# Access uncertainties
if hasattr(parser, 'observable_uncertainties'):
    for source, unc_poly in parser.observable_uncertainties.items():
        print(f"\n{source} uncertainty:")
        print(f"  Parameters: {unc_poly.parameters}")
        print(f"  Terms: {len(unc_poly)}")
```

### Serialization

```python
# Convert polynomial to JSON string
json_str = poly.to_jstr()
print(json_str)

# Convert to plain dictionary
poly_dict = poly.to_dict()

# Suppress real/imaginary specifiers in output
json_str = poly.to_jstr(suppress_RI=True)
```

## Requirements

- Python 3.7+
- numpy
- jsonschema

## References

- POPxf Format Specification: [Link to specification]
- JSON Schema: https://json-schema.org/popxf-1.0/json

## Contact

For questions or issues, please open an issue on GitHub.
