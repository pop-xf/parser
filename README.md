# POPxf Parser

A lightweight Python parser and validator for POPxf (Polynomial Parameterization eXchange Format) files. POPxf is a JSON-based format for storing predictions as polynomial expansions in parameters.

## Features

- **Schema Validation**: Validates POPxf JSON files against versioned JSON schemas
- **Dual Mode Support**: Handles both Single-Polynomial (SP) and Function-Of-Polynomials (FOP) modes
- **Type-Safe Polynomials**: Stores polynomial data in validated Python objects with numpy arrays
- **Comprehensive Error Messages**: Provides detailed, actionable error messages for validation failures
- **Parameter Checking**: Ensures all polynomial parameters are properly declared in metadata
- **Array Length Validation**: Verifies consistency between data arrays and metadata specifications

## Installation

```bash
# Clone the repository
git clone https://github.com/pop-xf/parser.git
cd parser

# Install dependencies
pip install numpy jsonschema
```

## Quick Start

```python
import json
from parser import POPxfParser

# Load a POPxf JSON file
with open('examples/BR_B0_mumu.json') as f:
    data = json.load(f)

# Parse and validate
parser = POPxfParser(data)

# Access parsed data
print(f"Mode: {parser.mode}")
print(f"Parameters: {parser.parameters}")
print(f"Observables: {parser.metadata['observable_names']}")

# Access polynomial data
if parser.mode == 'SP':
    poly = parser.observable_central
    print(f"Polynomial has {len(poly)} terms")
    print(f"Parameters used: {poly.parameters}")
```

## Command-Line Tool

The package includes a command-line tool `parse-popxf` for validating and inspecting POPxf files.

### Basic Usage

```bash
# Validate a POPxf file
python bin/parse-popxf examples/BR_B0_mumu.json

# Validate with detailed information
python bin/parse-popxf examples/BR_B0_mumu.json --verbose

# Quiet mode (only show errors)
python bin/parse-popxf examples/BR_B0_mumu.json --quiet

# Display polynomial coefficients
python bin/parse-popxf examples/BR_B0_mumu.json --show-data

# Show detailed uncertainty information
python bin/parse-popxf examples/BR_B0_mumu.json --show-uncertainties

# Combine options
python bin/parse-popxf examples/BR_B0_mumu.json --verbose --show-data --show-uncertainties
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
$ python bin/parse-popxf examples/BR_B0_mumu.json
======================================================================
POPxfParser Object Properties
======================================================================

Schema Version: popxf-1.0.json
Mode: SP
Polynomial Order: 2
Length (number of observables): 1

Parameters: ['C10_bdmumu', 'C10p_bdmumu']

Observable Names: ['BR(B0->mumu)']
Scale: 4.8 [GeV]

Observable Central (type: POPxfPolynomial):
  - Number of polynomial terms: 9
  - Parameters in polynomial: ('C10_bdmumu', 'C10p_bdmumu')

Observable Uncertainties:
  - Number of uncertainty sources: 1
  - 'total' (type: POPxfPolynomialUncertainty)

======================================================================
✓ Validation successful
======================================================================
```

### Error Handling

The tool provides detailed error messages for various failure scenarios:

```bash
# Missing file
$ python bin/parse-popxf nonexistent.json
Error: File not found: nonexistent.json

# Invalid JSON
$ python bin/parse-popxf invalid.json
Error: Invalid JSON format:
  Line 1, Column 13: Expecting value

# Schema validation error
$ python bin/parse-popxf examples/bad/missing_schema.json
======================================================================
POPxf Parser Error
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

### POPxfParser

Main parser class that validates and loads POPxf JSON files.

**Attributes:**
- `mode`: Operating mode ('SP' or 'FOP')
- `schema_version`: POPxf schema version
- `polynomial_order`: Degree of polynomial expansion (default: 2)
- `parameters`: List of EFT parameters
- `metadata`: Full metadata dictionary
- `observable_central`: Central value polynomial (SP mode)
- `polynomial_central`: Auxiliary polynomials (FOP mode)
- `observable_uncertainties`: Dictionary of uncertainty polynomials

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

The parser performs multiple levels of validation:

1. **Schema Validation**: Ensures JSON structure conforms to POPxf schema
2. **Mode Detection**: Determines SP/FOP mode and checks required fields
3. **Scale Validation**: Checks consistency of scale field with mode
4. **Polynomial Format**: Validates polynomial keys (ordering, length) and values (type, shape)
5. **Array Length**: Ensures coefficient arrays match metadata specifications
6. **Parameter Consistency**: Verifies all polynomial parameters are declared

### Error Handling

```python
from parser import POPxfParser, POPxfValidationError, POPxfParserError

try:
    parser = POPxfParser(data)
except POPxfValidationError as e:
    # Validation error with detailed path and description
    print(f"Validation failed: {e}")
except POPxfParserError as e:
    # General parsing error (e.g., missing schema)
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
├── parser.py            # Main POPxfParser class
├── polynomial.py        # POPxfPolynomial and POPxfPolynomialUncertainty classes
├── schemas.py           # Schema definitions and validators
├── typedmapping.py      # Base class for type-validated dictionaries
├── formatting.py        # JSON formatting utilities
├── bin/                 # Command-line tools
│   └── parse-popxf      # CLI validator and inspector
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
python bin/parse-popxf examples/BR_B0_mumu.json

# Detailed inspection
python bin/parse-popxf examples/Gam_Wmunum.json --verbose --show-data

# Batch validation (quiet mode)
for file in examples/*.json; do
    python bin/parse-popxf "$file" --quiet && echo "✓ $file"
done
```

### Python API - Basic Usage

```python
import json
from parser import POPxfParser

# Load and parse
with open('examples/BR_B0_mumu.json') as f:
    parser = POPxfParser(json.load(f))

# Print summary information
print(parser.info())

# Access metadata
print(f"Observable: {parser.metadata['observable_names'][0]}")
print(f"EFT Basis: {parser.metadata['basis']}")
print(f"Scale: {parser.metadata['scale']} GeV")

# Access polynomial
poly = parser.observable_central
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
