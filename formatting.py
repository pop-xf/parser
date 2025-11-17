import numpy as np
import re

vround = np.vectorize(lambda x,y : np.round(x,y))

def signif(x, p):
    """Round a number or `numpy.array`, `x` to `p` significant digits. 
    
    Parameters
    ----------
    x : float or `numpy.array`
        Value(s) to round.
    p : int
        Number of significant digits.
    
    Returns
    -------
    float or `numpy.array`
        Rounded result.
    
    """
    x = np.asarray(x)
    x_positive = np.where(np.isfinite(x) & (x != 0), np.abs(x), 10**(p-1))
    mags = 10 ** (p - 1 - np.floor(np.log10(x_positive)))
    return np.round(x * mags) / mags

def truncate_by_error(number, error, extra_digits=0):
    """Round a number or `numpy.array`, `number` to a number of significant 
    digits, determined by the uncertainty, `error`, on it.
    
    The number of significant digits kept corresponds to the number of decimal 
    places to which the quantity is known, according to `error`.
    
    Parameters
    ----------
    number : float or `numpy.array`
        Value(s) to round.
    error : float or `numpy.array`
        Uncertainty on `number`.
    
    Optional
    --------
    grace : int, default=0
        Number of additional significant digits to keep.
    
    Returns
    -------
    float or `numpy.array`
        Rounded result.
    
    """
    is_zero = error == 0.
    error = signif(error, 1)
    order = np.where( is_zero, 0., np.floor(np.log10(error)) )
    try:
        return np.where( is_zero, number, vround(number, -np.int32(order)+extra_digits) )
    except OverflowError as e:
        print(number, order)
        raise e
################################################################################
# JSON string formatting functions
def reformat_numerical_array(s, indent=10):
    """Clean up whitespace to make nicer str representations of numerical 
    arrays in `to_json()` methods.
    
    Parameters
    ----------
    s : re.match 
        Matched pattern for a str representing a numerical array.
        
    Returns
    -------
    str
        Better formatted version of the numerical array.
    
    """
    # replace all whitespace including linebreaks with a single space
    sub1 = re.sub(r'\s+',r' ', s.group(0))
    # replace all whitespace after closing square brackets followed by a comma 
    # with a newline and indent
    return re.sub(r'\],(?!\s*\n+?)', r'],\n'+indent*' ', sub1)
 
def pretty_json_string(s, indent=4):
    """Clean up whitespace to make nicer str representations of numerical 
    arrays in string representations of json data.
    
    Parameters
    ----------
    s : str
        String representation of json data.
    
    Returns
    -------
    str
        Better formatted version of `s`.
    
    """

    reformatted_arrays = re.sub(
      r'\s*\[.*?\],{0,1}', 
      lambda s: reformat_numerical_array(s, indent=indent), s, 
      flags=re.DOTALL
    )

    nested_arrays = re.sub(
      r'\[\s*?\[', 
      r'[\n'+indent* ' '+r'[', 
      reformatted_arrays
    )
    
    pretty_jstr = re.sub(
      r'\],[\n\s]+(?=\n)', 
      r'],', 
      nested_arrays
    )

    return pretty_jstr