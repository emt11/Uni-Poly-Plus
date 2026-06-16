import math


def normalize_mass(value, unit):
    if value is None:
        return None, None, "missing"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None, None, "invalid"
    normalized_unit = str(unit or "g/mol").lower().replace(" ", "")
    factors = {"g/mol": 1.0, "gmol-1": 1.0, "kg/mol": 1000.0, "kgmol-1": 1000.0, "kda": 1000.0, "da": 1.0}
    factor = factors.get(normalized_unit)
    if factor is None:
        return None, None, "unsupported_unit"
    return number * factor, "g/mol", "normalized"


def log_numeric_bin(kind, value):
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number <= 0:
        return None
    exponent = math.floor(math.log10(number))
    mantissa = number / (10 ** exponent)
    bucket = 1 if mantissa < 2 else 2 if mantissa < 5 else 5
    upper = {1: 2, 2: 5, 5: 10}[bucket]
    return f"{kind}_bin_{bucket}e{exponent}_{upper}e{exponent}"

