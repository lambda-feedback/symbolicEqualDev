from typing import TypedDict

import sympy
from latex2sympy2 import latex2sympy
from sympy.printing.latex import LatexPrinter
from typing_extensions import NotRequired

from sympy.parsing.sympy_parser import T as parser_transformations
from .expression_utilities import (
    extract_latex,
    convert_absolute_notation,
    create_expression_set,
    create_sympy_parsing_params,
    latex_symbols,
    parse_expression,
    substitute_input_symbols,
    SymbolDict,
    sympy_symbols,
)

from .feedback.symbolic_equal import internal as symbolic_equal_internal_messages


class Params(TypedDict):
    is_latex: bool
    simplify: NotRequired[bool]
    symbols: NotRequired[SymbolDict]


class Preview(TypedDict):
    latex: str
    sympy: str
    feedback: str


class Result(TypedDict):
    preview: Preview


def parse_latex(response: str, symbols: SymbolDict) -> str:
    """Parse a LaTeX string to a sympy string while preserving custom symbols.

    Args:
        response (str): The LaTeX expression to parse.
        symbols (SymbolDict): A mapping of sympy symbol strings and LaTeX
        symbol strings.

    Raises:
        ValueError: If the LaTeX string or symbol couldn't be parsed.

    Returns:
        str: The expression in sympy syntax.
    """
    substitutions = {}

    for sympy_symbol_str in symbols:
        symbol_str = symbols[sympy_symbol_str]["latex"]
        latex_symbol_str = extract_latex(symbol_str)

        try:
            latex_symbol = latex2sympy(latex_symbol_str)
        except Exception:
            raise ValueError(
                f"Couldn't parse latex symbol {latex_symbol_str} "
                f"to sympy symbol."
            )

        substitutions[latex_symbol] = sympy.Symbol(sympy_symbol_str)

    try:
        expression = latex2sympy(response, substitutions)

        if isinstance(expression, list):
            expression = expression.pop()

        return str(expression.xreplace(substitutions))  # type: ignore

    except Exception as e:
        raise ValueError(str(e))


def parse_symbolic(response: str, params):
    response_list = create_expression_set(response, params)
    result_sympy_expression = []
    feedback = []
    for response in response_list:
        response = response.strip()
        response = substitute_input_symbols([response], params)
    parsing_params = create_sympy_parsing_params(params)
    parsing_params["extra_transformations"] = parser_transformations[9]  # Add conversion of equal signs
    parsing_params["symbol_dict"].update(sympy_symbols(params.get("symbols", {})))

    # Converting absolute value notation to a form that SymPy accepts
    response, response_feedback = convert_absolute_notation(response, "response")
    if response_feedback is not None:
        feedback.append(response_feedback)

    for response in response_list:
        # Safely try to parse answer and response into symbolic expressions
        try:
            if "atol" in params.keys():
                parsing_params.update({"atol": params["atol"]})
            if "rtol" in params.keys():
                parsing_params.update({"rtol": params["rtol"]})
            res = parse_expression(response, parsing_params)
        except Exception as exc:
            raise SyntaxError(symbolic_equal_internal_messages["PARSE_ERROR"](response)) from exc
        result_sympy_expression.append(res)

    return result_sympy_expression, feedback


def preview_function(response: str, params: Params) -> Result:
    """
    Function used to preview a student response.
    ---
    The handler function passes three arguments to preview_function():

    - `response` which are the answers provided by the student.
    - `params` which are any extra parameters that may be useful,
        e.g., error tolerances.

    The output of this function is what is returned as the API response
    and therefore must be JSON-encodable. It must also conform to the
    response schema.

    Any standard python library may be used, as well as any package
    available on pip (provided it is added to requirements.txt).

    The way you wish to structure you code (all in this function, or
    split into many) is entirely up to you.
    """
    symbols: SymbolDict = params.get("symbols", {})

    if not response:
        return Result(preview=Preview(latex="", sympy=""))

    try:
        if params.get("is_latex", False):
            response = parse_latex(response, symbols)

        params.update({"rationalise": False})
        expression_list, _ = parse_symbolic(response, params)

        latex_out = []
        sympy_out = []
        for expression in expression_list:
            latex_out.append(
                LatexPrinter({"symbol_names": latex_symbols(symbols)}).doprint(expression)
            )
            sympy_out.append(str(expression))

        if len(sympy_out) == 1:
            sympy_out = sympy_out[0]
        sympy_out = str(sympy_out)

        if len(latex_out) > 1:
            latex_out = "\\left\\{"+",~".join(latex_out)+"\\right\\}"
        else:
            latex_out = latex_out[0]

    except SyntaxError as e:
        raise ValueError("Failed to parse Sympy expression") from e
    except ValueError as e:
        raise ValueError("Failed to parse LaTeX expression") from e

    return Result(preview=Preview(latex=latex_out, sympy=sympy_out))
