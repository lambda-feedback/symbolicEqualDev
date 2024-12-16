from .preview_utilities import (
    Params,
    Preview,
    Result,
)

from .quantity_comparison_preview import preview_function as quantity_preview
from .symbolic_comparison_preview import preview_function as symbolic_comparison_preview

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

    if params.get("text_prototype", False) is True:
        response_original = response
        if params.get("is_latex", False) is True:
            latex_array_start = r"\\begin{array}{l}\n"
            latex_array_end = r"\n\\end{array}"
            latex_array_newline = r"\\\\\n"
            response = response.replace(latex_array_start, "")
            response = response.replace(latex_array_end, "")
            response = response.replace(latex_array_newline, " ")
        result = Result(preview=Preview(latex=response_original, sympy=response))
        return result

    if params.get("physical_quantity", False):
        result = quantity_preview(response, params)
    else:
        result = symbolic_comparison_preview(response, params)

    return result
