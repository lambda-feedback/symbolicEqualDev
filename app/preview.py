from .utility.preview_utilities import (
    Params,
    Result,
)

from .preview_implementations.physical_quantity_preview import preview_function as physical_quantity_preview
from .preview_implementations.symbolic_preview import preview_function as symbolic_preview


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

    if params.get("physical_quantity", False):
        result = physical_quantity_preview(response, params)
    else:
        result = symbolic_preview(response, params)

    return result
