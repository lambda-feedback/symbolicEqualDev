from .evaluation import evaluation_function

response = r"\\begin{array}{l}\na+b \\text { and more text }\\\\\n\\begin{array}{l}\nq+x \\\\\nc+d\n\\end{array}\n\\end{array}"
answer = "\\begin{array}{l}\na+b \\text { and more text }\\\\\n\\begin{array}{l}\nq+x \\\\\nc+d\n\\end{array}"
params = {
    "strict_syntax": False,
    "elementary_functions": True,
    "is_latex": True,
    "text_prototype": True,
}
result = evaluation_function(response, answer, params)
print(result)