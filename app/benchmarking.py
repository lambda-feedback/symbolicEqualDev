benchmarks = [
    {
        "response": "2a",
        "answer": "a",
        "params": {
            "strict_syntax": False,
            "elementary_functions": True,
            "criteria": "response/answer=2",
        }
    },
    {
        "response": "2*x**2 = 10*y**2+20",
        "answer": "x**2-5*y**2-10=0",
        "params": {"strict_syntax": False}
    },
    {
        "response": "1.24 mile/hour",
        "answer": "1.24 mile/hour",
        "params": {
            "strict_syntax": False,
            "elementary_functions": True,
            "physical_quantity": True,
        }
    },
    {
        "response": "sin(x)+2",
        "answer": "sin(x)",
        "params": {
            "strict_syntax": False,
            "elementary_functions": True,
            "criteria": "Derivative(response,x)=cos(x)",
        }
    },
    {
        "response": "cos(x)**2 + sin(x)**2 + y",
        "answer": "y + 1",
        "params": {"strict_syntax": False}
    },
    {
    "response": "log(2)/2+I*(3*pi/4 plus_minus 2*n*pi)",
    "answer": "log(2)/2+I*(3*pi/4 plus_minus 2*n*pi)",
    "params": {
            "strict_syntax": False,
            "elementary_functions": True,
        }
    }
]