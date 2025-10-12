import ast
import operator
from typing import Optional

# Safe eval for arithmetic expressions
# Supports +, -, *, /, **, %, parentheses, ints/floats

_ALLOWED_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

class _Eval(ast.NodeVisitor):
    def visit(self, node):
        if isinstance(node, ast.Expression):
            return self.visit(node.body)
        if isinstance(node, ast.Num):  # py<3.8 compatibility
            return node.n
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return node.value
            raise ValueError("Only numeric constants allowed")
        if isinstance(node, ast.BinOp):
            left = self.visit(node.left)
            right = self.visit(node.right)
            op = type(node.op)
            if op not in _ALLOWED_OPERATORS:
                raise ValueError("Operator not allowed")
            return _ALLOWED_OPERATORS[op](left, right)
        if isinstance(node, ast.UnaryOp):
            operand = self.visit(node.operand)
            op = type(node.op)
            if op not in _ALLOWED_OPERATORS:
                raise ValueError("Operator not allowed")
            return _ALLOWED_OPERATORS[op](operand)
        if isinstance(node, ast.Call):
            raise ValueError("Function calls not allowed")
        if isinstance(node, ast.Name):
            raise ValueError("Names not allowed")
        if isinstance(node, ast.Subscript):
            raise ValueError("Subscripts not allowed")
        raise ValueError("Unsupported expression")

def safe_eval(expr: str) -> float:
    tree = ast.parse(expr, mode="eval")
    return _Eval().visit(tree)


def try_calculate(text: str) -> Optional[float]:
    text = text.strip()
    # naive heuristic: looks like an arithmetic expression
    if any(ch in text for ch in "+-*/%") and not any(c.isalpha() for c in text):
        try:
            return safe_eval(text)
        except Exception:
            return None
    return None
