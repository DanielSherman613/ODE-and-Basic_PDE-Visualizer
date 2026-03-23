from __future__ import annotations

from dataclasses import dataclass

from sympy import Abs, E, Symbol, cos, exp, latex, log, pi, sin, sqrt, tan
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_application,
    implicit_multiplication,
    parse_expr,
    standard_transformations,
)

x = Symbol("x")
y = Symbol("y")
z = Symbol("z")
t = Symbol("t")

_TRANSFORMS = standard_transformations + (
    implicit_multiplication,
    implicit_application,
    convert_xor,
)

_ALLOWED_NAMES = {
    "x": x,
    "y": y,
    "z": z,
    "t": t,
    "sin": sin,
    "cos": cos,
    "tan": tan,
    "sqrt": sqrt,
    "exp": exp,
    "log": log,
    "ln": log,
    "abs": Abs,
    "pi": pi,
    "e": E,
}


@dataclass(slots=True)
class ParsedMathExpression:
    rawText: str
    expr: object
    latexText: str


def parseMathExpression(text: str) -> ParsedMathExpression:
    cleaned = text.strip()
    if not cleaned:
        raise ValueError("Enter an expression.")

    expr = parse_expr(
        cleaned,
        local_dict=_ALLOWED_NAMES,
        transformations=_TRANSFORMS,
        evaluate=True,
    )

    return ParsedMathExpression(
        rawText=cleaned,
        expr=expr,
        latexText=latex(expr),
    )
