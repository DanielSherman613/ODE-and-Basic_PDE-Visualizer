from __future__ import annotations

from dataclasses import dataclass
import re

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

_AXIS_SYMBOLS = {
    f"x{index}": Symbol(f"x{index}")
    for index in range(1, 25)
}

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
    **_AXIS_SYMBOLS,
}

_EXPLICIT_AXIS_PATTERN = re.compile(r"x([1-9][0-9]*)$")
_ALIAS_ORDER = ("x", "y", "z")


@dataclass(slots=True)
class ParsedMathExpression:
    rawText: str
    expr: object
    latexText: str


@dataclass(slots=True)
class ExpressionSignature:
    spatialVariableNames: list[str]
    usesTime: bool
    modeName: str
    renderAxisNames: list[str]
    userVariableCount: int


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


def analyzeParsedExpression(
        parsed: ParsedMathExpression,
        minimumRenderDimensions: int = 3,
) -> ExpressionSignature:
    freeSymbolNames = {str(symbol) for symbol in parsed.expr.free_symbols}
    spatialVariableNames = _orderedSpatialVariableNames(freeSymbolNames)
    usesTime = "t" in freeSymbolNames

    modeName = "PDE" if len(spatialVariableNames) > 1 else "ODE"
    renderAxisNames = spatialVariableNames.copy()

    # Keep enough axes for the 3D renderer. These pad axes are singleton axes and
    # do not represent user variables.
    while len(renderAxisNames) < minimumRenderDimensions:
        renderAxisNames.append(f"_pad{len(renderAxisNames) + 1}")

    return ExpressionSignature(
        spatialVariableNames=spatialVariableNames,
        usesTime=usesTime,
        modeName=modeName,
        renderAxisNames=renderAxisNames,
        userVariableCount=len(spatialVariableNames),
    )


def formatExpressionSignature(signature: ExpressionSignature) -> str:
    variablesText = ", ".join(signature.spatialVariableNames) or "none"
    timeText = "time dependent" if signature.usesTime else "static"
    return (
        f"Mode: {signature.modeName} | "
        f"Variables: {variablesText} | "
        f"Spatial variable count: {signature.userVariableCount} | "
        f"{timeText}"
    )


def _orderedSpatialVariableNames(freeSymbolNames: set[str]) -> list[str]:
    spatial: list[str] = []

    for alias in _ALIAS_ORDER:
        if alias in freeSymbolNames:
            spatial.append(alias)

    explicitAxes = []
    for name in freeSymbolNames:
        match = _EXPLICIT_AXIS_PATTERN.fullmatch(name)
        if match:
            explicitAxes.append((int(match.group(1)), name))

    explicitAxes.sort(key=lambda item: item[0])
    spatial.extend(name for _, name in explicitAxes)
    return spatial
