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
    isImplicitEquation: bool = False


@dataclass(slots=True)
class ExpressionSignature:
    spatialVariableNames: list[str]
    parameterNames: list[str]
    usesTime: bool
    modeName: str
    renderAxisNames: list[str]
    userVariableCount: int
    isImplicitEquation: bool


def parseMathExpression(text: str) -> ParsedMathExpression:
    cleaned = text.strip()
    if not cleaned:
        raise ValueError("Enter an expression.")

    if _looksLikeEquation(cleaned):
        lhsText, rhsText = _splitEquation(cleaned)
        lhsExpr = _parseSingleExpression(lhsText)
        rhsExpr = _parseSingleExpression(rhsText)
        return ParsedMathExpression(
            rawText=cleaned,
            expr=lhsExpr - rhsExpr,
            latexText=f"{latex(lhsExpr)} = {latex(rhsExpr)}",
            isImplicitEquation=True,
        )

    expr = _parseSingleExpression(cleaned)
    return ParsedMathExpression(
        rawText=cleaned,
        expr=expr,
        latexText=latex(expr),
        isImplicitEquation=False,
    )


def analyzeParsedExpression(
    parsed: ParsedMathExpression,
    minimumRenderDimensions: int = 3,
) -> ExpressionSignature:
    freeSymbolNames = {str(symbol) for symbol in parsed.expr.free_symbols}
    usesTime = "t" in freeSymbolNames

    if parsed.isImplicitEquation:
        spatialVariableNames = _orderedImplicitSpatialVariableNames(freeSymbolNames)
        parameterNames = sorted(
            name for name in freeSymbolNames
            if name not in set(spatialVariableNames) and name != "t"
        )
        modeName = "Implicit"
    else:
        spatialVariableNames = _orderedSpatialVariableNames(freeSymbolNames)
        parameterNames = []
        modeName = "PDE" if len(spatialVariableNames) > 1 else "ODE"

    renderAxisNames = spatialVariableNames.copy()
    while len(renderAxisNames) < minimumRenderDimensions:
        renderAxisNames.append(f"_pad{len(renderAxisNames) + 1}")

    return ExpressionSignature(
        spatialVariableNames=spatialVariableNames,
        parameterNames=parameterNames,
        usesTime=usesTime,
        modeName=modeName,
        renderAxisNames=renderAxisNames,
        userVariableCount=len(spatialVariableNames),
        isImplicitEquation=parsed.isImplicitEquation,
    )


def formatExpressionSignature(signature: ExpressionSignature) -> str:
    variablesText = ", ".join(signature.spatialVariableNames) or "none"
    parameterText = ", ".join(signature.parameterNames) or "none"
    timeText = "time dependent" if signature.usesTime else "static"
    relationText = "implicit equation" if signature.isImplicitEquation else "expression"
    return (
        f"Mode: {signature.modeName} | "
        f"Type: {relationText} | "
        f"Variables: {variablesText} | "
        f"Parameters: {parameterText} | "
        f"Spatial variable count: {signature.userVariableCount} | "
        f"{timeText}"
    )


def _parseSingleExpression(text: str):
    return parse_expr(
        text,
        local_dict=_ALLOWED_NAMES,
        transformations=_TRANSFORMS,
        evaluate=True,
    )


def _looksLikeEquation(text: str) -> bool:
    return text.count("=") == 1 and "==" not in text and "<=" not in text and ">=" not in text


def _splitEquation(text: str) -> tuple[str, str]:
    lhsText, rhsText = text.split("=", maxsplit=1)
    lhsText = lhsText.strip()
    rhsText = rhsText.strip()
    if not lhsText or not rhsText:
        raise ValueError("Both sides of the equation must be non empty.")
    return lhsText, rhsText


def _orderedSpatialVariableNames(freeSymbolNames: set[str]) -> list[str]:
    spatial: list[str] = []
    consumed: set[str] = set()

    for alias in _ALIAS_ORDER:
        if alias in freeSymbolNames:
            spatial.append(alias)
            consumed.add(alias)

    explicitAxes = []
    for name in freeSymbolNames:
        match = _EXPLICIT_AXIS_PATTERN.fullmatch(name)
        if match:
            explicitAxes.append((int(match.group(1)), name))
            consumed.add(name)

    explicitAxes.sort(key=lambda item: item[0])
    spatial.extend(name for _, name in explicitAxes)

    remainingNames = sorted(
        name for name in freeSymbolNames
        if name not in consumed and name != "t"
    )
    spatial.extend(remainingNames)
    return spatial


def _orderedImplicitSpatialVariableNames(freeSymbolNames: set[str]) -> list[str]:
    spatial: list[str] = []
    consumed: set[str] = set()

    for alias in _ALIAS_ORDER:
        if alias in freeSymbolNames:
            spatial.append(alias)
            consumed.add(alias)

    explicitAxes = []
    for name in freeSymbolNames:
        match = _EXPLICIT_AXIS_PATTERN.fullmatch(name)
        if match:
            explicitAxes.append((int(match.group(1)), name))
            consumed.add(name)

    explicitAxes.sort(key=lambda item: item[0])
    spatial.extend(name for _, name in explicitAxes)

    if not spatial:
        fallbackNames = sorted(name for name in freeSymbolNames if name != "t")
        spatial.extend(fallbackNames[:3])

    return spatial
