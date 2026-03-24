from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable

from sympy import Abs, E, Symbol, cos, diff, exp, integrate, latex, log, pi, \
    sin, sqrt, tan
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

_EXPLICIT_AXIS_PATTERN = re.compile(r"x([1-9][0-9]*)$")
_ZERO_AXIS_PATTERN = re.compile(r"(?<![\w.])0_([A-Za-z]\w*)(?!\w)")
_ALIAS_ORDER = ("x", "y", "z")


def _gradientMagnitude(expr):
    symbolNames = {str(symbol) for symbol in expr.free_symbols}
    axisNames = _orderedSpatialVariableNames(symbolNames)
    if not axisNames:
        raise ValueError("gradient requires at least one spatial variable.")
    return sqrt(sum(diff(expr, Symbol(name)) ** 2 for name in axisNames))


def _laplacian(expr):
    symbolNames = {str(symbol) for symbol in expr.free_symbols}
    axisNames = _orderedSpatialVariableNames(symbolNames)
    if not axisNames:
        raise ValueError("laplacian requires at least one spatial variable.")
    return sum(diff(expr, Symbol(name), 2) for name in axisNames)


def _divergence(*components):
    if not components:
        raise ValueError("divergence requires at least one component.")

    symbolNames: set[str] = set()
    for component in components:
        symbolNames.update(str(symbol) for symbol in component.free_symbols)

    axisNames = _orderedSpatialVariableNames(symbolNames)
    if len(axisNames) < len(components):
        for fallback in [*list(_ALIAS_ORDER), *(f"x{i}" for i in range(1, 25))]:
            if fallback not in axisNames:
                axisNames.append(fallback)
            if len(axisNames) >= len(components):
                break

    return sum(diff(component, Symbol(axisNames[index])) for index, component in
               enumerate(components))


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
    "diff": diff,
    "differentiate": diff,
    "integrate": integrate,
    "integral": integrate,
    "int": integrate,
    "grad": _gradientMagnitude,
    "gradient": _gradientMagnitude,
    "gradmag": _gradientMagnitude,
    "lap": _laplacian,
    "laplacian": _laplacian,
    "div": _divergence,
    "divergence": _divergence,
    **_AXIS_SYMBOLS,
}


@dataclass(slots=True)
class ParsedMathExpression:
    rawText: str
    expr: object
    latexText: str
    isImplicitEquation: bool = False
    hintSymbolNames: list[str] | None = None
    frozenAxes: dict[str, float] | None = None


@dataclass(slots=True)
class ExpressionSignature:
    spatialVariableNames: list[str]
    parameterNames: list[str]
    usesTime: bool
    modeName: str
    renderAxisNames: list[str]
    userVariableCount: int
    isImplicitEquation: bool
    frozenAxes: dict[str, float]

def _extractZeroAxes(text: str) -> tuple[str, dict[str, float]]:
    frozenAxes: dict[str, float] = {}

    def replacer(match: re.Match[str]) -> str:
        axisName = match.group(1)
        frozenAxes[axisName] = 0.0
        return "0"

    cleaned = _ZERO_AXIS_PATTERN.sub(replacer, text)
    return cleaned, frozenAxes


def parseMathExpression(text: str) -> ParsedMathExpression:
    cleaned = text.strip()
    if not cleaned:
        raise ValueError("Enter an expression.")

    cleaned, frozenAxes = _extractZeroAxes(cleaned)

    if _looksLikeEquation(cleaned):
        lhsText, rhsText = _splitEquation(cleaned)
        lhsExpr = _parseSingleExpression(lhsText)
        rhsExpr = _parseSingleExpression(rhsText)
        normalizedLhs = _normalizeExpressionText(lhsText)
        normalizedRhs = _normalizeExpressionText(rhsText)
        return ParsedMathExpression(
            rawText=text.strip(),
            expr=lhsExpr - rhsExpr,
            latexText=f"{latex(lhsExpr)} = {latex(rhsExpr)}",
            isImplicitEquation=True,
            hintSymbolNames=_extractUserSymbolNames(f"{normalizedLhs} {normalizedRhs}"),
            frozenAxes=frozenAxes,
        )

    normalized = _normalizeExpressionText(cleaned)
    expr = _parseSingleExpression(cleaned)
    return ParsedMathExpression(
        rawText=text.strip(),
        expr=expr,
        latexText=latex(expr),
        isImplicitEquation=False,
        hintSymbolNames=_extractUserSymbolNames(normalized),
        frozenAxes=frozenAxes,
    )


def analyzeParsedExpression(
    parsed: ParsedMathExpression,
    minimumRenderDimensions: int = 3,
) -> ExpressionSignature:
    freeSymbolNames = {str(symbol) for symbol in parsed.expr.free_symbols}
    hintSymbolNames = set(parsed.hintSymbolNames or [])
    frozenAxisNames = set((parsed.frozenAxes or {}).keys())

    conflicts = frozenAxisNames & freeSymbolNames
    if conflicts:
        joined = ", ".join(sorted(conflicts))
        raise ValueError(f"Axis cannot be both frozen and varying: {joined}")

    allRelevantSymbolNames = freeSymbolNames | hintSymbolNames | frozenAxisNames
    usesTime = "t" in allRelevantSymbolNames

    if parsed.isImplicitEquation:
        spatialVariableNames = _orderedImplicitSpatialVariableNames(allRelevantSymbolNames)
        parameterNames = sorted(
            name for name in allRelevantSymbolNames
            if name not in set(spatialVariableNames) and name != "t"
        )
        modeName = "Implicit"
    else:
        spatialVariableNames = _orderedSpatialVariableNames(allRelevantSymbolNames)
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
        frozenAxes=dict(parsed.frozenAxes or {}),
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
    normalizedText = _normalizeExpressionText(text)
    return parse_expr(
        normalizedText,
        local_dict=_ALLOWED_NAMES,
        transformations=_TRANSFORMS,
        evaluate=True,
    )


def _normalizeExpressionText(text: str) -> str:
    normalized = text.strip()
    normalized = normalized.replace("∂/∂", "d/d")
    normalized = normalized.replace("∫", "integrate")
    normalized = _rewriteLeibnizDerivatives(normalized)
    return normalized


def _rewriteLeibnizDerivatives(text: str) -> str:
    index = 0
    pieces: list[str] = []
    length = len(text)

    while index < length:
        if text.startswith("d/d", index):
            variableStart = index + 3
            variableEnd = variableStart
            while variableEnd < length and (
                    text[variableEnd].isalnum() or text[variableEnd] == "_"):
                variableEnd += 1

            if variableEnd == variableStart or variableEnd >= length or text[
                variableEnd] != "(":
                pieces.append(text[index])
                index += 1
                continue

            depth = 0
            bodyStart = variableEnd + 1
            cursor = variableEnd
            while cursor < length:
                char = text[cursor]
                if char == "(":
                    depth += 1
                elif char == ")":
                    depth -= 1
                    if depth == 0:
                        bodyText = text[bodyStart:cursor]
                        variableName = text[variableStart:variableEnd]
                        rewrittenBody = _rewriteLeibnizDerivatives(bodyText)
                        pieces.append(f"diff({rewrittenBody}, {variableName})")
                        index = cursor + 1
                        break
                cursor += 1
            else:
                pieces.append(text[index])
                index += 1
                continue

            continue

        pieces.append(text[index])
        index += 1

    return "".join(pieces)


def _looksLikeEquation(text: str) -> bool:
    return text.count(
        "=") == 1 and "==" not in text and "<=" not in text and ">=" not in text


def _splitEquation(text: str) -> tuple[str, str]:
    lhsText, rhsText = text.split("=", maxsplit=1)
    lhsText = lhsText.strip()
    rhsText = rhsText.strip()
    if not lhsText or not rhsText:
        raise ValueError("Both sides of the equation must be non empty.")
    return lhsText, rhsText


def _extractUserSymbolNames(text: str) -> list[str]:
    tokenPattern = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
    reserved = {
        "sin", "cos", "tan", "sqrt", "exp", "log", "ln", "abs",
        "pi", "e", "diff", "differentiate", "integrate", "integral", "int",
        "grad", "gradient", "gradmag", "lap", "laplacian", "div", "divergence",
    }
    seen: list[str] = []
    for token in tokenPattern.findall(text):
        if token in reserved:
            continue
        if token not in seen:
            seen.append(token)
    return seen


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


def _orderedImplicitSpatialVariableNames(freeSymbolNames: set[str]) -> list[
    str]:
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
