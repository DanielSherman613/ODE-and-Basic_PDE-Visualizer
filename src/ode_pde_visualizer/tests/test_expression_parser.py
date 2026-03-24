from ode_pde_visualizer.math_tools.expression_parser import parseMathExpression, analyzeParsedExpression


def test_derivative_expression_is_parsed_and_uses_x() -> None:
    parsed = parseMathExpression("diff(sin(x), x)")
    signature = analyzeParsedExpression(parsed)
    assert parsed.expr is not None
    assert signature.spatialVariableNames == ["x"]


def test_integral_expression_is_parsed_and_uses_x() -> None:
    parsed = parseMathExpression("integrate(x^2, x)")
    signature = analyzeParsedExpression(parsed)
    assert parsed.expr is not None
    assert signature.spatialVariableNames == ["x"]


def test_implicit_equation_with_parameter_still_marks_parameter() -> None:
    parsed = parseMathExpression("x^2 + y^2 + z^2 = r^2")
    signature = analyzeParsedExpression(parsed)
    assert signature.isImplicitEquation
    assert signature.parameterNames == ["r"]
