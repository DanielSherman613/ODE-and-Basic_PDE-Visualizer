from ode_pde_visualizer.math_tools.expression_parser import parseMathExpression


def test_parse_gradient_operator() -> None:
    parsed = parseMathExpression("grad(x^2 + y^2)")
    text = str(parsed.expr)
    assert "Derivative(x**2 + y**2, x)" not in text
    assert "x" in text and "y" in text


def test_parse_laplacian_operator() -> None:
    parsed = parseMathExpression("lap(x^2 + y^2)")
    assert str(parsed.expr) == "4"


def test_parse_divergence_operator() -> None:
    parsed = parseMathExpression("div(x, -y)")
    assert str(parsed.expr) == "0"
