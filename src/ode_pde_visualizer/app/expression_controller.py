from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sympy import Symbol, lambdify

from ode_pde_visualizer.core.grids import AxisSpec, HyperGrid
from ode_pde_visualizer.core.time_series import PDETimeSeries
from ode_pde_visualizer.core.view_state import DimensionWindow, HiddenAxisPolicy, RenderMode, ViewState
from ode_pde_visualizer.math_tools.expression_parser import (
    ExpressionSignature,
    ParsedMathExpression,
    analyzeParsedExpression,
)


@dataclass(slots=True)
class CompiledExpression:
    parsed: ParsedMathExpression
    signature: ExpressionSignature
    symbolNames: list[str]
    numericFunction: object
    parameterValues: dict[str, float] = field(default_factory=dict)


class ExpressionController:
    def __init__(
        self,
        axisHalfSpan: float = 1.0,
        axisResolution: int = 64,
        timeMax: float = 1.0,
        timeSamples: int = 40,
    ) -> None:
        self.axisHalfSpan = float(axisHalfSpan)
        self.axisResolution = int(axisResolution)
        self.timeMax = float(timeMax)
        self.timeSamples = int(timeSamples)
        self._compiled: CompiledExpression | None = None

    def hasExpression(self) -> bool:
        return self._compiled is not None

    def clearExpression(self) -> None:
        self._compiled = None

    def setExpression(
        self,
        parsed: ParsedMathExpression,
        parameterValues: dict[str, float] | None = None,
    ) -> None:
        signature = analyzeParsedExpression(parsed)
        symbolNames = sorted(str(symbol) for symbol in parsed.expr.free_symbols)
        symbols = [Symbol(name) for name in symbolNames]
        numericFunction = lambdify(symbols, parsed.expr, modules="numpy")
        normalizedParameterValues = {
            name: float((parameterValues or {}).get(name, 1.0))
            for name in signature.parameterNames
        }
        self._compiled = CompiledExpression(
            parsed=parsed,
            signature=signature,
            symbolNames=symbolNames,
            numericFunction=numericFunction,
            parameterValues=normalizedParameterValues,
        )

    def setParameterValues(self, parameterValues: dict[str, float]) -> None:
        if self._compiled is None:
            return
        for name in self._compiled.signature.parameterNames:
            self._compiled.parameterValues[name] = float(parameterValues.get(name, 1.0))

    def currentExpressionText(self) -> str | None:
        if self._compiled is None:
            return None
        return self._compiled.parsed.rawText

    def currentSignature(self) -> ExpressionSignature | None:
        if self._compiled is None:
            return None
        return self._compiled.signature

    def currentParameterValues(self) -> dict[str, float]:
        if self._compiled is None:
            return {}
        return dict(self._compiled.parameterValues)

    def buildGrid(self) -> HyperGrid:
        signature = self._requireSignature()

        halfSpan = self.axisHalfSpan

        if self._compiled is not None and signature.isImplicitEquation:
            for parameterName in ("r", "R", "radius"):
                if parameterName in self._compiled.parameterValues:
                    radiusValue = abs(
                        float(self._compiled.parameterValues[parameterName]))
                    halfSpan = max(halfSpan, radiusValue * 1.25)
                    break

        axes: list[AxisSpec] = []

        for axisName in signature.renderAxisNames:
            if axisName.startswith("_pad"):
                coords = np.array([0.0], dtype=np.float32)
            elif axisName in signature.frozenAxes:
                coords = np.array([float(signature.frozenAxes[axisName])],
                                  dtype=np.float32)
            else:
                coords = np.linspace(
                    -halfSpan,
                    halfSpan,
                    self.axisResolution,
                    dtype=np.float32,
                )
            axes.append(AxisSpec(axisName, coords))

        return HyperGrid(axes)

    def buildTimeSeries(self, grid: HyperGrid) -> PDETimeSeries:
        signature = self._requireSignature()
        if signature.usesTime:
            times = np.linspace(0.0, self.timeMax, self.timeSamples, dtype=np.float32)
        else:
            times = np.array([0.0], dtype=np.float32)

        zeroFrame = np.zeros(grid.shape, dtype=np.float32)
        frames = [zeroFrame.copy() for _ in range(len(times))]
        return PDETimeSeries(times=times, fieldsByName={"u": frames})

    def buildViewState(self) -> ViewState:
        signature = self._requireSignature()
        renderMode = RenderMode.ISOSURFACE if signature.isImplicitEquation else RenderMode.VOLUME
        return ViewState(
            timeIndex=0,
            dimensionWindow=DimensionWindow(startAxis=0, windowSize=3, wrap=False),
            hiddenAxisPolicy=HiddenAxisPolicy(),
            renderMode=renderMode,
        )

    def evaluate(
        self,
        grid: HyperGrid,
        viewState: ViewState,
        timeValue: float,
    ) -> np.ndarray:
        if self._compiled is None:
            raise ValueError("No active expression.")

        inputMap = self._buildInputMap(grid, viewState, timeValue)
        args = [inputMap[name] for name in self._compiled.symbolNames]
        value = self._compiled.numericFunction(*args)
        array = self._coerceToGridArray(value, grid.shape)
        return np.nan_to_num(array, nan=0.0, posinf=1.0e6, neginf=-1.0e6)

    def _buildInputMap(
            self,
            grid: HyperGrid,
            viewState: ViewState,
            timeValue: float,
    ) -> dict[str, np.ndarray | float]:
        axisInputs: dict[str, np.ndarray | float] = {}
        for axisIndex, axis in enumerate(grid.axes):
            coords = np.asarray(axis.coords, dtype=np.float32)
            shape = [1] * grid.ndim
            shape[axisIndex] = coords.size
            axisInputs[axis.name] = coords.reshape(shape)

        visibleAxes = viewState.dimensionWindow.visibleAxes(grid.ndim)
        visibleAxisNames = [grid.axes[index].name for index in visibleAxes]

        if "x" not in axisInputs and visibleAxisNames:
            axisInputs["x"] = axisInputs[visibleAxisNames[0]]
        if "y" not in axisInputs and len(visibleAxisNames) > 1:
            axisInputs["y"] = axisInputs[visibleAxisNames[1]]
        if "z" not in axisInputs and len(visibleAxisNames) > 2:
            axisInputs["z"] = axisInputs[visibleAxisNames[2]]

        axisInputs["t"] = float(timeValue)

        if self._compiled is not None:
            for name, value in self._compiled.parameterValues.items():
                axisInputs[name] = float(value)

        return axisInputs

    @staticmethod
    def _coerceToGridArray(
        value: object,
        shape: tuple[int, ...],
    ) -> np.ndarray:
        if np.isscalar(value):
            return np.full(shape, float(value), dtype=np.float32)

        array = np.asarray(value, dtype=np.float32)
        if array.shape != shape:
            array = np.broadcast_to(array, shape).astype(np.float32, copy=False)
        return array

    def _requireSignature(self) -> ExpressionSignature:
        if self._compiled is None:
            raise ValueError("No active expression.")
        return self._compiled.signature
