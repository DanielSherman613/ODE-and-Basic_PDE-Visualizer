from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Protocol

from ode_pde_visualizer.app.expression_controller import ExpressionController
from ode_pde_visualizer.core.grids import HyperGrid
from ode_pde_visualizer.core.projection import ProjectionResult, ProjectionEngine
from ode_pde_visualizer.core.time_series import PDETimeSeries
from ode_pde_visualizer.core.trajectory import TrajectorySeries
from ode_pde_visualizer.core.view_state import ViewState
from ode_pde_visualizer.math_tools.expression_parser import ExpressionSignature, ParsedMathExpression
from ode_pde_visualizer.rendering.color_policy import ScalarColorPolicy


@dataclass
class ViewerModel:
    grid: HyperGrid
    timeSeries: PDETimeSeries
    activeFieldName: str
    viewState: ViewState = field(default_factory=ViewState)
    colorPolicy: ScalarColorPolicy = field(default_factory=ScalarColorPolicy)
    trajectorySeries: TrajectorySeries | None = None


class SceneRenderer(Protocol):
    def render(
        self,
        projection: ProjectionResult,
        colorPolicy: ScalarColorPolicy,
    ) -> None:
        ...

    def renderTrajectory(
        self,
        trajectory: TrajectorySeries,
        timeIndex: int,
        colorPolicy: ScalarColorPolicy,
    ) -> None:
        ...


class HyperPDEController:
    def __init__(
        self,
        model: ViewerModel,
        renderer: SceneRenderer,
        projectionEngine: ProjectionEngine | None = None,
        expressionController: ExpressionController | None = None,
    ) -> None:
        self.model = model
        self._baseModel = deepcopy(model)
        self.renderer = renderer
        self.projectionEngine = projectionEngine or ProjectionEngine()
        self.expressionController = expressionController or ExpressionController()

    def refresh(self) -> None:
        if self.expressionController.hasExpression():
            timeValue = float(self.model.timeSeries.times[self.model.viewState.timeIndex])
            field = self.expressionController.evaluate(
                grid=self.model.grid,
                viewState=self.model.viewState,
                timeValue=timeValue,
            )
            projection = self.projectionEngine.project(
                field=field,
                grid=self.model.grid,
                viewState=self.model.viewState,
            )
            self.renderer.render(projection, self.model.colorPolicy)
            return

        if self.model.trajectorySeries is not None:
            self.renderer.renderTrajectory(
                self.model.trajectorySeries,
                self.model.viewState.timeIndex,
                self.model.colorPolicy,
            )
            return

        field = self.model.timeSeries.getFieldAt(
            self.model.activeFieldName,
            self.model.viewState.timeIndex,
        )
        projection = self.projectionEngine.project(
            field=field,
            grid=self.model.grid,
            viewState=self.model.viewState,
        )
        self.renderer.render(projection, self.model.colorPolicy)

    def setExpression(
        self,
        parsed: ParsedMathExpression | None,
        parameterValues: dict[str, float] | None = None,
    ) -> None:
        if parsed is None:
            self.clearExpression()
            return

        self.expressionController.setExpression(parsed, parameterValues=parameterValues)
        self._rebuildModelForExpression()
        self.refresh()

    def updateExpressionParameters(self, parameterValues: dict[str, float]) -> None:
        self.expressionController.setParameterValues(parameterValues)
        if self.expressionController.hasExpression():
            self.refresh()

    def clearExpression(self) -> None:
        self.expressionController.clearExpression()
        self.model.grid = deepcopy(self._baseModel.grid)
        self.model.timeSeries = deepcopy(self._baseModel.timeSeries)
        self.model.activeFieldName = self._baseModel.activeFieldName
        self.model.viewState = deepcopy(self._baseModel.viewState)
        self.model.colorPolicy = deepcopy(self._baseModel.colorPolicy)
        self.model.trajectorySeries = deepcopy(self._baseModel.trajectorySeries)
        self.refresh()

    def loadModel(self, model: ViewerModel, clearExpression: bool = True) -> None:
        self.model = deepcopy(model)
        self._baseModel = deepcopy(model)
        if clearExpression:
            self.expressionController.clearExpression()
        self.refresh()

    def currentExpressionSignature(self) -> ExpressionSignature | None:
        return self.expressionController.currentSignature()

    def currentExpressionParameterValues(self) -> dict[str, float]:
        return self.expressionController.currentParameterValues()

    def scrollDimensionWindow(self, delta: int) -> None:
        self.model.viewState.dimensionWindow.scroll(delta, self.model.grid.ndim)
        self.refresh()

    def setHiddenSlice(self, axis: int, index: int) -> None:
        self.model.viewState.hiddenAxisPolicy.sliceIndices[axis] = index
        self.refresh()

    def nextFrame(self) -> None:
        maxIndex = self.frameCount() - 1
        self.model.viewState.timeIndex = min(self.model.viewState.timeIndex + 1, maxIndex)
        self.refresh()

    def previousFrame(self) -> None:
        self.model.viewState.timeIndex = max(self.model.viewState.timeIndex - 1, 0)
        self.refresh()

    def setTimeIndex(self, index: int) -> None:
        maxIndex = self.frameCount() - 1
        self.model.viewState.timeIndex = max(0, min(int(index), maxIndex))
        self.refresh()

    def frameCount(self) -> int:
        if self.model.trajectorySeries is not None and not self.expressionController.hasExpression():
            return self.model.trajectorySeries.frameCount
        return len(self.model.timeSeries.times)

    def currentTimeIndex(self) -> int:
        return int(self.model.viewState.timeIndex)

    def setReductionMode(self, mode) -> None:
        self.model.viewState.hiddenAxisPolicy.reductionMode = mode
        self.refresh()

    def setActiveField(self, fieldName: str) -> None:
        self.model.activeFieldName = fieldName
        self.refresh()

    def _rebuildModelForExpression(self) -> None:
        grid = self.expressionController.buildGrid()
        self.model.grid = grid
        self.model.timeSeries = self.expressionController.buildTimeSeries(grid)
        self.model.activeFieldName = "u"
        self.model.viewState = self.expressionController.buildViewState()
        self.model.trajectorySeries = None
