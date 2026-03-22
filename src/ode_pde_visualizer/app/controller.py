from dataclasses import dataclass, field
from typing import Protocol

from ode_pde_visualizer.core.grids import HyperGrid
from ode_pde_visualizer.core.projection import ProjectionResult, ProjectionEngine
from ode_pde_visualizer.core.time_series import PDETimeSeries
from ode_pde_visualizer.core.view_state import ViewState
from ode_pde_visualizer.rendering.color_policy import ScalarColorPolicy


@dataclass
class ViewerModel:
    grid: HyperGrid
    timeSeries: PDETimeSeries
    activeFieldName: str
    viewState: ViewState = field(default_factory=ViewState)
    colorPolicy: ScalarColorPolicy = field(default_factory=ScalarColorPolicy)


class SceneRenderer(Protocol):
    def render(self, projection: ProjectionResult,
               colorPolicy: ScalarColorPolicy) -> None:
        ...


class HyperPDEController:
    def __init__(
            self,
            model: ViewerModel,
            renderer: SceneRenderer,
            projectionEngine: ProjectionEngine | None = None,
    ) -> None:
        self.model = model
        self.renderer = renderer
        self.projectionEngine = projectionEngine or ProjectionEngine()

    def refresh(self) -> None:
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

    def scrollDimensionWindow(self, delta: int) -> None:
        self.model.viewState.dimensionWindow.scroll(delta, self.model.grid.ndim)
        self.refresh()

    def setHiddenSlice(self, axis: int, index: int) -> None:
        self.model.viewState.hiddenAxisPolicy.sliceIndices[axis] = index
        self.refresh()

    def nextFrame(self) -> None:
        maxIndex = len(self.model.timeSeries.times) - 1
        self.model.viewState.timeIndex = min(self.model.viewState.timeIndex + 1,
                                             maxIndex)
        self.refresh()

    def previousFrame(self) -> None:
        self.model.viewState.timeIndex = max(self.model.viewState.timeIndex - 1,
                                             0)
        self.refresh()

    def setReductionMode(self, mode) -> None:
        self.model.viewState.hiddenAxisPolicy.reductionMode = mode
        self.refresh()

    def setActiveField(self, fieldName: str) -> None:
        self.model.activeFieldName = fieldName
        self.refresh()
