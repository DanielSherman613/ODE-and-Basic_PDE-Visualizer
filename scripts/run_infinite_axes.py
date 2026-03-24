from math import inf

from ode_pde_visualizer.core.graph_settings import InfiniteAxesSettings
from ode_pde_visualizer.rendering.infinite_axes_renderer import InfiniteAxesRenderer


def main() -> None:
    settings = InfiniteAxesSettings(
        minAxisHalfSpan=12.0,
        maxAxisHalfSpan=inf,
        axisScaleWithDistance=6.0,
        xlabel="x",
        ylabel="y",
        zlabel="z",
    )

    renderer = InfiniteAxesRenderer(settings)
    renderer.render()
    renderer.show()


if __name__ == "__main__":
    main()