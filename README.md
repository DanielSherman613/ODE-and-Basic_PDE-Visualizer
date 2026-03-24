# ODE-and-Basic_PDE-Visualizer

This is a ODE visualizer which can help students/self learners to better
visualize their ordinary differential and partially differential equations in
2-space or 3-space.

The user manual is in the top left corner, but to make sure everone can look and familiarize themselves with the control ahead of time:

# ODE / PDE Visualizer Manual

Welcome. This panel explains the main input rules and the safest ways to use the visualizer.
## 0. Basic operation
- Press 'f' key to zoom in the direction of the cursor
- Press 'w' to render a wire mesh
- Press 'v' or 'r' to reset the graph
- Press 'i' to hide and reveal axis
- Right click and drag for local zoom
- Left click and drag for rotation
- Mouse wheel click and drag to move fixed camera
- Left click and drag for the same effect as mouse wheel click and drag
## 1. Basic expressions

Type normal expressions such as:

- `x^2`
- `x^2 + y^2`
- `sin(x) * cos(y)`
- `exp(-t) * (x^2 + y^2)`

Use `t` for time dependent expressions.

## 2. Implicit equations

To draw shapes such as circles and spheres, use an equation with `=`:

- `x^2 + y^2 = 1`
- `x^2 + y^2 = r^2`
- `x^2 + y^2 + z^2 = r^2`

If you use a parameter such as `r`, a parameter box will appear below the preview.

## 3. Zero axis or pinned axis

To force an axis to be zero, use the format:

- `0_y`
- `0_z`
- `0_x2`

Examples:

- `x^2 + z^2 = 1 + 0_y`
- `x^2 + y^2 = r^2 + 0_z`
- `x1^2 + x3^2 = 1 + 0_x2`

This means that axis is kept in the model, but fixed at zero.

Important:
Do not freeze and vary the same axis at the same time.

Bad example:

- `x^2 + 0_x`

## 4. Higher dimensions

For higher dimensional input, the safest naming style is:

- `x1, x2, x3, x4, x5, ...`

Examples:

- `x1^2 + x2^2 + x3^2 + x4^2`
- `x1^2 + x3^2 = 1 + 0_x2`
- `x1^2 + x2^2 + x3^2 - x4 + x5`

You can also use names like `x, y, z, n`, but the indexed form is the most predictable.

## 5. Mouse wheel dimension scrolling

If your expression has more than 3 spatial axes, the mouse wheel changes which 3 axes are visible.

Examples:

- `(x+1)*(y+1)*(z+1)*(n+1)`
- `x1^2 + x2^2 + x3^2 + x4^2`

If an axis should stay fixed instead of scrolling, pin it with `0_axisName`.

## 6. Derivatives and integrals

Supported examples:

- `diff(sin(x), x)`
- `diff(x^2 * y, y)`
- `integrate(x^2, x)`
- `integrate(x*y, x)`

You can also use the operator buttons in the panel.

## 7. Numerical style operators

Available operator forms include:

- `grad(expr)`
- `lap(expr)`
- `div(Fx, Fy, Fz)`

Examples:

- `grad(x^2 + y^2 + z^2)`
- `lap(sin(x) * cos(y))`
- `div(x*y, -x*y, z)`

## 8. Systems

Use the system dropdown to switch between:

- Expression
- Heat ND
- Lorenz
- Lotka Volterra

Expression mode is for typed formulas and implicit surfaces.
The other modes run built in simulations.

## 9. Animation

Use the **Play** button to animate:
- time dependent expressions using `t`
- stored PDE frames
- ODE trajectories

## 10. Safe input tips

To avoid crashes or confusing plots:

- prefer `x1, x2, x3, ...` for many dimensions
- use `0_y`, `0_z`, or `0_x2` to pin axes cleanly
- do not use the same axis as both varying and pinned
- if you use a parameter like `r`, make sure it has a numeric value
- for spheres and circles, prefer implicit equations with `=`

## 11. Good starter examples

Try these:

- `x^2 + y^2`
- `x^2 + y^2 + z^2 = r^2`
- `x^2 + z^2 = 1 + 0_y`
- `exp(-t) * (x^2 + y^2)`
- `lap(sin(x) * cos(y))`
- `x1^2 + x3^2 = 1 + 0_x2`

Have fun.

The executable should work as expected, sorry for any lag, I haven't optimized the product to its fullest yet.
