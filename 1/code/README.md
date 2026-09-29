# SYDE 572 Assignment 1: code

Requirements: Python 3, numpy, matplotlib.

Run from this folder:

    python part1_distance.py
    python part2_fitting.py

| File | Contents |
| --- | --- |
| distance_methods.py | Part 1 solvers. `find_distance_newton` (Algorithm A) and `golden_section_search` (Algorithm B) are copied from the assignment handout; the only change is an optional `history` list that records each iteration for the plots. Also: analytic parabola (cubic) solution and a grid-scan wrapper that picks a safe starting point for any curve |
| part1_distance.py | Runs the assigned problem (y = x^2 + 5, five points) and four non-polynomial curves; prints a results table and writes the plots |
| fitting_methods.py | Part 2 solvers: design matrix A, MSE with its gradient and Hessian, normal equations A^T A theta = A^T y solved by Gaussian elimination, coordinate-wise Newton (one parameter at a time), and one full multivariate Newton step |
| part2_fitting.py | Fits the line y = c0 + c1 x and the parabola y = Bx^2 + Cx + D; prints the parameters and MSE and writes the plots |
| plot_style.py | Shared plot styling and colour roles, used by every figure in both parts |

Figure conventions (both parts): black circles are given or data points, the grey line is a known curve, blue squares are Newton-Raphson (coordinate Newton in Part 2, with light-to-dark blue for successive iterations), green triangles are the second method (golden section in Part 1, full multivariate Newton in Part 2), and the orange star or line is the solution.

Part 1 figures go to `../figures/part1`:

| Figure | Purpose |
| --- | --- |
| `p1_five_points.png` | Both methods and closest-point segments for all five assigned points |
| `p1_newton_geometry.png` | Newton iterates on the curve for the representative point (-4, 0) |
| `p1_newton_convergence.png` | Newton \|D'\| and step size per iteration on a log scale |
| `p1_golden_brackets.png` | Golden-section brackets, probes, and final x* for (-4, 0) |
| `p1_convergence_comparison.png` | Newton error vs golden-section bracket width per iteration |
| `p1_non_polynomial.png` | Both methods applied to exponential, logarithmic, reciprocal, and radical curves |

Part 2 figures go to `../figures/part2`:

| Figure | Purpose |
| --- | --- |
| `p2_line_iterations.png` | The line after each coordinate Newton iteration, with the update equations |
| `p2_line_contour.png` | Coordinate Newton staircase and one full Newton step on the MSE(c0, c1) contours |
| `p2_parabola_iterations.png` | The parabola after each coordinate Newton iteration, with the update equations |
| `p2_convergence.png` | MSE - MSE* per iteration for both models |
| `p2_final_fits.png` | Best line and parabola with residuals and MSE |
