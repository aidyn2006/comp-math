import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import math
from typing import Dict, Tuple, Callable
from dataclasses import dataclass


@dataclass
class FitResult:
    equation_name: str
    equation_str: str
    coefficients: np.ndarray
    rss: float
    fit_func: Callable


class CurveFitter:
    def __init__(self):
        self.fitting_methods = {
            'quadratic': self._fit_quadratic,
            'linear': self._fit_linear,
            'exponential': self._fit_exponential,
            'quad_origin': self._fit_quad_origin,
            'rational': self._fit_rational
        }

    def _exp_func(self, x: np.ndarray, a: float, b: float) -> np.ndarray:
        return a * np.exp(b * x)

    def _quad_origin_func(self, x: np.ndarray, a: float, b: float) -> np.ndarray:
        return a * x + b * x ** 2

    def _rational_func(self, x: np.ndarray, a: float, b: float) -> np.ndarray:
        # Add safety check for zero values
        with np.errstate(divide='raise', invalid='raise'):
            try:
                safe_x = np.where(x != 0, x, np.inf)
                return a * x + b / safe_x
            except FloatingPointError:
                return np.full_like(x, np.inf)

    def _calculate_rss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        # Handle infinite or NaN values in prediction
        valid_mask = np.isfinite(y_pred)
        if not np.any(valid_mask):
            return float('inf')
        residuals = y_true[valid_mask] - y_pred[valid_mask]
        return math.sqrt(np.sum(residuals ** 2))

    def _fit_quadratic(self, x: np.ndarray, y: np.ndarray) -> FitResult:
        coeffs = np.polyfit(x, y, 2)
        poly = np.poly1d(coeffs)
        y_fit = poly(x)
        rss = self._calculate_rss(y, y_fit)
        equation = f"{coeffs[0]:.2f}x² + {coeffs[1]:.2f}x + {coeffs[2]:.2f}"
        return FitResult('Quadratic', equation, coeffs, rss, poly)

    def _fit_linear(self, x: np.ndarray, y: np.ndarray) -> FitResult:
        coeffs = np.polyfit(x, y, 1)
        poly = np.poly1d(coeffs)
        y_fit = poly(x)
        rss = self._calculate_rss(y, y_fit)
        equation = f"{coeffs[0]:.2f}x + {coeffs[1]:.2f}"
        return FitResult('Linear', equation, coeffs, rss, poly)

    def _fit_exponential(self, x: np.ndarray, y: np.ndarray) -> FitResult:
        try:
            coeffs, _ = curve_fit(self._exp_func, x, y)
            y_fit = self._exp_func(x, *coeffs)
            rss = self._calculate_rss(y, y_fit)
            equation = f"{coeffs[0]:.2f}e^({coeffs[1]:.2f}x)"
            return FitResult('Exponential', equation, coeffs, rss,
                             lambda x: self._exp_func(x, *coeffs))
        except:
            return FitResult('Exponential', 'Failed to fit', np.array([]), float('inf'),
                             lambda x: np.zeros_like(x))

    def _fit_quad_origin(self, x: np.ndarray, y: np.ndarray) -> FitResult:
        try:
            coeffs, _ = curve_fit(self._quad_origin_func, x, y)
            y_fit = self._quad_origin_func(x, *coeffs)
            rss = self._calculate_rss(y, y_fit)
            equation = f"{coeffs[0]:.2f}x + {coeffs[1]:.2f}x²"
            return FitResult('Quadratic through origin', equation, coeffs, rss,
                             lambda x: self._quad_origin_func(x, *coeffs))
        except:
            return FitResult('Quadratic through origin', 'Failed to fit', np.array([]), float('inf'),
                             lambda x: np.zeros_like(x))

    def _fit_rational(self, x: np.ndarray, y: np.ndarray) -> FitResult:
        # Check if there are any zero values in x
        if np.any(x == 0):
            return FitResult('Rational', 'Cannot fit - contains x=0', np.array([]), float('inf'),
                             lambda x: np.zeros_like(x))

        try:
            coeffs, _ = curve_fit(self._rational_func, x, y,
                                  bounds=([-np.inf, -1e6], [np.inf, 1e6]))
            y_fit = self._rational_func(x, *coeffs)
            rss = self._calculate_rss(y, y_fit)
            equation = f"{coeffs[0]:.2f}x + {coeffs[1]:.2f}/x"
            return FitResult('Rational', equation, coeffs, rss,
                             lambda x: self._rational_func(x, *coeffs))
        except:
            return FitResult('Rational', 'Failed to fit', np.array([]), float('inf'),
                             lambda x: np.zeros_like(x))

    def fit_all(self, x: np.ndarray, y: np.ndarray) -> Dict[str, FitResult]:
        results = {}
        for method_name, method_func in self.fitting_methods.items():
            results[method_name] = method_func(x, y)
        return results

    def plot_fits(self, x: np.ndarray, y: np.ndarray, results: Dict[str, FitResult],
                  title: str = "Curve Fitting Comparison"):
        plt.figure(figsize=(12, 8))
        plt.scatter(x, y, color='red', label='Data points')

        x_smooth = np.linspace(min(x), max(x), 100)
        for method_name, result in results.items():
            if result.rss != float('inf'):
                try:
                    y_smooth = result.fit_func(x_smooth)
                    # Only plot if the results are finite
                    if np.all(np.isfinite(y_smooth)):
                        plt.plot(x_smooth, y_smooth,
                                 label=f'{result.equation_name}: {result.equation_str}')
                except:
                    continue

        plt.legend()
        plt.title(title)
        plt.grid(True)
        plt.show()

    def find_best_fit(self, results: Dict[str, FitResult]) -> Tuple[str, FitResult]:
        # Filter out failed fits
        valid_results = {k: v for k, v in results.items() if v.rss != float('inf')}
        if not valid_results:
            return "No valid fits", FitResult("None", "No valid fits", np.array([]), float('inf'),
                                              lambda x: np.zeros_like(x))
        best_method = min(valid_results.items(), key=lambda x: x[1].rss)
        return best_method


if __name__ == "__main__":
    fitter = CurveFitter()

    test_cases = [
        (np.array([6, 7, 8, 8.5, 9, 10]), np.array([5, 5, 4, 4.5, 4, 3.3]))

    ]

    for i, (x, y) in enumerate(test_cases, 1):
        print(f"\nTest Case {i}:")
        print("=" * 50)

        results = fitter.fit_all(x, y)
        best_method, best_result = fitter.find_best_fit(results)

    print("All fits RSS values:")
    for method_name, result in results.items():
        if result.rss == float('inf'):
            print(f"{result.equation_name}: Failed to fit")
        else:
            print(f"{result.equation_name}: RSS = {result.rss:.4f}")

    print(f"\nBest fit: {best_result.equation_name}")
    print(f"Equation: y = {best_result.equation_str}")
    print(f"RSS: {best_result.rss:.4f}")

    fitter.plot_fits(x, y, results, f"Test Case {i} - Curve Fitting Comparison")
