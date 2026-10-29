"""TP5 - Ejercicio 3: aproximar sqrt(3) con el metodo de la secante.

sqrt(3) es la raiz positiva de f(x) = x^2 - 3. Un error porcentual de 0.01% equivale
a un error relativo de 10^-4.
"""

from math import sqrt

TOL_PORCENTUAL = 0.01
MAX_IT = 100


def f(x):
    return x**2 - 3


def secante(p0, p1):
    print(f"{'n':>3} {'p':>12} {'f(p)':>12} {'Er %':>12}")
    for n in range(2, MAX_IT + 2):
        p = p1 - f(p1) * (p1 - p0) / (f(p1) - f(p0))
        er = abs((p - p1) / p) * 100
        print(f"{n:>3} {p:12.8f} {f(p):12.4e} {er:12.4e}")
        if er < TOL_PORCENTUAL:
            return p, n - 1
        p0, p1 = p1, p
    raise RuntimeError("no converge")


if __name__ == "__main__":
    # f(1) = -2 < 0 y f(2) = 1 > 0, asi que sqrt(3) esta en [1, 2]
    p, it = secante(1, 2)
    print(f"\nsqrt(3) ~ {p:.8f} en {it} iteraciones")
    print(f"Valor exacto: {sqrt(3):.8f} · error porcentual real: {abs(p - sqrt(3)) / sqrt(3) * 100:.2e} %")
