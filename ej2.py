"""TP5 - Ejercicio 2: biseccion y secante para varias ecuaciones."""

from math import cos, exp

TOL = 1e-4
MAX_IT = 100


def error_relativo(nuevo, viejo):
    return abs((nuevo - viejo) / nuevo)


def biseccion(f, a, b):
    if f(a) * f(b) > 0:
        raise ValueError("f(a) y f(b) deben tener signos opuestos")
    print(f"{'n':>3} {'a':>12} {'b':>12} {'p':>12} {'f(p)':>12} {'Er':>12}")
    p_ant = None
    for n in range(1, MAX_IT + 1):
        p = (a + b) / 2
        er = error_relativo(p, p_ant) if p_ant is not None else float("nan")
        print(f"{n:>3} {a:12.8f} {b:12.8f} {p:12.8f} {f(p):12.4e} {er:12.4e}")
        if f(p) == 0 or (p_ant is not None and er < TOL):
            return p, n
        if f(a) * f(p) < 0:
            b = p
        else:
            a = p
        p_ant = p
    raise RuntimeError("no converge")


def secante(f, p0, p1):
    print(f"{'n':>3} {'p':>12} {'f(p)':>12} {'Er':>12}")
    for n in range(2, MAX_IT + 2):
        p = p1 - f(p1) * (p1 - p0) / (f(p1) - f(p0))
        er = error_relativo(p, p1)
        print(f"{n:>3} {p:12.8f} {f(p):12.4e} {er:12.4e}")
        if er < TOL:
            return p, n - 1
        p0, p1 = p1, p
    raise RuntimeError("no converge")


ECUACIONES = [
    ("a", "x - 2^(-x) = 0", lambda x: x - 2 ** (-x), (0, 1)),
    ("b", "e^x - x^2 + 3x - 2 = 0", lambda x: exp(x) - x**2 + 3 * x - 2, (0, 1)),
    ("c", "2x cos(2x) - (x+1)^2 = 0", lambda x: 2 * x * cos(2 * x) - (x + 1) ** 2, (-3, -2)),
    ("c", "2x cos(2x) - (x+1)^2 = 0", lambda x: 2 * x * cos(2 * x) - (x + 1) ** 2, (-1, 0)),
]


if __name__ == "__main__":
    resumen = []
    for nombre, texto, f, (a, b) in ECUACIONES:
        print(f"\n===== {nombre}) {texto} en [{a}, {b}] =====")
        print("\n-- Biseccion --")
        rb = biseccion(f, a, b)
        print("\n-- Secante (p0=a, p1=b) --")
        rs = secante(f, a, b)
        resumen.append((f"{nombre}) [{a}, {b}]", rb, rs))

    print("\n===== Resumen =====")
    print(f"{'':14} {'Biseccion':>22} {'Secante':>22}")
    for nombre, *rs in resumen:
        print(f"{nombre:14} " + " ".join(f"{p:12.8f} ({n:>2} it)    " for p, n in rs))
