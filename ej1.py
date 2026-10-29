"""TP5 - Ejercicio 1: biseccion, secante y Newton para x^3 - 7x^2 + 14x - 6 = 0."""

TOL = 1e-4
MAX_IT = 100


def f(x):
    return x**3 - 7 * x**2 + 14 * x - 6


def df(x):
    return 3 * x**2 - 14 * x + 14


def error_relativo(nuevo, viejo):
    return abs((nuevo - viejo) / nuevo)


def biseccion(a, b):
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


def secante(p0, p1):
    print(f"{'n':>3} {'p':>12} {'f(p)':>12} {'Er':>12}")
    for n in range(2, MAX_IT + 2):
        p = p1 - f(p1) * (p1 - p0) / (f(p1) - f(p0))
        er = error_relativo(p, p1)
        print(f"{n:>3} {p:12.8f} {f(p):12.4e} {er:12.4e}")
        if er < TOL:
            return p, n - 1
        p0, p1 = p1, p
    raise RuntimeError("no converge")


def newton(p0):
    print(f"{'n':>3} {'p':>12} {'f(p)':>12} {'Er':>12}")
    for n in range(1, MAX_IT + 1):
        p = p0 - f(p0) / df(p0)
        er = error_relativo(p, p0)
        print(f"{n:>3} {p:12.8f} {f(p):12.4e} {er:12.4e}")
        if er < TOL:
            return p, n
        p0 = p
    raise RuntimeError("no converge")


if __name__ == "__main__":
    intervalos = {"a": (0, 1), "b": (1, 3.2), "c": (3.2, 4)}
    resumen = []
    for nombre, (a, b) in intervalos.items():
        print(f"\n===== {nombre}) [{a}, {b}] =====")
        print("\n-- Biseccion --")
        rb = biseccion(a, b)
        print("\n-- Secante (p0=a, p1=b) --")
        rs = secante(a, b)
        # Newton arranca en el punto medio: en b) desde x=3.2 diverge del intervalo
        # porque f'(3.2) = -0.08 es casi cero y la tangente salta a la raiz 2+sqrt(2)
        x0 = (a + b) / 2
        print(f"\n-- Newton (p0={x0}) --")
        rn = newton(x0)
        resumen.append((nombre, rb, rs, rn))

    print("\n===== Resumen =====")
    print(f"{'':3} {'Biseccion':>22} {'Secante':>22} {'Newton':>22}")
    for nombre, *rs in resumen:
        print(f"{nombre + ')':3} " + " ".join(f"{p:12.8f} ({n:>2} it)    " for p, n in rs))
