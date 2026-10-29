"""Metodos para raices: biseccion, posicion falsa, secante y Newton.

USO EN EL PARCIAL: editar solo el bloque CONFIGURACION de abajo y correr
    python3 metodos.py

La ecuacion se puede escribir casi como en el enunciado:
    ^  -> potencia          2x, 3(x+1), x(x-1)  -> multiplicacion implicita
    sen, sin, cos, tan      ln (= log natural), log10, exp, sqrt
    e  -> numero e          pi -> numero pi       e^x o exp(x)
Los extremos del intervalo tambien pueden ser texto: ("e", 4), ("-pi", 0)
"""

import math
import re

# ============================ CONFIGURACION ============================
F = "x^3 - 7x^2 + 14x - 6"
DF = None  # derivada para Newton; None = se calcula numericamente
INTERVALOS = [(0, 1), (1, 3.2), (3.2, 4)]
TOL = 1e-4
TIPO_ERROR = "relativo"  # "relativo" (ej. 1e-4) o "porcentual" (ej. 0.01 para 0.01%)
METODOS = ["biseccion", "posicion_falsa", "secante", "newton"]
X0_NEWTON = None  # None = punto medio de cada intervalo
VER_ITERACIONES = True
# =======================================================================

MAX_IT = 200

_NOMBRES = {
    "sin": math.sin, "sen": math.sin, "cos": math.cos, "tan": math.tan, "tg": math.tan,
    "asin": math.asin, "acos": math.acos, "atan": math.atan,
    "exp": math.exp, "ln": math.log, "log": math.log, "log10": math.log10,
    "sqrt": math.sqrt, "abs": abs, "e": math.e, "pi": math.pi,
}


def _normalizar(texto):
    t = texto.lower().replace("^", "**").replace("·", "*")
    t = re.sub(r"(?<=[a-z)])\.(?=[a-z(])", "*", t)  # 2x.cos(2x)
    t = t.replace(",", ".")
    t = re.sub(r"\b(sen|sin|cos|tan|tg|ln|log|exp|sqrt)\s+(x|\d+(?:\.\d+)?)\b", r"\1(\2)", t)  # ln x
    t = re.sub(r"\b(\d+(?:\.\d+)?)\s*(?=[a-zA-Z(])", r"\1*", t)  # 2x, 3(x+1), 2cos(x)
    t = re.sub(r"\)\s*(?=[\w(])", ")*", t)                        # (x+1)(x-1), (x+1)x
    t = re.sub(r"\bx\b\s*(?=[\w(])", "x*", t)                      # x(x-1), x cos(x)
    return t


def a_funcion(texto):
    codigo = compile(_normalizar(texto), texto, "eval")
    return lambda x: eval(codigo, {"__builtins__": {}}, {**_NOMBRES, "x": x})


def a_numero(valor):
    if isinstance(valor, str):
        return float(eval(compile(_normalizar(valor), valor, "eval"), {"__builtins__": {}}, _NOMBRES))
    return float(valor)


def derivada_numerica(f):
    def df(x):
        h = 1e-6 * max(1.0, abs(x))
        return (f(x + h) - f(x - h)) / (2 * h)
    return df


class Error(Exception):
    pass


def _error(nuevo, viejo, porcentual):
    er = abs((nuevo - viejo) / nuevo) if nuevo != 0 else abs(nuevo - viejo)
    return er * 100 if porcentual else er


def _cabecera(columnas):
    print(" ".join(f"{c:>14}" for c in columnas))


def _fila(n, valores, mostrar):
    if mostrar:
        print(f"{n:>14} " + " ".join(f"{v:>14.9f}" if abs(v) >= 1e-4 or v == 0 else f"{v:>14.4e}"
                                      for v in valores))


def _cerrado(f, a, b, tol, porcentual, mostrar, nuevo_punto):
    if f(a) * f(b) > 0:
        raise Error(f"f(a)={f(a):.4g} y f(b)={f(b):.4g} tienen el mismo signo")
    if mostrar:
        _cabecera(["n", "a", "b", "p", "f(p)", "error"])
    p_ant = None
    for n in range(1, MAX_IT + 1):
        p = nuevo_punto(a, b)
        er = _error(p, p_ant, porcentual) if p_ant is not None else float("nan")
        _fila(n, [a, b, p, f(p), er], mostrar)
        if f(p) == 0 or er < tol:
            return p, n, er
        if f(a) * f(p) < 0:
            b = p
        else:
            a = p
        p_ant = p
    raise Error("no converge")


def biseccion(f, a, b, tol, porcentual=False, mostrar=True):
    return _cerrado(f, a, b, tol, porcentual, mostrar, lambda a, b: (a + b) / 2)


def posicion_falsa(f, a, b, tol, porcentual=False, mostrar=True):
    return _cerrado(f, a, b, tol, porcentual, mostrar,
                    lambda a, b: b - f(b) * (b - a) / (f(b) - f(a)))


def secante(f, p0, p1, tol, porcentual=False, mostrar=True):
    if mostrar:
        _cabecera(["n", "p", "f(p)", "error"])
        _fila(0, [p0, f(p0), float("nan")], mostrar)
        _fila(1, [p1, f(p1), float("nan")], mostrar)
    for n in range(2, MAX_IT + 2):
        if f(p1) == f(p0):
            raise Error("f(p1) = f(p0), division por cero")
        p = p1 - f(p1) * (p1 - p0) / (f(p1) - f(p0))
        er = _error(p, p1, porcentual)
        _fila(n, [p, f(p), er], mostrar)
        if f(p) == 0 or er < tol:
            return p, n - 1, er
        p0, p1 = p1, p
    raise Error("no converge")


def newton(f, df, p0, tol, porcentual=False, mostrar=True):
    if mostrar:
        _cabecera(["n", "p", "f(p)", "f'(p)", "error"])
        _fila(0, [p0, f(p0), df(p0), float("nan")], mostrar)
    for n in range(1, MAX_IT + 1):
        if df(p0) == 0:
            raise Error(f"f'({p0}) = 0, division por cero")
        p = p0 - f(p0) / df(p0)
        er = _error(p, p0, porcentual)
        _fila(n, [p, f(p), df(p), er], mostrar)
        if f(p) == 0 or er < tol:
            return p, n, er
        p0 = p
    raise Error("no converge")


def resolver(f_texto, intervalos, tol, tipo_error="relativo", metodos=None,
             df_texto=None, x0=None, mostrar=True, titulo=None):
    """Corre los metodos pedidos en cada intervalo e imprime la tabla comparativa."""
    metodos = metodos or ["biseccion", "posicion_falsa", "secante", "newton"]
    porcentual = tipo_error == "porcentual"
    f = a_funcion(f_texto)
    df = a_funcion(df_texto) if df_texto else derivada_numerica(f)
    unidad = "%" if porcentual else ""

    print("\n" + "#" * 78)
    if titulo:
        print(f"# {titulo}")
    print(f"# f(x) = {f_texto}      tolerancia: {tol}{unidad} ({tipo_error})")
    print("#" * 78)

    resumen = []
    for intervalo in intervalos:
        a, b = (a_numero(v) for v in intervalo)
        nombre_int = f"[{intervalo[0]}, {intervalo[1]}]"
        for metodo in metodos:
            p_ini = a_numero(x0) if x0 is not None else (a + b) / 2
            detalle = {"secante": f" (p0={a:g}, p1={b:g})", "newton": f" (p0={p_ini:g})"}.get(metodo, "")
            if mostrar:
                print(f"\n=== {nombre_int}  {metodo}{detalle} ===")
            try:
                if metodo == "biseccion":
                    p, n, er = biseccion(f, a, b, tol, porcentual, mostrar)
                elif metodo == "posicion_falsa":
                    p, n, er = posicion_falsa(f, a, b, tol, porcentual, mostrar)
                elif metodo == "secante":
                    p, n, er = secante(f, a, b, tol, porcentual, mostrar)
                elif metodo == "newton":
                    p, n, er = newton(f, df, p_ini, tol, porcentual, mostrar)
                else:
                    raise Error(f"metodo desconocido: {metodo}")
                nota = "" if min(a, b) <= p <= max(a, b) else "FUERA DEL INTERVALO"
                resumen.append((nombre_int, metodo, f"{p:.9f}", str(n), f"{er:.3e}", f"{f(p):.3e}", nota))
            except (Error, ZeroDivisionError, ValueError, OverflowError) as ex:
                if mostrar:
                    print(f"  !! {ex}")
                resumen.append((nombre_int, metodo, "-", "-", "-", "-", str(ex)))

    print(f"\n--- Tabla comparativa: {titulo or f_texto} ---")
    cols = ("Intervalo", "Metodo", "Raiz", "Iter.", f"Error{unidad}", "f(raiz)", "Obs.")
    anchos = [max(len(str(fila[i])) for fila in resumen + [cols]) for i in range(len(cols))]
    linea = lambda fila: " | ".join(str(v).ljust(w) for v, w in zip(fila, anchos))
    print(linea(cols))
    print("-+-".join("-" * w for w in anchos))
    anterior = None
    for fila in resumen:
        print(linea(("" if fila[0] == anterior else fila[0],) + fila[1:]))
        anterior = fila[0]
    return resumen


if __name__ == "__main__":
    resolver(F, INTERVALOS, TOL, TIPO_ERROR, METODOS, DF, X0_NEWTON, VER_ITERACIONES)
