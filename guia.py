"""TP5 resuelto completo con metodos.py.

    python3 guia.py               -> todo, con tablas de iteraciones
    python3 guia.py resumen       -> solo las tablas comparativas
    python3 guia.py 1e-6 resumen  -> todos los ejercicios con error relativo 10^-6
"""

import sys

from metodos import resolver

mostrar = "resumen" not in sys.argv
TOL_NUEVA = next((float(a) for a in sys.argv[1:] if a != "resumen"), None)


def tol(valor, tipo):
    """Tolerancia del enunciado, o la pasada por linea de comandos (relativa)."""
    return (TOL_NUEVA, "relativo") if TOL_NUEVA is not None else (valor, tipo)


resolver("x^3 - 7x^2 + 14x - 6", [(0, 1), (1, 3.2), (3.2, 4)], *tol(1e-4, "relativo"),
         ["biseccion", "secante", "newton"], df_texto="3x^2 - 14x + 14",
         mostrar=mostrar, titulo="Ej. 1")

resolver("x - 2^(-x)", [(0, 1)], *tol(1e-4, "relativo"), ["biseccion", "secante"],
         mostrar=mostrar, titulo="Ej. 2a")
resolver("e^x - x^2 + 3x - 2", [(0, 1)], *tol(1e-4, "relativo"), ["biseccion", "secante"],
         mostrar=mostrar, titulo="Ej. 2b")
resolver("2x cos(2x) - (x+1)^2", [(-3, -2), (-1, 0)], *tol(1e-4, "relativo"), ["biseccion", "secante"],
         mostrar=mostrar, titulo="Ej. 2c")

# raiz de 3 = solucion de x^2 - 3 = 0, esta entre 1 y 2
resolver("x^2 - 3", [(1, 2)], *tol(0.01, "porcentual"), ["secante"],
         mostrar=mostrar, titulo="Ej. 3: raiz cuadrada de 3")

# raiz cubica de 25 = solucion de x^3 - 25 = 0, esta entre 2 y 3 (2^3=8, 3^3=27)
resolver("x^3 - 25", [(2, 3)], *tol(1e-4, "relativo"), ["newton"], df_texto="3x^2",
         mostrar=mostrar, titulo="Ej. 4: raiz cubica de 25")

resolver("(x-2)^2 - ln x", [(1, 2), ("e", 4)], *tol(0.001, "porcentual"),
         df_texto="2(x-2) - 1/x", mostrar=mostrar, titulo="Ej. 5a")
resolver("sen x - e^(-x)", [(0, 1), (3, 4)], *tol(0.001, "porcentual"),
         df_texto="cos(x) + e^(-x)", mostrar=mostrar, titulo="Ej. 5b")
resolver("230x^4 + 18x^3 + 9x^2 - 221x - 9", [(-1, 0), (0, 1)], *tol(0.001, "porcentual"),
         df_texto="920x^3 + 54x^2 + 18x - 221", mostrar=mostrar, titulo="Ej. 5c")
# en [0,1] secante (desde 0 y 1) y Newton (desde 0.5) se escapan a la raiz -0.0407:
# la funcion es casi plana cerca de 0.5. Arrancando cerca de 1 convergen bien.
resolver("230x^4 + 18x^3 + 9x^2 - 221x - 9", [(0.9, 1)], *tol(0.001, "porcentual"),
         ["secante", "newton"], df_texto="920x^3 + 54x^2 + 18x - 221", x0=1,
         mostrar=mostrar, titulo="Ej. 5c [0,1] con mejores puntos iniciales")
