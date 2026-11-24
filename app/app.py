"""
app.py
~~~~~~

Formulario web para que un vendedor estime la probabilidad de que su cliente compre la casa.

Si el vendedor ya tiene la calificacion del cliente despues de la visita, se usa el modelo
completo; si no, el modelo "antes de la visita", que no la necesita. La mensualidad del
prestamo se calcula aqui con la misma formula del dataset (caracteristicas.py).

Local:  python app.py            -> http://127.0.0.1:8000
Azure:  gunicorn --bind=0.0.0.0 --timeout 120 app:app
"""
import json
import os
import warnings

import joblib
import numpy as np
from flask import Flask, jsonify, render_template, request

from caracteristicas import mensualidad, proporcion_mensualidad_ingreso

# Los modelos se entrenaron con DataFrames; aqui se predice con arreglos para no cargar pandas
warnings.filterwarnings("ignore", message="X does not have valid feature names")

BASE = os.path.dirname(os.path.abspath(__file__))
MODELOS = {v: joblib.load(os.path.join(BASE, "modelos", f"{v}.joblib")) for v in ("completo", "sin_satisfaccion")}
with open(os.path.join(BASE, "modelos", "metadatos.json")) as f:
    METADATOS = json.load(f)
PLAZOS = [10, 15, 20, 25, 30]  # los unicos plazos que aparecen en el dataset

app = Flask(__name__)


def leer(formulario):
    """Valida el formulario y devuelve (datos, errores)."""
    errores, datos = [], {}

    def numero(clave, nombre, minimo, maximo, entero=False, opcional=False):
        texto = (formulario.get(clave) or "").strip().replace(",", "")
        if texto == "":
            if not opcional:
                errores.append(f"Falta {nombre}.")
            return None
        try:
            valor = int(texto) if entero else float(texto)
        except ValueError:
            errores.append(f"{nombre.capitalize()} debe ser un número.")
            return None
        if not minimo <= valor <= maximo:
            errores.append(f"{nombre.capitalize()} debe estar entre {minimo:,} y {maximo:,}.")
            return None
        return valor

    datos["salario"] = numero("salario", "el salario mensual", 1, 10_000_000)
    datos["prestamo"] = numero("prestamo", "el monto del préstamo", 1, 1_000_000_000)
    datos["plazo"] = numero("plazo", "el plazo", 10, 30, entero=True)
    if datos["plazo"] is not None and datos["plazo"] not in PLAZOS:
        errores.append("El plazo debe ser de 10, 15, 20, 25 o 30 años.")
    datos["crimenes"] = numero("crimenes", "el número de delitos reportados", 0, 50, entero=True)
    legal = formulario.get("legal")
    if legal not in ("0", "1"):
        errores.append("Indica si la propiedad tiene algún caso legal.")
    datos["legal"] = int(legal) if legal in ("0", "1") else None
    datos["satisfaccion"] = numero("satisfaccion", "la calificación de la visita", 1, 10, entero=True, opcional=True)
    return datos, errores


def predecir(d):
    emi = float(proporcion_mensualidad_ingreso(d["prestamo"], d["plazo"], d["salario"]))
    if d["satisfaccion"] is not None:
        variante, x = "completo", [d["satisfaccion"], d["legal"], d["crimenes"], emi]
    else:
        variante, x = "sin_satisfaccion", [d["legal"], d["crimenes"], emi]
    prob = float(MODELOS[variante].predict_proba(np.array([x], dtype=float))[0, 1])

    # Avisos para el vendedor: que condiciones del cliente o de la casa estan bajando la probabilidad
    avisos = []
    if d["legal"] == 1:
        avisos.append("La propiedad tiene un caso legal abierto.")
    if d["crimenes"] > 2:
        avisos.append(f"Se reportan {d['crimenes']} delitos en la zona; con más de 2 casi nadie compra.")
    if emi > 0.39:
        avisos.append(f"La mensualidad sería el {emi * 100:.0f} % del salario; por encima del 39 % casi nadie compra.")
    if d["satisfaccion"] is not None and d["satisfaccion"] < 7:
        avisos.append(f"El cliente calificó la visita con {d['satisfaccion']}; con menos de 7 casi nadie compra.")
    return {
        "probabilidad": prob, "variante": variante, "proporcion_mensualidad": emi,
        "mensualidad": float(mensualidad(d["prestamo"], d["plazo"])), "avisos": avisos,
    }


@app.route("/", methods=["GET", "POST"])
def inicio():
    resultado, errores, datos = None, [], {}
    if request.method == "POST":
        datos, errores = leer(request.form)
        if not errores:
            resultado = predecir(datos)
    return render_template("index.html", resultado=resultado, errores=errores, form=request.form, plazos=PLAZOS)


@app.route("/api/prediccion", methods=["POST"])
def api_prediccion():
    datos, errores = leer({k: str(v) for k, v in (request.get_json(silent=True) or {}).items()})
    if errores:
        return jsonify({"errores": errores}), 400
    return jsonify(predecir(datos))


@app.route("/salud")
def salud():
    return jsonify({"estado": "ok", "modelos": METADATOS})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000, debug=False)
