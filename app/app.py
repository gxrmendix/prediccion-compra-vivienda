"""
app.py
~~~~~~

App web para que un asesor inmobiliario estime la probabilidad de que su cliente compre.

El formulario pide las 22 caracteristicas que un vendedor puede conocer (ubicacion y entorno, la
vivienda, el comprador y el credito hipotecario) y, si ya hubo visita, la calificacion del
cliente. Con calificacion se usa el modelo completo (23 variables); sin ella, el modelo
"antes de la visita" (22). El credito hipotecario, la mensualidad y su proporcion sobre el
ingreso se calculan aqui con la misma formula del dataset (caracteristicas.py).

Funciona sin JavaScript (formulario normal). Con JavaScript, static/app.js recalcula el
resultado en vivo pidiendo solo el fragmento del resultado (cabecera X-Parcial).

Local:  python app.py            -> http://127.0.0.1:8000
Azure:  python -m gunicorn --bind=0.0.0.0 --timeout 120 app:app
"""
import json
import os

import joblib
import numpy as np
from flask import Flask, jsonify, render_template, request

from caracteristicas import (CATEGORIAS, PIES2_POR_M2, VARIABLES_COMPLETO, VARIABLES_SIN_SATISFACCION, codificar,
                             credito_maximo, mensualidad, proporcion_mensualidad_ingreso)

BASE = os.path.dirname(os.path.abspath(__file__))
MODELOS = {v: joblib.load(os.path.join(BASE, "modelos", f"{v}.joblib")) for v in ("completo", "sin_satisfaccion")}
with open(os.path.join(BASE, "modelos", "metadatos.json")) as f:
    METADATOS = json.load(f)
RANGOS = METADATOS["rangos"]
PLAZOS = [10, 15, 20, 25, 30]  # los unicos plazos que aparecen en el dataset
ANIO_ACTUAL = 2026

# Nombres en espanol para mostrar; el modelo recibe los valores originales del dataset
PAISES = {"Australia": "Australia", "Brazil": "Brasil", "Canada": "Canadá", "China": "China", "France": "Francia",
          "Germany": "Alemania", "India": "India", "Japan": "Japón", "Singapore": "Singapur",
          "South Africa": "Sudáfrica", "UAE": "Emiratos Árabes Unidos", "UK": "Reino Unido", "USA": "Estados Unidos"}
TIPOS = {"Independent House": "Casa independiente", "Townhouse": "Casa adosada", "Apartment": "Departamento",
         "Villa": "Villa", "Farmhouse": "Casa de campo", "Studio": "Estudio"}
AMUEBLADO = {"Unfurnished": "Sin amueblar", "Semi-Furnished": "Semiamueblada", "Fully-Furnished": "Amueblada"}

# Casos de ejemplo para probar la app con un clic
EJEMPLOS = {
    "ideal": dict(pais="Canada", ciudad="Toronto", vecindario="8", conectividad="7", delitos="1", tipo="Independent House",
                  amueblado="Semi-Furnished", superficie_m2="140", anio="2012", recamaras="3", banos="2", cochera="1",
                  jardin="1", duenos="1", legal="0", ingreso="45000", gastos="9000", precio="1200000", enganche="420000",
                  plazo="20", satisfaccion="9"),
    "sin_visita": dict(pais="India", ciudad="Bangalore", vecindario="6", conectividad="8", delitos="2", tipo="Apartment",
                       amueblado="Fully-Furnished", superficie_m2="85", anio="2018", recamaras="2", banos="2", cochera="0",
                       jardin="0", duenos="0", legal="0", ingreso="30000", gastos="7000", precio="650000",
                       enganche="200000", plazo="25", satisfaccion=""),
    "problemas": dict(pais="USA", ciudad="Chicago", vecindario="4", conectividad="5", delitos="4", tipo="Townhouse",
                      amueblado="Unfurnished", superficie_m2="120", anio="1985", recamaras="3", banos="1", cochera="1",
                      jardin="1", duenos="3", legal="1", ingreso="20000", gastos="6000", precio="1500000",
                      enganche="250000", plazo="15", satisfaccion="5"),
}

# Seguridad: solo recursos propios y las fuentes de Google; no se puede incrustar en otra pagina
CABECERAS = {
    "Content-Security-Policy": "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline' "
                               "https://fonts.googleapis.com; font-src https://fonts.gstatic.com; img-src 'self' data:; "
                               "connect-src 'self'; form-action 'self'; frame-ancestors 'none'; base-uri 'none'",
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "Strict-Transport-Security": "max-age=31536000",
    "Permissions-Policy": "camera=(), microphone=(), geolocation=()",
}

app = Flask(__name__)


@app.after_request
def cabeceras_de_seguridad(respuesta):
    respuesta.headers.update(CABECERAS)
    return respuesta


def miles(n):
    return f"{n:,.0f}".replace(",", " ")


# ---------------------------------------------------------------- validacion

def leer(form):
    """Valida el formulario y devuelve (datos para el modelo, datos para mostrar, errores)."""
    errores = []

    def numero(clave, nombre, minimo, maximo, entero=False, opcional=False):
        texto = str(form.get(clave) or "").strip().replace(",", "").replace(" ", "")
        if texto == "":
            if not opcional:
                errores.append(f"Falta {nombre}.")
            return None
        try:
            valor = float(texto)
        except ValueError:
            errores.append(f"{nombre.capitalize()} debe ser un número.")
            return None
        if not minimo <= valor <= maximo:  # tambien descarta nan e infinitos
            errores.append(f"{nombre.capitalize()} debe estar entre {miles(minimo)} y {miles(maximo)} "
                           f"(rango con el que se entrenó el modelo).")
            return None
        if entero and valor != int(valor):
            errores.append(f"{nombre.capitalize()} debe ser un número entero.")
            return None
        return int(valor) if entero else valor

    def opcion(clave, nombre, validas):
        valor = form.get(clave)
        if valor not in validas:
            errores.append(f"Elige {nombre}.")
            return None
        return valor

    def r(var):
        return RANGOS[var]

    pais = opcion("pais", "el país", CATEGORIAS["country"])
    ciudad = opcion("ciudad", "la ciudad", CATEGORIAS["city"])
    if pais and ciudad and ciudad not in METADATOS["ciudades_por_pais"][pais]:
        errores.append(f"{ciudad} no está en {PAISES[pais]}: elige una ciudad de ese país.")
    m2 = numero("superficie_m2", "la superficie", round(r("property_size_sqft")[0] / PIES2_POR_M2),
                round(r("property_size_sqft")[1] / PIES2_POR_M2))
    v = {
        "country": pais, "city": ciudad,
        "neighbourhood_rating": numero("vecindario", "la calificación del vecindario", 1, 10, entero=True),
        "connectivity_score": numero("conectividad", "la conectividad", 1, 10, entero=True),
        "crime_cases_reported": numero("delitos", "el número de delitos", *r("crime_cases_reported"), entero=True),
        "property_type": opcion("tipo", "el tipo de vivienda", CATEGORIAS["property_type"]),
        "furnishing_status": opcion("amueblado", "si está amueblada", CATEGORIAS["furnishing_status"]),
        "property_size_sqft": None if m2 is None else round(m2 * PIES2_POR_M2),
        "constructed_year": numero("anio", "el año de construcción", *r("constructed_year"), entero=True),
        "rooms": numero("recamaras", "el número de recámaras", *r("rooms"), entero=True),
        "bathrooms": numero("banos", "el número de baños", *r("bathrooms"), entero=True),
        "garage": numero("cochera", "si tiene cochera", 0, 1, entero=True),
        "garden": numero("jardin", "si tiene jardín", 0, 1, entero=True),
        "previous_owners": numero("duenos", "el número de dueños anteriores", *r("previous_owners"), entero=True),
        "legal_cases_on_property": numero("legal", "si tiene casos legales", 0, 1, entero=True),
        "customer_salary": numero("ingreso", "el ingreso mensual", *r("customer_salary")),
        "monthly_expenses": numero("gastos", "los gastos mensuales", *r("monthly_expenses")),
        "price": numero("precio", "el precio de la vivienda", *r("price")),
        "down_payment": numero("enganche", "el enganche", *r("down_payment")),
        "loan_tenure_years": numero("plazo", "el plazo", 10, 30, entero=True),
        "satisfaction_score": numero("satisfaccion", "la calificación de la visita", 1, 10, entero=True, opcional=True),
    }
    if v["loan_tenure_years"] is not None and v["loan_tenure_years"] not in PLAZOS:
        errores.append("El plazo debe ser de 10, 15, 20, 25 o 30 años.")

    # El credito hipotecario sale del precio y el enganche (en el dataset precio = enganche + credito)
    if v["price"] is not None and v["down_payment"] is not None:
        prop = v["down_payment"] / v["price"]
        lo, hi = RANGOS["proporcion_enganche"]
        if not lo - 0.005 <= prop <= hi + 0.005:
            errores.append(f"El enganche es el {prop * 100:.0f} % del precio; en los datos va del "
                           f"{lo * 100:.0f} % al {hi * 100:.0f} %.")
        v["loan_amount"] = v["price"] - v["down_payment"]
    if not errores:
        v["emi_to_income_ratio"] = float(proporcion_mensualidad_ingreso(v["loan_amount"], v["loan_tenure_years"],
                                                                        v["customer_salary"]))
    return v, errores


# ---------------------------------------------------------------- prediccion

def predecir(v):
    if v["satisfaction_score"] is not None:
        variante, variables = "completo", VARIABLES_COMPLETO
    else:
        variante, variables = "sin_satisfaccion", VARIABLES_SIN_SATISFACCION
    x = codificar([v])[:, [VARIABLES_COMPLETO.index(c) for c in variables]]
    prob = float(MODELOS[variante].predict_proba(x)[0, 1])

    emi, sat = v["emi_to_income_ratio"], v["satisfaction_score"]
    condiciones = [
        {"texto": "Sin casos legales en la propiedad", "estado": v["legal_cases_on_property"] == 0,
         "detalle": "Expediente limpio." if v["legal_cases_on_property"] == 0 else "Tiene un caso legal abierto: en los datos nadie compra así."},
        {"texto": "Zona con 2 delitos reportados o menos", "estado": v["crime_cases_reported"] <= 2,
         "detalle": f"{v['crime_cases_reported']} reportado{'' if v['crime_cases_reported'] == 1 else 's'}."},
        {"texto": "Mensualidad de 39 % del ingreso o menos", "estado": emi <= 0.39,
         "detalle": f"La mensualidad sería el {emi * 100:.0f} % del ingreso."},
        {"texto": "Calificación de la visita de 7 o más", "estado": None if sat is None else sat >= 7,
         "detalle": "Pendiente: todavía no visita la casa." if sat is None else f"Calificó con {sat} de 10."},
    ]

    # Sugerencia: cuanto enganche (o que plazo) haria que la mensualidad quepa en el 39 %
    sugerencia = None
    if emi > 0.39:
        tope = 0.394  # el dataset redondea a 2 decimales: 0.394 todavia cuenta como 0.39
        enganche_min = v["price"] - float(credito_maximo(v["customer_salary"], v["loan_tenure_years"], tope))
        plazos_ok = [p for p in PLAZOS if p > v["loan_tenure_years"] and
                     proporcion_mensualidad_ingreso(v["loan_amount"], p, v["customer_salary"]) <= 0.39]
        if enganche_min <= RANGOS["proporcion_enganche"][1] * v["price"]:
            sugerencia = (f"Con un enganche de al menos {miles(np.ceil(enganche_min / 1000) * 1000)} "
                          f"({enganche_min / v['price'] * 100:.0f} % del precio) la mensualidad bajaría al 39 % del ingreso.")
        if plazos_ok:
            sugerencia = (sugerencia + " O bien, " if sugerencia else "") + \
                f"con un plazo de {plazos_ok[0]} años la mensualidad ya cabría."
        if not sugerencia:
            sugerencia = "Ni con el enganche máximo ni con 30 años de plazo la mensualidad baja al 39 % del ingreso."

    nivel = "Muy probable" if prob >= 0.7 else "Posible" if prob >= 0.3 else "Poco probable"
    return {
        "probabilidad": prob, "variante": variante, "nivel": nivel, "condiciones": condiciones,
        "sugerencia": sugerencia, "proporcion_mensualidad": emi, "credito": v["loan_amount"],
        "mensualidad": float(mensualidad(v["loan_amount"], v["loan_tenure_years"])),
        "proporcion_enganche": v["down_payment"] / v["price"],
        "antiguedad": ANIO_ACTUAL - v["constructed_year"], "n_variables": len(variables),
        "avisos": [c["detalle"] for c in condiciones if c["estado"] is False],
    }


# ---------------------------------------------------------------- rutas

def contexto(form, resultado=None, errores=()):
    return dict(form=form, resultado=resultado, errores=list(errores), plazos=PLAZOS, miles=miles, paises=PAISES,
                tipos=TIPOS, amueblado=AMUEBLADO, ciudades=METADATOS["ciudades_por_pais"], rangos=RANGOS,
                metricas=METADATOS["metricas_prueba"], pies2=PIES2_POR_M2, anio_actual=ANIO_ACTUAL)


@app.route("/", methods=["GET", "POST"])
def inicio():
    if request.method == "POST":
        form = request.form
    else:
        form = EJEMPLOS.get(request.args.get("ejemplo"), {})
    resultado, errores = None, []
    if form:
        v, errores = leer(form)
        if not errores:
            resultado = predecir(v)
    ctx = contexto(form, resultado, errores)
    if request.headers.get("X-Parcial"):  # la app en vivo solo pide el panel del resultado
        return render_template("_resultado.html", parcial=True, **ctx)
    return render_template("index.html", **ctx)


@app.route("/api/prediccion", methods=["POST"])
def api_prediccion():
    datos = request.get_json(silent=True)
    if not isinstance(datos, dict):
        return jsonify({"errores": ["Envía un objeto JSON con los campos del formulario."]}), 400
    v, errores = leer({k: str(x) for k, x in datos.items()})
    if errores:
        return jsonify({"errores": errores}), 400
    return jsonify(predecir(v))


@app.route("/salud")
def salud():
    return jsonify({"estado": "ok", "modelos": METADATOS["modelos"], "sklearn": METADATOS["sklearn"]})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000, debug=False)
