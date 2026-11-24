"""
entrenar.py
~~~~~~~~~~~

Entrenamiento, seleccion y evaluacion de los modelos, con registro en DagsHub (MLflow).

Se entrenan dos variantes con TODAS las caracteristicas que un vendedor puede conocer:
  - completo (23 variables): incluye la satisfaccion del cliente despues de la visita;
  - sin_satisfaccion (22):   para estimar ANTES de la visita.
En cada variante compiten una linea base (siempre la tasa de compra), una regresion logistica
y un gradient boosting. Se elige el de menor log-loss en validacion (lo que importa es que la
probabilidad sea buena, no solo la clase) y solo ese se evalua una vez en prueba.

Como referencia se entrena tambien un gradient boosting con solo las variables que resultan
importantes (4 y 3), para mostrar que el resto no aporta. La importancia de cada variable se
mide en validacion revolviendo sus valores (permutacion).

Todo se acumula en memoria y se sube a DagsHub al final, una peticion por corrida (log_batch).

Uso: python src/entrenar.py [--sin-dagshub]
"""
import argparse
import json
import os
import sys
import time

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, brier_score_loss, confusion_matrix, f1_score, log_loss,
                             precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import estilo

sys.path.insert(0, "app")
from caracteristicas import CATEGORIAS, VARIABLES_COMPLETO, VARIABLES_SIN_SATISFACCION, codificar  # noqa: E402

SEMILLA = 42
OBJETIVO = "decision"
IMPORTANTES = {"completo": ["satisfaction_score", "legal_cases_on_property", "crime_cases_reported", "emi_to_income_ratio"],
               "sin_satisfaccion": ["legal_cases_on_property", "crime_cases_reported", "emi_to_income_ratio"]}
VARIABLES = {"completo": VARIABLES_COMPLETO, "sin_satisfaccion": VARIABLES_SIN_SATISFACCION}
NOMBRES = {"linea_base": "Línea base", "regresion_logistica": "Regresión logística",
           "gradient_boosting": "Gradient boosting", "referencia_reducida": "Gradient boosting reducido"}
TITULO = {"completo": "con satisfacción de la visita", "sin_satisfaccion": "antes de la visita"}
NOMBRES_ES = {
    "country": "país", "city": "ciudad", "property_type": "tipo de vivienda", "furnishing_status": "amueblado",
    "property_size_sqft": "superficie", "price": "precio", "constructed_year": "año de construcción",
    "previous_owners": "dueños anteriores", "rooms": "recámaras", "bathrooms": "baños", "garage": "cochera",
    "garden": "jardín", "crime_cases_reported": "delitos en la zona", "legal_cases_on_property": "casos legales",
    "customer_salary": "ingreso mensual", "loan_amount": "crédito hipotecario", "loan_tenure_years": "plazo",
    "monthly_expenses": "gastos mensuales", "down_payment": "enganche", "emi_to_income_ratio": "mensualidad / ingreso",
    "neighbourhood_rating": "calificación del vecindario", "connectivity_score": "conectividad",
    "satisfaction_score": "satisfacción de la visita",
}


def cargar():
    partes = {}
    for n in ["entrenamiento", "validacion", "prueba"]:
        d = pd.read_parquet(f"datos/{n}.parquet")
        partes[n] = (codificar(d), d[OBJETIVO].to_numpy())
    return partes


def columnas(variables):
    return [VARIABLES_COMPLETO.index(v) for v in variables]


def metricas(y, prob):
    pred = (prob >= 0.5).astype(int)
    return {
        "auc": roc_auc_score(y, prob), "log_loss": log_loss(y, prob, labels=[0, 1]),
        "brier": brier_score_loss(y, prob), "exactitud": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0), "exhaustividad": recall_score(y, pred),
        "f1": f1_score(y, pred), "tasa_predicha_media": float(prob.mean()),
    }


def gradient_boosting(variables):
    # Early stopping con su propia fraccion interna del entrenamiento; validacion queda intacta
    return HistGradientBoostingClassifier(
        max_iter=500, learning_rate=0.1, early_stopping=True, validation_fraction=0.1, n_iter_no_change=20,
        scoring="loss", categorical_features=[v in CATEGORIAS for v in variables], random_state=SEMILLA)


def regresion_logistica(variables):
    cat = [i for i, v in enumerate(variables) if v in CATEGORIAS]
    num = [i for i, v in enumerate(variables) if v not in CATEGORIAS]
    prep = ColumnTransformer([("categorias", OneHotEncoder(handle_unknown="ignore"), cat),
                              ("numeros", StandardScaler(), num)])
    return make_pipeline(prep, LogisticRegression(max_iter=3000))


def curva(modelo):
    # sklearn guarda la perdida con signo negativo
    return {"entrenamiento": (-modelo.train_score_).tolist(), "validacion_interna": (-modelo.validation_score_).tolist()}


# ---------------------------------------------------------------- entrenamiento

def entrenar_variante(partes, variante):
    (x_ent, y_ent), (x_val, y_val) = partes["entrenamiento"], partes["validacion"]
    todas, reducidas = VARIABLES[variante], IMPORTANTES[variante]
    candidatos = {
        "linea_base": (todas, DummyClassifier(strategy="prior")),
        "regresion_logistica": (todas, regresion_logistica(todas)),
        "gradient_boosting": (todas, gradient_boosting(todas)),
        "referencia_reducida": (reducidas, gradient_boosting(reducidas)),
    }
    corridas = []
    for nombre, (variables, modelo) in candidatos.items():
        idx = columnas(variables)
        inicio = time.time()
        modelo.fit(x_ent[:, idx], y_ent)
        segundos = time.time() - inicio
        prob_val = modelo.predict_proba(x_val[:, idx])[:, 1]
        c = {"nombre": f"{variante}__{nombre}", "variante": variante, "modelo_nombre": nombre, "modelo": modelo,
             "variables": variables, "segundos": segundos,
             "parametros": {"variante": variante, "modelo": nombre, "n_variables": len(variables),
                            "variables": ",".join(variables), "candidato_para_la_app": nombre != "referencia_reducida"},
             "metricas_validacion": metricas(y_val, prob_val)}
        if isinstance(modelo, HistGradientBoostingClassifier):
            c["parametros"]["iteraciones"] = modelo.n_iter_
            c["curva"] = curva(modelo)
        corridas.append(c)
        print(f"{variante:17s} {nombre:20s} ({len(variables):2d} var) validacion: " +
              " ".join(f"{k}={v:.4f}" for k, v in c["metricas_validacion"].items()))
    return corridas


def importancia(c, partes):
    x_val, y_val = partes["validacion"]
    imp = permutation_importance(c["modelo"], x_val[:, columnas(c["variables"])], y_val, scoring="neg_log_loss",
                                 n_repeats=3, random_state=SEMILLA, n_jobs=-1)
    serie = pd.Series(imp.importances_mean, c["variables"]).sort_values(ascending=False)
    return {k: round(float(v), 5) for k, v in serie.items()}


def evaluar_en_prueba(c, partes):
    x, y = partes["prueba"]
    prob = c["modelo"].predict_proba(x[:, columnas(c["variables"])])[:, 1]
    c["metricas_prueba"] = metricas(y, prob)
    vn, fp, fn, vp = confusion_matrix(y, (prob >= 0.5).astype(int)).ravel()
    c["matriz_confusion_prueba"] = {"VN": int(vn), "FP": int(fp), "FN": int(fn), "VP": int(vp)}
    c["prob_maxima_prueba"] = float(prob.max())
    c["clientes_sobre_50"] = int((prob >= 0.5).sum())
    c["_y"], c["_prob"] = y, prob
    return c


# ---------------------------------------------------------------- figuras

def fig_importancia(imp_completo, imp_sin):
    fig, (a1, a2) = estilo.plt.subplots(1, 2, figsize=(11, 6.4), gridspec_kw={"wspace": 0.75})
    fig.patch.set_facecolor(estilo.FONDO)
    for ax, imp, titulo, sub in [
        (a1, imp_completo, "1. Qué variables usa el modelo", "con satisfacción · 23 variables · validación"),
        (a2, imp_sin, "Antes de la visita", "sin satisfacción · 22 variables · validación"),
    ]:
        serie = pd.Series(imp).rename(index=NOMBRES_ES).sort_values()
        colores = [estilo.AZUL if v > 0.001 else estilo.REJILLA for v in serie.values]
        ax.barh(serie.index, serie.values, color=colores, height=0.7)
        for i, v in enumerate(serie.values):
            if v > 0.001:
                ax.text(v, i, f"  {v:.2f}", va="center", fontsize=9, color=estilo.TINTA)
        estilo.marco(ax, titulo=titulo, subtitulo=sub, xlabel="cuánto empeora al revolver la variable", leyenda=False)
        ax.grid(axis="y", visible=False)
        ax.set_xlim(0, serie.max() * 1.25)
    # dos paneles con etiquetas largas: tight_layout no alcanza, se recorta al contenido
    fig.savefig("figuras/01-importancia-variables.png", dpi=200, facecolor=estilo.FONDO, bbox_inches="tight", pad_inches=0.25)
    estilo.plt.close(fig)


def fig_curvas(elegidos):
    fig, ax = estilo.figura()
    for c, color in zip(elegidos, [estilo.AZUL, estilo.NARANJA]):
        etiqueta = "Con satisfacción" if c["variante"] == "completo" else "Antes de la visita"
        ax.plot(c["curva"]["entrenamiento"], color=color, lw=2, label=f"{etiqueta} · entrenamiento")
        ax.plot(c["curva"]["validacion_interna"], color=color, lw=1.6, ls="--", label=f"{etiqueta} · validación")
    ax.set_yscale("log")
    estilo.marco(ax, titulo="2. Curvas de entrenamiento del gradient boosting",
                 subtitulo="log-loss por iteración · la validación (punteada) queda encima del entrenamiento: no hay sobreajuste",
                 xlabel="iteración (árboles agregados)", ylabel="log-loss (escala logarítmica)")
    estilo.guardar(fig, "figuras/02-curvas-entrenamiento.png")


def fig_comparacion(corridas):
    variantes = ["completo", "sin_satisfaccion"]
    modelos = ["linea_base", "regresion_logistica", "gradient_boosting", "referencia_reducida"]
    colores = [estilo.REJILLA, estilo.NARANJA, estilo.AZUL, estilo.VERDE]
    fig, ax = estilo.figura(9.4, 4.6)
    ancho = 0.2
    for i, (m, color) in enumerate(zip(modelos, colores)):
        valores = [next(c for c in corridas if c["variante"] == v and c["modelo_nombre"] == m)
                   ["metricas_validacion"]["auc"] for v in variantes]
        etiqueta = NOMBRES[m] + (" (solo 4 o 3 variables)" if m == "referencia_reducida" else " (todas)" if m != "linea_base" else "")
        barras = ax.bar(np.arange(2) + (i - 1.5) * ancho, valores, ancho * 0.92, color=color, label=etiqueta)
        for b, v in zip(barras, valores):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}", ha="center", fontsize=8.5, color=estilo.TINTA)
    ax.set_xticks(np.arange(2), [f"Modelo {TITULO[v]}" for v in variantes])
    ax.set_ylim(0.4, 1.1)
    estilo.marco(ax, titulo="3. Comparación de modelos en validación",
                 subtitulo="AUC en 20 000 clientes · con todas las variables o solo las importantes da lo mismo",
                 ylabel="AUC")
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    ax.grid(axis="x", visible=False)
    estilo.guardar(fig, "figuras/03-comparacion-modelos.png")


def fig_prueba(c, numero):
    y, prob = c["_y"], c["_prob"]
    fig, (a1, a2) = estilo.plt.subplots(1, 2, figsize=(10.4, 4.6))
    fig.patch.set_facecolor(estilo.FONDO)
    fpr, tpr, _ = roc_curve(y, prob)
    a1.plot([0, 1], [0, 1], color=estilo.REJILLA, lw=1.2, ls="--", label="Al azar")
    a1.plot(fpr, tpr, color=estilo.AZUL, lw=2.4, label=f"Modelo · AUC {c['metricas_prueba']['auc']:.4f}")
    corto = "con satisfacción" if c["variante"] == "completo" else "antes de la visita"
    estilo.marco(a1, titulo=f"{numero}. Modelo {corto} · curva ROC",
                 subtitulo=f"{NOMBRES[c['modelo_nombre']]} · {len(y):,} clientes de prueba".replace(",", " "),
                 xlabel="tasa de falsos positivos", ylabel="tasa de verdaderos positivos")
    real, predicha = calibration_curve(y, prob, n_bins=10, strategy="uniform")
    a2.plot([0, 1], [0, 1], color=estilo.REJILLA, lw=1.2, ls="--", label="Calibración perfecta")
    a2.plot(predicha, real, color=estilo.NARANJA, lw=2.4, marker="o", ms=6, label="Modelo")
    estilo.marco(a2, titulo="Calibración", subtitulo="probabilidad del modelo contra la compra real",
                 xlabel="probabilidad predicha", ylabel="frecuencia real de compra")
    estilo.guardar(fig, f"figuras/{numero:02d}-prueba-{c['variante'].replace('_', '-')}.png")


# ---------------------------------------------------------------- DagsHub

def subir_a_dagshub(corridas, info):
    os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "60")
    os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "3")
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    import dagshub
    import mlflow
    from mlflow.entities import Metric, Param
    dagshub.init(repo_owner="garmendiavictor03", repo_name="prediccion-compra-vivienda", mlflow=True)
    mlflow.set_experiment("compra-vivienda")
    cliente = mlflow.MlflowClient()
    ahora = int(time.time() * 1000)
    for c in corridas:
        with mlflow.start_run(run_name=c["nombre"]) as run:
            params = [Param(k, str(v)[:6000]) for k, v in {**c["parametros"], **info}.items()]
            m = [Metric(f"val_{k}", float(v), ahora, 0) for k, v in c["metricas_validacion"].items()]
            m.append(Metric("segundos_entrenamiento", float(c["segundos"]), ahora, 0))
            if "metricas_prueba" in c:
                m += [Metric(f"prueba_{k}", float(v), ahora, 0) for k, v in c["metricas_prueba"].items()]
                mlflow.set_tag("seleccionado", "si")
            for paso, (a, b) in enumerate(zip(*c["curva"].values()) if "curva" in c else []):
                m += [Metric("curva_perdida_entrenamiento", a, ahora, paso),
                      Metric("curva_perdida_validacion", b, ahora, paso)]
            for i in range(0, len(m), 900):  # limite de MLflow: 1000 elementos por lote
                cliente.log_batch(run.info.run_id, metrics=m[i:i + 900], params=params if i == 0 else [])
            for ruta in c.get("figuras", []):
                mlflow.log_artifact(ruta)
            if "importancia" in c:
                mlflow.log_dict(c["importancia"], "importancia_variables.json")
    return "https://dagshub.com/garmendiavictor03/prediccion-compra-vivienda.mlflow"


# ---------------------------------------------------------------- principal

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sin-dagshub", action="store_true")
    args = ap.parse_args()
    partes = cargar()
    info = {"semilla": SEMILLA, "sklearn": sklearn.__version__, "filas_entrenamiento": len(partes["entrenamiento"][1]),
            "filas_validacion": len(partes["validacion"][1]), "filas_prueba": len(partes["prueba"][1])}

    corridas, elegidos = [], {}
    for variante in ["completo", "sin_satisfaccion"]:
        candidatas = entrenar_variante(partes, variante)
        # Para la app solo compiten los modelos con todas las variables del formulario
        mejor = min((c for c in candidatas if c["parametros"]["candidato_para_la_app"]),
                    key=lambda c: c["metricas_validacion"]["log_loss"])
        mejor["importancia"] = importancia(mejor, partes)
        evaluar_en_prueba(mejor, partes)
        elegidos[variante] = mejor
        joblib.dump(mejor["modelo"], f"app/modelos/{variante}.joblib")
        print(f"elegido {variante}: {mejor['modelo_nombre']} | prueba: " +
              " ".join(f"{k}={v:.4f}" for k, v in mejor["metricas_prueba"].items()))
        print("  importancia:", dict(list(mejor["importancia"].items())[:5]))
        corridas += candidatas

    fig_importancia(elegidos["completo"]["importancia"], elegidos["sin_satisfaccion"]["importancia"])
    fig_curvas([elegidos["completo"], elegidos["sin_satisfaccion"]])
    fig_comparacion(corridas)
    fig_prueba(elegidos["completo"], 4)
    fig_prueba(elegidos["sin_satisfaccion"], 5)
    elegidos["completo"]["figuras"] = ["figuras/01-importancia-variables.png", "figuras/02-curvas-entrenamiento.png",
                                       "figuras/03-comparacion-modelos.png", "figuras/04-prueba-completo.png"]
    elegidos["sin_satisfaccion"]["figuras"] = ["figuras/05-prueba-sin-satisfaccion.png"]

    resumen = {
        "info": info,
        "corridas": [{k: c[k] for k in ("nombre", "parametros", "metricas_validacion", "segundos")} for c in corridas],
        "elegidos": {v: {k: c[k] for k in ("nombre", "metricas_validacion", "metricas_prueba", "matriz_confusion_prueba",
                                           "importancia", "prob_maxima_prueba", "clientes_sobre_50")}
                     for v, c in elegidos.items()},
    }
    if not args.sin_dagshub:
        inicio = time.time()
        try:
            resumen["dagshub_mlflow"] = subir_a_dagshub(corridas, info)
            print(f"subido a DagsHub en {time.time() - inicio:.0f} s")
        except Exception as e:  # el entrenamiento ya quedo guardado; solo se reporta el fallo
            resumen["dagshub_error"] = repr(e)
            print("no se pudo subir a DagsHub:", e)
    with open("resultados/entrenamiento.json", "w") as f:
        json.dump(resumen, f, indent=1, default=float)

    # Rangos del entrenamiento: la app rechaza valores fuera de ellos
    d = pd.read_parquet("datos/entrenamiento.parquet")
    numericas = [v for v in VARIABLES_COMPLETO if v not in CATEGORIAS]
    rangos = {v: [float(d[v].min()), float(d[v].max())] for v in numericas}
    rangos["proporcion_enganche"] = [float((d.down_payment / d.price).min()), float((d.down_payment / d.price).max())]
    ciudades = {p: sorted(d.loc[d.country == p, "city"].unique().tolist()) for p in CATEGORIAS["country"]}
    with open("app/modelos/metadatos.json", "w") as f:
        json.dump({"sklearn": sklearn.__version__, "variables": VARIABLES_COMPLETO, "rangos": rangos,
                   "ciudades_por_pais": ciudades,
                   "metricas_prueba": {v: {k: round(float(x), 4) for k, x in elegidos[v]["metricas_prueba"].items()}
                                       for v in elegidos},
                   "modelos": {v: elegidos[v]["modelo_nombre"] for v in elegidos}}, f, indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
