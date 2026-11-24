"""
entrenar.py
~~~~~~~~~~~

Entrenamiento, seleccion y evaluacion de los modelos, con registro en DagsHub (MLflow).

Etapa 1. Con las 23 variables que un vendedor podria conocer se entrena un gradient boosting y
se mide en VALIDACION cuanto aporta cada variable (importancia por permutacion).
Etapa 2. Con las variables que si aportan se entrenan dos variantes:
  - completo:         incluye la satisfaccion del cliente despues de la visita;
  - sin_satisfaccion: para estimar ANTES de la visita.
  En cada variante compiten una linea base (siempre la tasa de compra), una regresion logistica
  y un gradient boosting. Se elige el de menor log-loss en validacion (lo que importa es que la
  probabilidad sea buena, no solo la clase) y solo ese se evalua una vez en prueba.

Todo se acumula en memoria y se sube a DagsHub al final, una peticion por corrida (log_batch):
registrar metrica por metrica contra un servidor remoto hace que la subida tarde mas que el
entrenamiento.

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
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, brier_score_loss, confusion_matrix, f1_score, log_loss,
                             precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import estilo

sys.path.insert(0, "app")
from caracteristicas import VARIABLES_COMPLETO, VARIABLES_SIN_SATISFACCION  # noqa: E402

SEMILLA = 42
CATEGORICAS = ["country", "city", "property_type", "furnishing_status"]
OBJETIVO = "decision"
NOMBRES = {"linea_base": "Línea base", "regresion_logistica": "Regresión logística",
           "gradient_boosting": "Gradient boosting"}
TITULO_VARIANTE = {"completo": "con satisfacción de la visita", "sin_satisfaccion": "sin satisfacción (antes de la visita)"}


def cargar():
    partes = {n: pd.read_parquet(f"datos/{n}.parquet") for n in ["entrenamiento", "validacion", "prueba"]}
    for p in partes.values():
        for c in CATEGORICAS:
            p[c] = p[c].astype("category")
    return partes


def metricas(y, prob):
    pred = (prob >= 0.5).astype(int)
    return {
        "auc": roc_auc_score(y, prob), "log_loss": log_loss(y, prob, labels=[0, 1]),
        "brier": brier_score_loss(y, prob), "exactitud": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0), "exhaustividad": recall_score(y, pred),
        "f1": f1_score(y, pred), "tasa_predicha_media": float(prob.mean()),
    }


def gradient_boosting():
    # Early stopping con su propia fraccion interna del entrenamiento; validacion queda intacta
    return HistGradientBoostingClassifier(max_iter=500, learning_rate=0.1, early_stopping=True,
                                          validation_fraction=0.1, n_iter_no_change=20, scoring="loss",
                                          categorical_features="from_dtype", random_state=SEMILLA)


def curva(modelo):
    # sklearn guarda la perdida con signo negativo
    return {"entrenamiento": (-modelo.train_score_).tolist(), "validacion_interna": (-modelo.validation_score_).tolist()}


# ---------------------------------------------------------------- etapas

def etapa_seleccion(partes):
    variables = [c for c in partes["entrenamiento"].columns if c not in ("property_id", OBJETIVO)]
    ent, val = partes["entrenamiento"], partes["validacion"]
    inicio = time.time()
    modelo = gradient_boosting().fit(ent[variables], ent[OBJETIVO])
    segundos = time.time() - inicio
    prob = modelo.predict_proba(val[variables])[:, 1]
    imp = permutation_importance(modelo, val[variables], val[OBJETIVO], scoring="neg_log_loss",
                                 n_repeats=3, random_state=SEMILLA, n_jobs=-1)
    importancia = pd.Series(imp.importances_mean, variables).sort_values(ascending=False)
    return {
        "nombre": "seleccion_23_variables", "modelo": modelo,
        "parametros": {"etapa": "seleccion", "variables": len(variables), "modelo": "gradient_boosting",
                       "iteraciones": modelo.n_iter_},
        "metricas_validacion": metricas(val[OBJETIVO].values, prob), "segundos": segundos,
        "importancia": {k: round(float(v), 5) for k, v in importancia.items()}, "curva": curva(modelo),
    }


def etapa_variante(partes, variante, variables):
    ent, val = partes["entrenamiento"], partes["validacion"]
    candidatos = {
        "linea_base": DummyClassifier(strategy="prior"),
        "regresion_logistica": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
        "gradient_boosting": gradient_boosting(),
    }
    corridas = []
    for nombre, modelo in candidatos.items():
        inicio = time.time()
        modelo.fit(ent[variables], ent[OBJETIVO])
        segundos = time.time() - inicio
        prob_val = modelo.predict_proba(val[variables])[:, 1]
        c = {"nombre": f"{variante}__{nombre}", "variante": variante, "modelo_nombre": nombre, "modelo": modelo,
             "parametros": {"etapa": "variante", "variante": variante, "modelo": nombre, "variables": ",".join(variables)},
             "metricas_validacion": metricas(val[OBJETIVO].values, prob_val), "segundos": segundos}
        if nombre == "gradient_boosting":
            c["parametros"]["iteraciones"] = modelo.n_iter_
            c["curva"] = curva(modelo)
        corridas.append(c)
        print(f"{variante:17s} {nombre:20s} validacion: " +
              " ".join(f"{k}={v:.4f}" for k, v in c["metricas_validacion"].items()))
    return corridas


def evaluar_en_prueba(c, prueba, variables):
    y, prob = prueba[OBJETIVO].values, c["modelo"].predict_proba(prueba[variables])[:, 1]
    c["metricas_prueba"] = metricas(y, prob)
    vn, fp, fn, vp = confusion_matrix(y, (prob >= 0.5).astype(int)).ravel()
    c["matriz_confusion_prueba"] = {"VN": int(vn), "FP": int(fp), "FN": int(fn), "VP": int(vp)}
    c["probabilidades_distintas"] = sorted({round(float(p), 3) for p in prob})
    c["_y"], c["_prob"] = y, prob
    return c


# ---------------------------------------------------------------- figuras

def fig_importancia(sel):
    imp = pd.Series(sel["importancia"]).sort_values()
    fig, ax = estilo.figura(8.4, 6.2)
    colores = [estilo.AZUL if v > 0.001 else estilo.REJILLA for v in imp.values]
    ax.barh(imp.index, imp.values, color=colores, height=0.7)
    for i, v in enumerate(imp.values):
        if v > 0.001:
            ax.text(v, i, f"  {v:.2f}", va="center", fontsize=9, color=estilo.TINTA)
    estilo.marco(ax, titulo="1. Solo 4 de las 23 variables aportan algo",
                 subtitulo="Gradient boosting con las 23 variables · importancia por permutación en validación",
                 xlabel="aumento del log-loss al revolver la variable", leyenda=False)
    ax.grid(axis="y", visible=False)
    estilo.guardar(fig, "figuras/01-importancia-variables.png")


def fig_curvas(corridas):
    fig, ax = estilo.figura()
    for c, color in zip(corridas, [estilo.AZUL, estilo.NARANJA]):
        etiqueta = "Con satisfacción" if c["variante"] == "completo" else "Sin satisfacción"
        ax.plot(c["curva"]["entrenamiento"], color=color, lw=2, label=f"{etiqueta} · entrenamiento")
        ax.plot(c["curva"]["validacion_interna"], color=color, lw=1.6, ls="--", label=f"{etiqueta} · validación")
    ax.set_yscale("log")
    estilo.marco(ax, titulo="2. Curvas de entrenamiento del gradient boosting",
                 subtitulo="log-loss por iteración · la validación (punteada) queda encima del entrenamiento: no hay sobreajuste",
                 xlabel="iteración (árboles agregados)", ylabel="log-loss (escala logarítmica)")
    estilo.guardar(fig, "figuras/02-curvas-entrenamiento.png")


def fig_prueba(c, numero):
    y, prob = c["_y"], c["_prob"]
    fig, (a1, a2) = estilo.plt.subplots(1, 2, figsize=(10.4, 4.6))
    fig.patch.set_facecolor(estilo.FONDO)
    fpr, tpr, _ = roc_curve(y, prob)
    a1.plot([0, 1], [0, 1], color=estilo.REJILLA, lw=1.2, ls="--", label="Al azar")
    a1.plot(fpr, tpr, color=estilo.AZUL, lw=2.4, label=f"Modelo · AUC {c['metricas_prueba']['auc']:.4f}")
    corto = "con satisfacción" if c["variante"] == "completo" else "sin satisfacción"
    estilo.marco(a1, titulo=f"{numero}. Modelo {corto} · curva ROC",
                 subtitulo=f"{NOMBRES[c['modelo_nombre']]} · {len(y):,} clientes de prueba".replace(",", " "),
                 xlabel="tasa de falsos positivos", ylabel="tasa de verdaderos positivos")
    real, predicha = calibration_curve(y, prob, n_bins=10, strategy="uniform")
    a2.plot([0, 1], [0, 1], color=estilo.REJILLA, lw=1.2, ls="--", label="Calibración perfecta")
    a2.plot(predicha, real, color=estilo.NARANJA, lw=2.4, marker="o", ms=6, label="Modelo")
    estilo.marco(a2, titulo="Calibración",
                 subtitulo="probabilidad del modelo contra la compra real",
                 xlabel="probabilidad predicha", ylabel="frecuencia real de compra")
    estilo.guardar(fig, f"figuras/{numero:02d}-prueba-{c['variante'].replace('_', '-')}.png")


def fig_comparacion(todas):
    variantes = ["completo", "sin_satisfaccion"]
    modelos = ["linea_base", "regresion_logistica", "gradient_boosting"]
    fig, ax = estilo.figura(8.4, 4.4)
    ancho = 0.26
    for i, (m, color) in enumerate(zip(modelos, [estilo.REJILLA, estilo.NARANJA, estilo.AZUL])):
        valores = [next(c for c in todas if c.get("variante") == v and c.get("modelo_nombre") == m)
                   ["metricas_validacion"]["auc"] for v in variantes]
        barras = ax.bar(np.arange(2) + (i - 1) * ancho, valores, ancho * 0.92, color=color, label=NOMBRES[m])
        for b, v in zip(barras, valores):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}", ha="center", fontsize=9, color=estilo.TINTA)
    ax.set_xticks(np.arange(2), [f"Modelo {TITULO_VARIANTE[v]}" for v in variantes])
    ax.set_ylim(0.4, 1.08)
    estilo.marco(ax, titulo="3. Comparación de modelos en validación",
                 subtitulo="AUC en las 20 000 filas de validación · se elige el de menor log-loss",
                 ylabel="AUC")
    ax.grid(axis="x", visible=False)
    estilo.guardar(fig, "figuras/03-comparacion-modelos.png")


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
            params = [Param(k, str(v)) for k, v in {**c["parametros"], **info}.items()]
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
    info = {"semilla": SEMILLA, "sklearn": sklearn.__version__, "filas_entrenamiento": len(partes["entrenamiento"]),
            "filas_validacion": len(partes["validacion"]), "filas_prueba": len(partes["prueba"])}

    seleccion = etapa_seleccion(partes)
    print("importancia (validacion):", dict(list(seleccion["importancia"].items())[:6]))

    corridas, elegidos = [seleccion], {}
    for variante, variables in [("completo", VARIABLES_COMPLETO), ("sin_satisfaccion", VARIABLES_SIN_SATISFACCION)]:
        candidatas = etapa_variante(partes, variante, variables)
        mejor = min(candidatas, key=lambda c: c["metricas_validacion"]["log_loss"])
        evaluar_en_prueba(mejor, partes["prueba"], variables)
        elegidos[variante] = mejor
        joblib.dump(mejor["modelo"], f"app/modelos/{variante}.joblib")
        print(f"elegido {variante}: {mejor['modelo_nombre']} | prueba: " +
              " ".join(f"{k}={v:.4f}" for k, v in mejor["metricas_prueba"].items()))
        corridas += candidatas

    fig_importancia(seleccion)
    fig_curvas([c for c in corridas if "curva" in c and not c["nombre"].startswith("seleccion")])
    fig_comparacion(corridas)
    fig_prueba(elegidos["completo"], 4)
    fig_prueba(elegidos["sin_satisfaccion"], 5)
    seleccion["figuras"] = ["figuras/01-importancia-variables.png", "figuras/02-curvas-entrenamiento.png",
                            "figuras/03-comparacion-modelos.png"]
    elegidos["completo"]["figuras"] = ["figuras/04-prueba-completo.png"]
    elegidos["sin_satisfaccion"]["figuras"] = ["figuras/05-prueba-sin-satisfaccion.png"]

    resumen = {
        "info": info,
        "seleccion": {k: seleccion[k] for k in ("parametros", "metricas_validacion", "importancia")},
        "corridas": [{k: c[k] for k in ("nombre", "parametros", "metricas_validacion", "segundos")} for c in corridas],
        "elegidos": {v: {k: c[k] for k in ("nombre", "metricas_validacion", "metricas_prueba",
                                           "matriz_confusion_prueba", "probabilidades_distintas")}
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
    with open("app/modelos/metadatos.json", "w") as f:
        json.dump({"sklearn": sklearn.__version__,
                   "completo": {"modelo": elegidos["completo"]["modelo_nombre"], "variables": VARIABLES_COMPLETO},
                   "sin_satisfaccion": {"modelo": elegidos["sin_satisfaccion"]["modelo_nombre"],
                                        "variables": VARIABLES_SIN_SATISFACCION}}, f, indent=1)


if __name__ == "__main__":
    main()
