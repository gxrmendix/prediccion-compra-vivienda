"""Division 70 % entrenamiento / 10 % validacion / 20 % prueba, estratificada por la decision.

Se hace antes de cualquier preprocesamiento para que nada de prueba influya en el modelo.
Tambien comprueba que la formula de app/caracteristicas.py reproduce emi_to_income_ratio.

Uso: python src/preparar_datos.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, "app")
from caracteristicas import proporcion_mensualidad_ingreso  # noqa: E402

SEMILLA = 42


def main():
    d = pd.read_csv("datos/global_house_purchase_dataset.csv")
    calculada = proporcion_mensualidad_ingreso(d.loan_amount, d.loan_tenure_years, d.customer_salary)
    coincide = float(np.mean(np.abs(calculada - d.emi_to_income_ratio) < 0.005))

    entrenamiento, resto = train_test_split(d, test_size=0.30, stratify=d.decision, random_state=SEMILLA)
    # del 30 % restante: un tercio es validacion (10 % del total) y dos tercios prueba (20 %)
    validacion, prueba = train_test_split(resto, test_size=2 / 3, stratify=resto.decision, random_state=SEMILLA)
    for nombre, parte in [("entrenamiento", entrenamiento), ("validacion", validacion), ("prueba", prueba)]:
        parte.to_parquet(f"datos/{nombre}.parquet", index=False)

    resumen = {
        "filas": len(d), "columnas": d.shape[1], "nulos": int(d.isna().sum().sum()),
        "tasa_compra": round(float(d.decision.mean()), 4),
        "particion": {n: {"filas": len(p), "porcentaje": round(100 * len(p) / len(d), 1),
                          "tasa_compra": round(float(p.decision.mean()), 4)}
                      for n, p in [("entrenamiento", entrenamiento), ("validacion", validacion), ("prueba", prueba)]},
        "formula_mensualidad_coincide": coincide,
    }
    os.makedirs("resultados", exist_ok=True)
    with open("resultados/preparacion.json", "w") as f:
        json.dump(resumen, f, indent=1)
    print(json.dumps(resumen, indent=1))


if __name__ == "__main__":
    main()
