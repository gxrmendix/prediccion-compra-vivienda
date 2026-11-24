"""Variables del modelo y calculo de la mensualidad, compartidos por el entrenamiento y la app.

Vive dentro de app/ para que el servidor de Azure use exactamente el mismo calculo con el
que se entreno el modelo.
"""
import numpy as np

# Tasa anual que reproduce exactamente la columna emi_to_income_ratio del dataset
# (se comprobo en las 200 000 filas: mensualidad de amortizacion al 6.5 % anual / salario mensual).
TASA_ANUAL = 0.065

# Variables finales, elegidas por importancia en validacion (ver entrenar.py)
VARIABLES_COMPLETO = ["satisfaction_score", "legal_cases_on_property", "crime_cases_reported", "emi_to_income_ratio"]
VARIABLES_SIN_SATISFACCION = ["legal_cases_on_property", "crime_cases_reported", "emi_to_income_ratio"]


def mensualidad(prestamo, plazo_anios, tasa_anual=TASA_ANUAL):
    """Pago mensual de un prestamo amortizado (formula francesa)."""
    r = tasa_anual / 12
    n = np.asarray(plazo_anios) * 12
    return np.asarray(prestamo) * r * (1 + r) ** n / ((1 + r) ** n - 1)


def proporcion_mensualidad_ingreso(prestamo, plazo_anios, salario_mensual):
    """Mensualidad entre salario mensual, redondeada a 2 decimales como en el dataset."""
    return np.round(mensualidad(prestamo, plazo_anios) / np.asarray(salario_mensual), 2)
