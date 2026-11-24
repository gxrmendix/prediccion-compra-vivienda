"""
caracteristicas.py
~~~~~~~~~~~~~~~~~~

Variables del modelo, codificacion de categorias y calculos del credito hipotecario.
Lo usan tanto el entrenamiento (src/entrenar.py) como la app, para que el servidor arme
exactamente el mismo vector con el que se entreno el modelo.

Todo trabaja con arreglos de numpy (sin pandas) para que la app quepa holgada en el plan
gratuito de Azure: las categorias se convierten a enteros con un orden fijo y el gradient
boosting las trata como categoricas.
"""
import numpy as np

# Tasa anual que reproduce exactamente la columna emi_to_income_ratio del dataset
# (se comprobo en las 200 000 filas: mensualidad de amortizacion al 6.5 % anual / salario mensual).
TASA_ANUAL = 0.065
PIES2_POR_M2 = 10.7639

# Valores posibles de las variables categoricas, en orden fijo (su posicion es el codigo)
CATEGORIAS = {
    "country": ["Australia", "Brazil", "Canada", "China", "France", "Germany", "India", "Japan", "Singapore",
                "South Africa", "UAE", "UK", "USA"],
    "city": ["Abu Dhabi", "Bangalore", "Beijing", "Berlin", "Birmingham", "Brisbane", "Cape Town", "Chennai",
             "Chicago", "Delhi", "Dubai", "Frankfurt", "Houston", "Hyderabad", "Johannesburg", "Kyoto", "Liverpool",
             "London", "Los Angeles", "Lyon", "Manchester", "Marseille", "Melbourne", "Montreal", "Mumbai", "Munich",
             "New York", "Osaka", "Paris", "Pune", "Rio de Janeiro", "San Francisco", "Shanghai", "Shenzhen",
             "Singapore", "Sydney", "São Paulo", "Tokyo", "Toronto", "Vancouver"],
    "property_type": ["Apartment", "Farmhouse", "Independent House", "Studio", "Townhouse", "Villa"],
    "furnishing_status": ["Fully-Furnished", "Semi-Furnished", "Unfurnished"],
}

# Orden de las columnas del modelo. La satisfaccion va al final para que el modelo
# "antes de la visita" sea el mismo vector sin la ultima columna.
VARIABLES_SIN_SATISFACCION = [
    "country", "city", "property_type", "furnishing_status", "property_size_sqft", "price", "constructed_year",
    "previous_owners", "rooms", "bathrooms", "garage", "garden", "crime_cases_reported", "legal_cases_on_property",
    "customer_salary", "loan_amount", "loan_tenure_years", "monthly_expenses", "down_payment", "emi_to_income_ratio",
    "neighbourhood_rating", "connectivity_score",
]
VARIABLES_COMPLETO = VARIABLES_SIN_SATISFACCION + ["satisfaction_score"]
CATEGORICAS = [c in CATEGORIAS for c in VARIABLES_COMPLETO]  # mascara para el gradient boosting


def codificar(filas):
    """Convierte una lista de diccionarios (o un DataFrame) en la matriz de numeros del modelo.

    Las categorias pasan a su posicion en CATEGORIAS; si se pasa un DataFrame se usan sus columnas.
    Devuelve una matriz con las 23 columnas de VARIABLES_COMPLETO (la satisfaccion puede faltar:
    queda como nan y el modelo sin satisfaccion simplemente no la usa).
    """
    if hasattr(filas, "columns"):  # DataFrame del entrenamiento: version vectorizada
        import pandas as pd
        columnas = []
        for var in VARIABLES_COMPLETO:
            if var not in filas.columns:
                columnas.append(np.full(len(filas), np.nan))
            elif var in CATEGORIAS:
                codigos = pd.Categorical(filas[var].astype(str), categories=CATEGORIAS[var]).codes
                assert (codigos >= 0).all(), f"categoria desconocida en {var}"
                columnas.append(codigos.astype(float))
            else:
                columnas.append(filas[var].to_numpy(dtype=float))
        return np.column_stack(columnas)
    matriz = np.full((len(filas), len(VARIABLES_COMPLETO)), np.nan)
    for i, fila in enumerate(filas):
        for j, var in enumerate(VARIABLES_COMPLETO):
            valor = fila.get(var)
            if valor is None:
                continue
            matriz[i, j] = CATEGORIAS[var].index(str(valor)) if var in CATEGORIAS else float(valor)
    return matriz


def factor_mensualidad(plazo_anios, tasa_anual=TASA_ANUAL):
    """Pago mensual por cada unidad de credito (formula de amortizacion francesa)."""
    r = tasa_anual / 12
    n = np.asarray(plazo_anios) * 12
    return r * (1 + r) ** n / ((1 + r) ** n - 1)


def mensualidad(credito, plazo_anios):
    return np.asarray(credito) * factor_mensualidad(plazo_anios)


def proporcion_mensualidad_ingreso(credito, plazo_anios, ingreso_mensual):
    """Mensualidad entre ingreso mensual, redondeada a 2 decimales como en el dataset."""
    return np.round(mensualidad(credito, plazo_anios) / np.asarray(ingreso_mensual), 2)


def credito_maximo(ingreso_mensual, plazo_anios, proporcion=0.39):
    """Credito mas alto cuya mensualidad no pasa de `proporcion` del ingreso."""
    return proporcion * ingreso_mensual / factor_mensualidad(plazo_anios)
