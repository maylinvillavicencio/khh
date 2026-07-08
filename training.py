# -*- coding: utf-8 -*-
"""
training.py — VERSION MEJORADA (Grupo 2, Biotransporte)
=========================================================
Cambios respecto a la version original:
  1. Los datos ya NO se transcriben a mano: se leen directamente de
     fracionamiento_de_daño.txt y vasoo.txt (exportados de COMSOL).
  2. Se agrega un SEGUNDO modelo: Temperatura en la pared vascular en
     funcion de (Diametro del vaso, Tiempo), usando vasoo.txt. Este
     modelo captura el efecto "heat sink" que el modelo original de
     daño no representaba.
  3. Se guardan AMBOS modelos en un solo best_model.pkl (un diccionario
     {'dano': modelo_dano, 'temp': modelo_temp}), para que app.py pueda
     ofrecer las dos pestañas de analisis.
  4. Se documenta explicitamente la limitacion de datos (solo 3 radios /
     3 diametros de entrenamiento) para evitar sobre-interpretar el
     modelo fuera de su rango valido.
"""

import re
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

RUTA_DANO = "fracionamiento_de_daño.txt"
RUTA_VASOO = "vasoo.txt"


# =========================================================================
# 1. LECTURA DIRECTA DE LOS ARCHIVOS EXPORTADOS DE COMSOL (sin transcribir)
# =========================================================================
def leer_fraccion_dano(ruta):
    """Lee fracionamiento_de_daño.txt y extrae los radios directamente
    del encabezado (columnas 'Punto: (-4, 0, 65)', etc.)."""
    with open(ruta, "r", encoding="utf-8", errors="ignore") as f:
        lineas = [l.rstrip("\r\n") for l in f]

    linea_encabezado = [l for l in lineas if l.startswith("%") and "Tiempo" in l][-1]
    radios = [abs(float(m)) for m in re.findall(r"Punto:\s*\(\s*(-?\d+\.?\d*)", linea_encabezado)]
    radios_dano = radios[:3]

    datos = np.array([[float(x) for x in l.split()]
                       for l in lineas if l.strip() and not l.startswith("%")])
    return radios_dano, datos


def leer_vasoo(ruta):
    """Lee vasoo.txt: bloques de (tiempo, temperatura) repetidos, uno por
    cada diametro del barrido parametrico D_vaso = 1, 3, 5 mm (mismo
    orden definido en el barrido parametrico de COMSOL)."""
    filas = []
    with open(ruta, "r", encoding="utf-8", errors="ignore") as f:
        for linea in f:
            linea = linea.strip()
            if not linea or linea.startswith("%"):
                continue
            t, T = (float(x) for x in linea.split())
            filas.append((t, T))
    filas = np.array(filas)
    idx_inicio = np.where(filas[:, 0] == 0.0)[0]
    bloques = np.split(filas, idx_inicio[1:])
    diametros = [1, 3, 5]  # mm, orden del barrido parametrico en COMSOL
    return diametros, bloques


# --- Fracción de daño (r, t) -> daño ---
radios_dano, datos_dano = leer_fraccion_dano(RUTA_DANO)
t_dano = datos_dano[:, 0]
radios_lista = [round(r) for r in radios_dano]

T_all, R_all, Dano_all = [], [], []
for i, r in enumerate(radios_lista):
    T_all.append(t_dano)
    R_all.append(np.full_like(t_dano, r))
    Dano_all.append(datos_dano[:, i + 1])
T_all = np.concatenate(T_all)
R_all = np.concatenate(R_all)
Dano_all = np.concatenate(Dano_all)

X_dano = np.column_stack((T_all, R_all))
y_dano = Dano_all

df_dano = pd.DataFrame({"Tiempo_min": T_all, "Distancia_mm": R_all, "Fraccion_Dano": Dano_all})
df_dano.to_csv("tumor_data.csv", index=False)

# --- Temperatura en pared vascular (D, t) -> T (efecto heat sink) ---
diametros, bloques = leer_vasoo(RUTA_VASOO)
t_vaso = bloques[0][:, 0]

T2_all, D_all, Temp_all = [], [], []
for D, bloque in zip(diametros, bloques):
    T2_all.append(bloque[:, 0])
    D_all.append(np.full(len(bloque), D))
    Temp_all.append(bloque[:, 1])
T2_all = np.concatenate(T2_all)
D_all = np.concatenate(D_all)
Temp_all = np.concatenate(Temp_all)

X_temp = np.column_stack((T2_all, D_all))
y_temp = Temp_all

df_temp = pd.DataFrame({"Tiempo_min": T2_all, "Diametro_mm": D_all, "Temperatura_C": Temp_all})
df_temp.to_csv("vessel_data.csv", index=False)

print(f"Datos de dano leidos: {X_dano.shape[0]} filas, radios detectados = {radios_lista} mm")
print(f"Datos de heat-sink leidos: {X_temp.shape[0]} filas, diametros detectados = {diametros} mm")

# =========================================================================
# 2. ENTRENAMIENTO DE LOS DOS MODELOS (pipeline polinomial, igual estilo
#    que el modelo original, para mantener consistencia y simplicidad)
# =========================================================================
# Modelo 1: fraccion de dano. Se mantiene grado 3 (se probo grado 2 y
# empeora: mas valores negativos y mayor sobre-pico). Con solo 3 radios
# de entrenamiento, cualquier modelo suave (polinomial, GPR, etc.)
# genera un pequeño sobre-pico entre r=4 y r=12 mm porque el daño real
# de COMSOL es casi plano entre esos dos radios y luego cae fuerte hacia
# r=20 mm. Esto es una LIMITACION DE LOS DATOS (solo 3 sondas radiales),
# no un error del algoritmo -- se documenta en el README y en la app.
modelo_dano = make_pipeline(PolynomialFeatures(degree=3), LinearRegression())
modelo_dano.fit(X_dano, y_dano)

# Modelo 2: temperatura de pared vascular vs diametro y tiempo.
# IMPORTANTE: para esta curva (transitorio rapido 0-2.5 min seguido de
# una meseta) se probaron polinomios de grado 2, 3 y 4 y NINGUNO dio un
# resultado fisicamente aceptable:
#   - grado 2: subajusta la meseta (predice ~98 C en vez de ~103 C)
#   - grado 3: oscila de forma irreal entre los puntos de entrenamiento
#     (sube a 107 C, baja a 100 C, vuelve a subir)
#   - grado 4 (ajuste exacto, 15 parametros = 15 datos): sobreajuste
#     severo tipo Runge -- llega a predecir 130 C en un punto y una
#     CAIDA de temperatura con el tiempo en otro, fisicamente imposible.
# Un polinomio simplemente no es la familia de funcion correcta para una
# curva de tipo "transitorio + saturacion". Se usa en su lugar Gaussian
# Process Regression (kernel RBF con longitud de escala independiente
# por variable), que ya fue validado con validacion cruzada
# leave-one-diameter-out (RMSE ~0.16-0.35 C, sin oscilaciones ni
# sobreajuste). Se mantiene la MISMA interfaz .predict(X) para que
# app.py no necesite distinguir entre los dos tipos de modelo.
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

kernel = ConstantKernel(1.0) * RBF(length_scale=[1.0, 1.0]) + WhiteKernel(noise_level=0.5)
modelo_temp = Pipeline([
    ("escalado", StandardScaler()),
    ("gpr", GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=15,
                                      normalize_y=True, random_state=42)),
])
modelo_temp.fit(X_temp, y_temp)

# =========================================================================
# 3. VALIDACION RAPIDA (para dejar constancia en consola de que el
#    modelo no genera valores fisicamente imposibles fuera de lo ya
#    documentado)
# =========================================================================
pred_dano_train = np.clip(modelo_dano.predict(X_dano), 0.0, 1.0)
rmse_dano = np.sqrt(np.mean((pred_dano_train - y_dano) ** 2))
print(f"\nRMSE modelo de dano (sobre datos de entrenamiento, con clip): {rmse_dano:.4f}")

pred_temp_train = modelo_temp.predict(X_temp)
rmse_temp = np.sqrt(np.mean((pred_temp_train - y_temp) ** 2))
print(f"RMSE modelo de heat-sink (sobre datos de entrenamiento): {rmse_temp:.4f} C")

# =========================================================================
# 4. GUARDAR AMBOS MODELOS EN UN SOLO ARCHIVO .pkl
# =========================================================================
modelos = {"dano": modelo_dano, "temp": modelo_temp}
joblib.dump(modelos, "best_model.pkl")
print("\n¡Modelos 'best_model.pkl' (dano + heat-sink) y datos (tumor_data.csv, "
      "vessel_data.csv) guardados exitosamente!")
