# Monitoreo Predictivo de Ablación Tumoral mediante IA
### v2 — Grupo 2, Biotransporte (RFA hepática)

App de Streamlit que funciona como **modelo subrogado** de la simulación COMSOL Multiphysics
del proyecto (ecuación de biocalor de Pennes + cinética de daño de Arrhenius), para predecir
en tiempo real lo que antes solo se podía obtener corriendo la simulación de elementos finitos.

## 🆕 Cambios respecto a la versión original

1. **Lectura directa de los `.txt` exportados de COMSOL** (`fracionamiento_de_daño.txt`,
   `vasoo.txt`). Ya no hay ningún valor transcrito a mano en `training.py`; si se vuelve a
   correr COMSOL y se reemplazan los `.txt`, basta con volver a correr `training.py`.

2. **Segundo modelo agregado — Efecto Heat-Sink**: predice la temperatura en la pared vascular
   en función del diámetro del vaso (1-5 mm) y el tiempo, usando `vasoo.txt` (antes la app solo
   tenía el modelo de daño tisular).

3. **Elección de algoritmo justificada por curva, no por costumbre**:
   - *Daño tisular (r, t) → Ω*: se mantiene el pipeline polinomial grado 3 original — la curva
     de Arrhenius es suave y monótona, y el polinomio la ajusta bien (RMSE ≈ 0.009).
   - *Heat-sink (D, t) → T*: se reemplazó el polinomio por **Gaussian Process Regression**. Se
     probaron polinomios de grado 2 (subajusta la meseta en ~5°C), grado 3 (oscila de forma
     irreal entre puntos) y grado 4 (sobreajuste severo tipo Runge, predice hasta una caída de
     temperatura con el tiempo). El GPR da RMSE ≈ 0.12-0.35°C sin oscilaciones.

4. **Limitación documentada explícitamente en la app**: con solo 3 radios simulados (4, 12,
   20 mm), cualquier modelo suave genera un pequeño sobre-pico artificial (~3-4%) alrededor de
   r≈8mm, porque el daño real es casi plano entre 4-12mm y cae fuerte después. Esto se explica
   en un expander dentro de la app en vez de ocultarse.

5. **Nueva pestaña de sensibilidad a la potencia**, claramente marcada como **extensión
   analítica y NO como modelo de IA entrenado** (los datos disponibles son solo a V0=22V). Se
   deriva de la propia ecuación de Joule del proyecto (Q_RF = σ|∇V|², sistema lineal en la
   fuente) para estimar cómo escalaría la meseta térmica con el voltaje, dejando explícito que
   no reemplaza una simulación real.

## 🚀 Características de la App

* **Pestaña 1 — Daño Tisular:** estimación en vivo de la fracción de daño (0 a 1) según tiempo
  y distancia al electrodo, con nota de limitación de datos.
* **Pestaña 2 — Efecto Heat-Sink:** temperatura predicha en la pared vascular según diámetro
  del vaso y tiempo, con hallazgo de que el diámetro casi no influye a 22V.
* **Pestaña 3 — Sensibilidad a la Potencia:** extensión teórica (no entrenada) de cómo
  escalaría la meseta térmica con el voltaje, basada en la física del propio modelo.

## 🛠️ Arranque rápido

1. Instalar dependencias: `pip install -r requirements.txt`
2. Correr el entrenamiento (regenera `best_model.pkl`, `tumor_data.csv`, `vessel_data.csv`
   directamente desde los `.txt`): `python training.py`
3. Lanzar la aplicación web: `streamlit run app.py`

**Archivos necesarios en la misma carpeta:** `fracionamiento_de_daño.txt`, `vasoo.txt`
(exportados de COMSOL), además de los `.py` de este repositorio.
