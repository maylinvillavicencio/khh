# Monitoreo Predictivo de Ablación Tumoral mediante IA

App de Streamlit para el análisis de biotransporte térmico en bioingeniería. Combina un **modelo
subrogado de IA** con **detección de patrones** para analizar simulaciones de ablación por
radiofrecuencia (RF) obtenidas en COMSOL Multiphysics.

### 🎯 Oportunidades de IA implementadas

1. **Predicción del modelado**: modelo subrogado (regresión polinomial grado 3) entrenado con 66
   puntos reales de COMSOL, que predice la fracción de daño tisular para cualquier combinación
   tiempo–distancia sin volver a simular.
2. **Detección de patrones**: cuantifica el efecto *heat-sink* del vaso sanguíneo comparando la
   velocidad de daño (dD/dt) por distancia al electrodo y la meseta térmica de enfriamiento
   convectivo cerca del vaso.
3. **Optimización con IA (propuesta a futuro)**: algoritmo genético para hallar el par
   (voltaje V0, tiempo de ablación) que maximice el daño tumoral sin exceder una temperatura
   segura en el vaso.

### 🚀 Características de la App

* **Estimación en vivo**: calcula la fracción de daño tisular (0 a 1) según tiempo y distancia
  al electrodo.
* **Modelado continuo**: interpola curvas suaves mediante regresión polinomial multivariable.
* **Análisis clínico**: identifica si las coordenadas ingresadas están en necrosis crítica o
  tejido viable.
* **Detección de patrones**: gráficas de velocidad de daño (dD/dt) y comparación por distancia.
* **Efecto heat-sink**: visualiza la meseta térmica cerca del vaso sanguíneo (`data/vasoo.txt`).

### 📁 Estructura del proyecto

```
├── app.py                          # App de Streamlit (predicción + patrones + heat-sink)
├── training.py                     # Entrena el modelo subrogado
├── best_model.pkl                  # Modelo entrenado (regresión polinomial grado 3)
├── tumor_data.csv                  # Datos de entrenamiento exportados (66 puntos)
├── requirements.txt                # Dependencias
└── data/                           # Datos crudos de COMSOL (trazabilidad)
    ├── fracionamiento_de_daño.txt
    ├── vasoo.txt
    └── parametros.txt
```

### 🛠️ Arranque rápido

1. Instalar dependencias: `pip install -r requirements.txt`
2. Correr el entrenamiento (opcional, ya incluye `best_model.pkl` generado): `python training.py`
3. Lanzar la aplicación web: `streamlit run app.py`

### ⚠️ Limitaciones

* Dataset pequeño (66 puntos); riesgo de extrapolación fuera de 4–20 mm o >10.52 min.
* No incluye datos clínicos/experimentales reales — es un sustituto de la simulación numérica,
  no del paciente.
