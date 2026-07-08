# Monitoreo Predictivo de Ablación Tumoral mediante IA

App de Streamlit diseñada para el análisis de biotransporte térmico en bioingeniería. Funciona como un modelo subrogado para predecir la necrosis tisular acumulada a partir de simulaciones en COMSOL Multiphysics.

### 🚀 Características de la App:
* **Estimación en Vivo:** Calcula la fracción de daño tisular (0 a 1) interactuando con el tiempo y la distancia al electrodo.
* **Modelado Continuo:** Resuelve el "efecto escalón" interpolando curvas suaves mediante regresión polinomial multivariable.
* **Análisis Clínico:** Identifica de forma inmediata si las coordenadas ingresadas se encuentran en estado de necrosis crítica o tejido viable.

### 🛠️ Arranque rápido
1. Instalar dependencias: `pip install -r requirements.txt`
2. Correr el entrenamiento (opcional): `python training.py`
3. Lanzar la aplicación web: `streamlit run app.py`