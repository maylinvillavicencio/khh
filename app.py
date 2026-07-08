import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os

# Configuración de la página web
st.set_page_config(page_title="BioAI - Ablación Tumoral", layout="wide")

st.title("🔬 Plataforma Predictiva de Biotransporte: Daño Tisular mediante IA")
st.markdown("""
Esta aplicación web interactiva funciona como un **Modelo Subrogado de Inteligencia Artificial**. 
Utiliza un regresor entrenado con las soluciones numéricas de la ecuación de Bioheat extraídas de COMSOL Multiphysics.
""")

with st.expander("🎯 Oportunidades de IA identificadas en este proyecto de Biotransporte"):
    st.markdown("""
    1. **Predicción del modelado** *(implementada abajo)*: modelo subrogado de regresión polinomial
       multivariable que reemplaza corridas costosas de COMSOL, prediciendo la fracción de daño tisular
       para cualquier combinación tiempo–distancia dentro del rango entrenado, en milisegundos en vez de
       minutos de simulación.
    2. **Análisis avanzado de datos / detección de patrones** *(implementada abajo)*: cuantificación
       objetiva del **efecto heat-sink** del vaso sanguíneo, comparando la velocidad de crecimiento del
       daño (dD/dt) a distintas distancias del electrodo y la meseta térmica de enfriamiento convectivo
       cerca del vaso — patrones difíciles de leer directamente en las tablas de COMSOL.
    3. **Optimización con IA (idea inédita, propuesta a futuro)**: un algoritmo genético podría explorar
       el espacio (voltaje V0, tiempo de ablación) para maximizar el daño tumoral sujeto a la restricción
       de no exceder una temperatura segura en el vaso — permitiendo personalizar el protocolo de
       ablación por paciente según el tamaño y ubicación real de su tumor y vasculatura.
    """)

# --- CARGA DEL MODELO PRE-ENTRENADO (.PKL) ---
# Usamos un validador para avisar si el archivo .pkl se subió correctamente
if os.path.exists('best_model.pkl') and os.path.getsize('best_model.pkl') > 0:
    try:
        modelo_ia = joblib.load('best_model.pkl')
    except Exception as e:
        st.error(f"Error al decodificar el archivo del modelo: {e}. Por favor, vuelve a generar y subir 'best_model.pkl'.")
        st.stop()
else:
    st.error("⚠️ No se encontró el archivo 'best_model.pkl' o el archivo está vacío (0 bytes). Asegúrate de haber ejecutado 'training.py' localmente y haber subido el archivo resultante a GitHub.")
    st.stop()

# --- DATOS HISTÓRICOS COMSOL PARA GRAFICAR ---
tiempos = np.array([0, 0.01, 0.02, 0.04, 0.08, 0.12, 0.2, 0.28, 0.44, 0.6, 0.92, 1.24, 1.88, 2.52, 3.52, 4.52, 5.52, 6.52, 7.52, 8.52, 9.52, 10.52])
dano_4mm = np.array([3.52e-7, 1.78e-4, 3.57e-4, 7.18e-4, 0.0014, 0.0022, 0.0038, 0.0056, 0.0096, 0.0141, 0.0256, 0.0400, 0.0799, 0.1311, 0.2318, 0.3418, 0.4497, 0.5479, 0.6329, 0.7041, 0.7626, 0.8101])
dano_12mm = np.array([3.52e-7, 1.77e-4, 3.56e-4, 7.15e-4, 0.0014, 0.0022, 0.0037, 0.0054, 0.0091, 0.0133, 0.0241, 0.0377, 0.0770, 0.1284, 0.2299, 0.3395, 0.4457, 0.5416, 0.6245, 0.6942, 0.7519, 0.7992])
dano_20mm = np.array([3.52e-7, 1.77e-4, 3.55e-4, 7.11e-4, 0.0014, 0.0021, 0.0036, 0.0052, 0.0086, 0.0123, 0.0211, 0.0313, 0.0574, 0.0884, 0.1449, 0.2049, 0.2650, 0.3232, 0.3783, 0.4298, 0.4775, 0.5216])

# --- INTERFAZ DE USUARIO EN VIVO (SIDEBAR) ---
st.sidebar.header("🕹️ Parámetros de Predicción en Vivo")
st.sidebar.markdown("Modifica las condiciones físicas para evaluar la respuesta de la IA de forma instantánea.")

tiempo_input = st.sidebar.slider("Tiempo de tratamiento (minutos):", 0.0, 10.52, 5.0, step=0.1)
distancia_input = st.sidebar.slider("Distancia analizada desde el electrodo (mm):", 4.0, 20.0, 8.0, step=0.5)

# Predicción puntual en vivo usando el modelo cargado
prediccion_viva = modelo_ia.predict([[tiempo_input, distancia_input]])[0]
prediccion_viva = np.clip(prediccion_viva, 0.0, 1.0) # Restricción de límites físicos

# Mostrar resultados numéricos destacados
col1, col2 = st.columns(2)
with col1:
    st.metric(label="📍 Fracción de Daño Tisular Predicho", value=f"{prediccion_viva:.4f} ({prediccion_viva*100:.2f}%)")
with col2:
    status = "🔴 Necrosis Crítica (>70%)" if prediccion_viva >= 0.7 else "🟡 Lesión Parcial / Tejido Viable"
    st.metric(label="⚠️ Estado Celular Estimado", value=status)

# --- GENERACIÓN DE GRÁFICAS DE ANÁLISIS AVANZADO ---
st.subheader("📊 Análisis Gráfico de Curvas de Daño Continuas")

tiempos_continuos = np.linspace(0, 10.52, 200)

fig, ax = plt.subplots(figsize=(10, 5))

# Graficar datos experimentales/COMSOL históricos
ax.scatter(tiempos, dano_4mm, color='blue', alpha=0.6, label='COMSOL Histórico (4 mm)')
ax.scatter(tiempos, dano_12mm, color='orange', alpha=0.6, label='COMSOL Histórico (12 mm)')
ax.scatter(tiempos, dano_20mm, color='green', alpha=0.6, label='COMSOL Histórico (20 mm)')

# Predicciones dinámicas continuas de la IA
X_dinamico = np.column_stack((tiempos_continuos, np.full_like(tiempos_continuos, distancia_input)))
pred_dinamica = np.clip(modelo_ia.predict(X_dinamico), 0.0, 1.0)

ax.plot(tiempos_continuos, pred_dinamica, color='red', linestyle='--', linewidth=2.5,
        label=f'Predicción IA Continua (Ajustada a {distancia_input} mm)')

# Corrección en la sintaxis de graficación del punto seleccionado (parámetros limpios para evitar errores de Matplotlib)
ax.plot(tiempo_input, prediccion_viva, marker='X', color='black', markersize=12, label='Punto en Vivo Seleccionado')

ax.set_title('Evolución del Daño Tisular: Simulación Física vs. Regresión Matemática de IA', fontsize=12)
ax.set_xlabel('Tiempo de Exposición (min)', fontsize=10)
ax.set_ylabel('Fracción de Daño Celular (0 a 1)', fontsize=10)
ax.grid(True, linestyle=':', alpha=0.6)
ax.legend(loc='upper left')

# Renderizar gráfico en el dashboard web de forma segura
st.pyplot(fig)

# --- SECCIÓN 2: DETECCIÓN DE PATRONES - EFECTO HEAT-SINK POR DISTANCIA ---
st.markdown("---")
st.subheader("🔍 Detección de Patrones: Efecto de la Distancia sobre la Velocidad de Daño")
st.markdown("""
Este análisis compara la **velocidad de crecimiento del daño tisular** (dD/dt) en los tres puntos de
sonda de COMSOL, ubicados a distintas distancias del electrodo de ablación. Permite identificar cuánto
se atenúa la eficacia térmica conforme el tejido se aleja de la fuente de calor — un patrón que sería
difícil de cuantificar mirando solo las curvas crudas.
""")

rate_4mm = np.gradient(dano_4mm, tiempos)
rate_12mm = np.gradient(dano_12mm, tiempos)
rate_20mm = np.gradient(dano_20mm, tiempos)

col3, col4, col5 = st.columns(3)
with col3:
    st.metric("Daño final @ 4 mm", f"{dano_4mm[-1]*100:.1f}%")
with col4:
    st.metric("Daño final @ 12 mm", f"{dano_12mm[-1]*100:.1f}%",
               delta=f"-{(dano_4mm[-1]-dano_12mm[-1])*100:.1f} pp vs 4mm")
with col5:
    st.metric("Daño final @ 20 mm", f"{dano_20mm[-1]*100:.1f}%",
               delta=f"-{(dano_4mm[-1]-dano_20mm[-1])*100:.1f} pp vs 4mm")

fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(12, 4.5))

ax2a.plot(tiempos, dano_4mm*100, 'o-', color='blue', label='4 mm')
ax2a.plot(tiempos, dano_12mm*100, 's-', color='orange', label='12 mm')
ax2a.plot(tiempos, dano_20mm*100, '^-', color='green', label='20 mm')
ax2a.set_xlabel('Tiempo (min)')
ax2a.set_ylabel('Fracción de daño (%)')
ax2a.set_title('Daño acumulado por distancia')
ax2a.legend()
ax2a.grid(True, linestyle=':', alpha=0.5)

ax2b.plot(tiempos, rate_4mm, 'o-', color='blue', label='4 mm')
ax2b.plot(tiempos, rate_12mm, 's-', color='orange', label='12 mm')
ax2b.plot(tiempos, rate_20mm, '^-', color='green', label='20 mm')
ax2b.set_xlabel('Tiempo (min)')
ax2b.set_ylabel('dD/dt (velocidad de daño)')
ax2b.set_title('Patrón detectado: atenuación de velocidad con la distancia')
ax2b.legend()
ax2b.grid(True, linestyle=':', alpha=0.5)

plt.tight_layout()
st.pyplot(fig2)

reduccion_pct = (1 - dano_20mm[-1]/dano_4mm[-1]) * 100
st.info(f"""
**Patrón identificado:** entre 4 mm y 20 mm, el daño final cae de {dano_4mm[-1]*100:.1f}% a
{dano_20mm[-1]*100:.1f}% (una reducción de {reduccion_pct:.0f}%), y el pico de velocidad de daño
(dD/dt) se retrasa y se aplana con la distancia — evidencia cuantitativa del **efecto heat-sink**
producido por la conducción térmica limitada y la perfusión sanguínea (ω_b = 6.4×10⁻³ s⁻¹, según
`parametros.txt`).
""")

# --- SECCIÓN 3: EFECTO DE ENFRIAMIENTO CONVECTIVO CERCA DEL VASO (vasoo.txt) ---
st.markdown("---")
st.subheader("🩸 Efecto del Vaso Sanguíneo: Enfriamiento Convectivo (Heat-Sink)")
st.markdown("""
Se graficaron 3 corridas paramétricas de COMSOL en un punto cercano al vaso sanguíneo. En vez de que
la temperatura ascienda sin control como en el tumor, **se estabiliza en una meseta cercana a 103 °C**
— la firma térmica del heat-sink por convección del flujo sanguíneo (perfusión ω_b y vaso de
D_vaso = 5 mm, según `parametros.txt`).
""")

t_vaso = np.array([0, 2.5, 5, 7.5, 10])
temp_A = np.array([37.01357406218045, 99.25661184628018, 102.28051861732422, 102.98079064649181, 103.16208754938538])
temp_B = np.array([37.01343745229815, 99.78851398109765, 102.48551032319796, 103.1450506874948, 103.32238392224195])
temp_C = np.array([37.01332988963907, 99.38768378853024, 102.41853714749703, 103.12056757724878, 103.30220118154881])

fig3, ax3 = plt.subplots(figsize=(9, 4.5))
ax3.plot(t_vaso, temp_A, 'o-', label='Escenario A')
ax3.plot(t_vaso, temp_B, 's-', label='Escenario B')
ax3.plot(t_vaso, temp_C, '^-', label='Escenario C')
ax3.axhline(103, color='red', linestyle='--', alpha=0.5, label='Meseta ≈103 °C (heat-sink)')
ax3.set_xlabel('Tiempo (min)')
ax3.set_ylabel('Temperatura (°C)')
ax3.set_title('Meseta térmica cerca del vaso sanguíneo')
ax3.legend()
ax3.grid(True, linestyle=':', alpha=0.5)
st.pyplot(fig3)

st.warning("""
⚠️ **Nota metodológica:** las 3 corridas corresponden a un barrido paramétrico en COMSOL (sobre
D_vaso y/o ω_b, los parámetros de `parametros.txt`). Como las diferencias entre corridas son pequeñas
(menores a 0.2 °C en la meseta final), esto sugiere que el sistema es **robusto** ante variaciones
menores en el diámetro del vaso o la perfusión sanguínea — el efecto heat-sink domina
independientemente del valor exacto dentro de ese rango explorado.
""")

st.markdown("""
---
### 🛠️ Diagrama de Flujo del Proceso
`COMSOL Multiphysics (Datos Numéricos)` ➡️ `Extracción de Sondas de Dominio` ➡️ `Pipeline Polinomial en Python` ➡️ `Detección de Patrones (dD/dt, meseta térmica)` ➡️ `Despliegue en Streamlit Web App`
""")
