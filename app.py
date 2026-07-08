import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os

# =============================================================================
# CONFIGURACION DE LA PAGINA
# =============================================================================
st.set_page_config(page_title="BioAI - Ablación Tumoral", layout="wide")

st.title("🔬 Plataforma Predictiva de Biotransporte: Ablación Tumoral mediante IA")
st.markdown("""
Esta aplicación web interactiva funciona como un **Modelo Subrogado de Inteligencia Artificial**.
Utiliza regresores entrenados con las soluciones numéricas del modelo COMSOL Multiphysics
(ecuación de biocalor de Pennes + cinética de daño de Arrhenius) para predecir, en tiempo real,
lo que antes solo se podía obtener corriendo una simulación de elementos finitos.
""")

# =============================================================================
# CARGA DE LOS MODELOS PRE-ENTRENADOS (.PKL)
# =============================================================================
if os.path.exists("best_model.pkl") and os.path.getsize("best_model.pkl") > 0:
    try:
        modelos = joblib.load("best_model.pkl")
        modelo_dano = modelos["dano"]
        modelo_temp = modelos["temp"]
    except Exception as e:
        st.error(f"Error al decodificar 'best_model.pkl': {e}. Vuelve a correr training.py.")
        st.stop()
else:
    st.error("⚠️ No se encontró 'best_model.pkl'. Ejecuta primero `python training.py` "
              "(debe estar en la misma carpeta que fracionamiento_de_daño.txt y vasoo.txt).")
    st.stop()

# =============================================================================
# DATOS HISTORICOS DE COMSOL (para graficar los puntos reales sobre las
# curvas predichas). Se leen de los CSV generados por training.py a partir
# de los .txt originales -- ya no hay ningun valor transcrito a mano aqui.
# =============================================================================
df_dano = pd.read_csv("tumor_data.csv")
df_temp = pd.read_csv("vessel_data.csv")

radios_comsol = sorted(df_dano["Distancia_mm"].unique())
diametros_comsol = sorted(df_temp["Diametro_mm"].unique())

tab1, tab2, tab3 = st.tabs([
    "🔥 Daño Tisular (Arrhenius)",
    "🩸 Efecto Heat-Sink (Diámetro Vascular)",
    "⚡ Sensibilidad a la Potencia (extensión teórica)",
])

# =============================================================================
# PESTAÑA 1 — DAÑO TISULAR (modelo original, mejorado)
# =============================================================================
with tab1:
    st.sidebar.header("🕹️ Parámetros — Daño Tisular")
    tiempo_input = st.sidebar.slider("Tiempo de tratamiento (min):", 0.0, 10.52, 5.0, step=0.1,
                                       key="t_dano")
    distancia_input = st.sidebar.slider("Distancia al electrodo (mm):", 4.0, 20.0, 8.0, step=0.5,
                                          key="r_dano")

    pred_viva = float(np.clip(modelo_dano.predict([[tiempo_input, distancia_input]])[0], 0.0, 1.0))

    col1, col2 = st.columns(2)
    with col1:
        st.metric("📍 Fracción de Daño Tisular Predicho", f"{pred_viva:.4f} ({pred_viva*100:.2f}%)")
    with col2:
        status = "🔴 Necrosis Crítica (>70%)" if pred_viva >= 0.7 else "🟡 Lesión Parcial / Tejido Viable"
        st.metric("⚠️ Estado Celular Estimado", status)

    st.subheader("📊 Curvas de Daño (COMSOL vs. predicción IA)")
    t_cont = np.linspace(0, 10.52, 200)
    fig, ax = plt.subplots(figsize=(10, 5))
    colores = {radios_comsol[0]: "blue", radios_comsol[1]: "orange", radios_comsol[2]: "green"}
    for r in radios_comsol:
        sub = df_dano[df_dano["Distancia_mm"] == r]
        ax.scatter(sub["Tiempo_min"], sub["Fraccion_Dano"], color=colores[r], alpha=0.6,
                    label=f"COMSOL histórico ({r:.0f} mm)")
    Xc = np.column_stack((t_cont, np.full_like(t_cont, distancia_input)))
    pred_c = np.clip(modelo_dano.predict(Xc), 0.0, 1.0)
    ax.plot(t_cont, pred_c, "r--", linewidth=2.5, label=f"Predicción IA continua ({distancia_input} mm)")
    ax.plot(tiempo_input, pred_viva, "kX", markersize=12, label="Punto en vivo seleccionado")
    ax.set_title("Evolución del Daño Tisular: Simulación Física vs. Regresión de IA")
    ax.set_xlabel("Tiempo de exposición (min)"); ax.set_ylabel("Fracción de daño celular (0 a 1)")
    ax.grid(True, linestyle=":", alpha=0.6); ax.legend(loc="upper left")
    st.pyplot(fig)

    with st.expander("⚠️ Limitación importante del modelo (léase antes de interpretar clínicamente)"):
        st.markdown("""
        El modelo se entrenó con **solo 3 radios simulados en COMSOL** (4, 12 y 20 mm). Entre
        r=4 mm y r=12 mm, el daño real de COMSOL es casi plano (0.810 → 0.799, apenas 1% de
        diferencia), pero entre r=12 mm y r=20 mm cae fuerte (0.799 → 0.522). Cualquier modelo
        suave (este polinomio, pero también un Gaussian Process — se probó ambos) que intente
        conectar "casi plano, luego caída brusca" genera un pequeño sobre-pico artificial
        alrededor de r≈8 mm (~3-4% por encima del valor real en r=4mm). **Esto es una limitación
        de tener solo 3 sondas espaciales en la simulación, no un error del algoritmo.** Para
        eliminarlo haría falta correr COMSOL con más sondas radiales (p.ej. cada 2 mm).
        """)

# =============================================================================
# PESTAÑA 2 — EFECTO HEAT-SINK (nueva, usando vasoo.txt)
# =============================================================================
with tab2:
    st.markdown("""
    Este modelo predice la **temperatura en la pared del vaso sanguíneo** en función de su
    diámetro y del tiempo de ablación, usando los datos del barrido paramétrico de COMSOL
    (D_vaso = 1, 3, 5 mm). A diferencia del modelo de daño, aquí se usa **Gaussian Process
    Regression** en vez de un polinomio — se probó con polinomios de grado 2, 3 y 4 y todos
    fallaron (ver nota abajo), porque esta curva tiene forma de "transitorio + meseta", no
    polinomial.
    """)

    st.sidebar.header("🕹️ Parámetros — Efecto Heat-Sink")
    tiempo_input2 = st.sidebar.slider("Tiempo de tratamiento (min):", 0.0, 10.0, 5.0, step=0.1,
                                        key="t_temp")
    diametro_input = st.sidebar.slider("Diámetro del vaso (mm):", 1.0, 5.0, 2.0, step=0.5,
                                         key="d_temp")

    pred_temp_viva = float(modelo_temp.predict([[tiempo_input2, diametro_input]])[0])

    col1, col2 = st.columns(2)
    with col1:
        st.metric("🌡️ Temperatura Predicha en Pared Vascular", f"{pred_temp_viva:.2f} °C")
    with col2:
        alerta = "🔴 Riesgo de vaporización (>100°C)" if pred_temp_viva >= 100 else \
                 "🟢 Rango de coagulación/necrosis (50-100°C)" if pred_temp_viva >= 50 else \
                 "🟡 Aún no alcanza umbral de necrosis (<50°C)"
        st.metric("Estado térmico estimado", alerta)

    st.subheader("📊 Curvas de Temperatura vs. Diámetro Vascular (efecto heat-sink)")
    t_cont2 = np.linspace(0, 10, 200)
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    colores_d = {diametros_comsol[0]: "blue", diametros_comsol[1]: "green", diametros_comsol[2]: "red"}
    for d in diametros_comsol:
        sub = df_temp[df_temp["Diametro_mm"] == d]
        ax2.scatter(sub["Tiempo_min"], sub["Temperatura_C"], color=colores_d[d], s=60,
                     label=f"COMSOL histórico (D={d:.0f} mm)", zorder=5)
    Xc2 = np.column_stack((t_cont2, np.full_like(t_cont2, diametro_input)))
    pred_c2 = modelo_temp.predict(Xc2)
    ax2.plot(t_cont2, pred_c2, "m--", linewidth=2.5, label=f"Predicción IA continua (D={diametro_input} mm)")
    ax2.plot(tiempo_input2, pred_temp_viva, "kX", markersize=12, label="Punto en vivo seleccionado")
    ax2.set_title("Temperatura en la pared vascular: datos COMSOL vs. predicción IA")
    ax2.set_xlabel("Tiempo (min)"); ax2.set_ylabel("Temperatura (°C)")
    ax2.grid(True, linestyle=":", alpha=0.6); ax2.legend(loc="lower right")
    st.pyplot(fig2)

    with st.expander("🔎 Hallazgo interesante: el diámetro casi no importa a 22V"):
        st.markdown("""
        Al entrenar el modelo, el kernel del Gaussian Process asignó una longitud de escala
        **extremadamente grande** a la variable diámetro — es decir, el modelo "aprendió" que a
        V₀=22V la temperatura final en la pared vascular es prácticamente la misma para D=1, 3 o
        5 mm (todas convergen a ~103°C). Esto confirma cuantitativamente, con una herramienta de
        IA independiente, la conclusión que el grupo ya había observado con las curvas
        superpuestas: a este voltaje, la generación de calor por efecto Joule domina tan
        fuertemente sobre la disipación convectiva del vaso, que el tamaño del vaso deja de ser
        el factor limitante.
        """)
        st.markdown("""
        **Por qué NO se usó un polinomio aquí:** se probaron polinomios de grado 2 (subajusta la
        meseta en ~5°C), grado 3 (oscila de forma irreal entre puntos: sube a 107°C, baja a
        100°C, vuelve a subir) y grado 4 -- ajuste exacto con 15 parámetros para 15 datos --
        (sobreajuste severo tipo Runge: llega a predecir una *caída* de temperatura con el
        tiempo, físicamente imposible). El Gaussian Process Regression, en cambio, dio un
        RMSE de validación cruzada de ~0.16-0.35°C sin ninguna oscilación.
        """)

# =============================================================================
# PESTAÑA 3 — SENSIBILIDAD A LA POTENCIA (extensión analítica, NO ENTRENADA)
# =============================================================================
with tab3:
    st.warning("""
    **Esta pestaña NO es un modelo de IA entrenado.** Los tres archivos de datos disponibles
    (fracionamiento_de_daño.txt, vasoo.txt, parametros.txt) corresponden **únicamente** a
    V0 = 22 V — no existe en el proyecto un barrido paramétrico de voltaje. Para entrenar un
    modelo real de "IA vs. potencia" haría falta correr COMSOL con más voltajes (p.ej. 15, 22,
    30 V) y aplicar el mismo enfoque de Gaussian Process usado arriba.

    Lo que sigue es una **extensión analítica** (no aprendida de datos) basada directamente en
    las ecuaciones del propio proyecto, mostrada aquí como una idea de análisis adicional, no
    como resultado predictivo validado.
    """)

    st.markdown(r"""
    ### Justificación física
    El modelo del grupo define la generación de calor por efecto Joule como:

    $$Q_{RF} = \sigma |\nabla V|^2$$

    y la ecuación de biocalor de Pennes que resuelve COMSOL es **lineal** en el término fuente
    $Q_{RF}$ (conducción + perfusión, ambos términos lineales en T). Para una geometría y
    conductividad fijas, esto implica que, en estado estacionario (meseta térmica), el
    incremento de temperatura sobre la basal escala aproximadamente con el cuadrado del
    voltaje aplicado:

    $$\Delta T_{meseta}(V_0) \approx \Delta T_{meseta}(22\text{V}) \cdot \left(\frac{V_0}{22}\right)^2$$

    Usando el valor de meseta medido en COMSOL a 22V (~103°C, es decir ΔT≈66°C sobre 37°C
    basal), se puede *estimar* (no simular) la meseta térmica para otros voltajes:
    """)

    V0_ref = 22.0
    T_base = 37.0
    dT_ref = float(df_temp[df_temp["Tiempo_min"] == df_temp["Tiempo_min"].max()]["Temperatura_C"].mean()) - T_base

    V_grid = np.linspace(10, 35, 100)
    T_est = T_base + dT_ref * (V_grid / V0_ref) ** 2

    fig3, ax3 = plt.subplots(figsize=(9, 5))
    ax3.plot(V_grid, T_est, color="darkred", linewidth=2.5,
              label="Estimación teórica ΔT ∝ V² (no simulada)")
    ax3.scatter([V0_ref], [T_base + dT_ref], color="black", zorder=5, s=80,
                 label=f"Único dato real disponible (V₀={V0_ref:.0f}V, COMSOL)")
    ax3.axhline(100, color="gray", linestyle=":", label="Umbral de vaporización (100°C)")
    ax3.set_xlabel("Voltaje aplicado V₀ (V)"); ax3.set_ylabel("Temperatura de meseta estimada (°C)")
    ax3.set_title("Sensibilidad teórica de la meseta térmica al voltaje (Q_RF ∝ V²)")
    ax3.legend(); ax3.grid(alpha=0.3)
    st.pyplot(fig3)

    V_input = st.slider("Voltaje a evaluar (V):", 10.0, 35.0, 22.0, step=0.5)
    T_pred_teorica = T_base + dT_ref * (V_input / V0_ref) ** 2
    st.metric("Meseta térmica estimada (extrapolación teórica)", f"{T_pred_teorica:.1f} °C")
    st.caption("Esta estimación asume que la respuesta térmica del tejido es lineal frente a la "
                "potencia inyectada (válido dentro de las hipótesis del modelo de Pennes con "
                "propiedades constantes). No reemplaza una simulación COMSOL real a otros voltajes.")

st.markdown("""
---
### 🛠️ Diagrama de Flujo del Proceso
`COMSOL Multiphysics (Datos Numéricos)` ➡️ `Lectura directa de los .txt exportados` ➡️
`Entrenamiento de 2 modelos (polinomial para daño, GPR para heat-sink)` ➡️
`Despliegue en Streamlit Web App`
""")
