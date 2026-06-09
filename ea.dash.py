from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from ea import (
    AUTONOMY_ORDER,
    METRIC_COLUMNS,
    build_agent_dataset,
    build_counts_table,
    build_heatmap_table,
    build_summary,
    extract_agent_type_short,
)


st.set_page_config(
    page_title="Diagnóstico de Agentes IA",
    page_icon=":bar_chart:",
    layout="wide",
)


@st.cache_data
def load_dashboard_data() -> pd.DataFrame:
    return build_agent_dataset()


def color_scale_chart(df: pd.DataFrame, x: str, y: str, color: str, title: str, horizontal: bool = False):
    if horizontal:
        fig = px.bar(
            df,
            y=y,
            x=x,
            color=color,
            orientation="h",
            color_continuous_scale="Tealgrn",
            title=title,
            text_auto=".1f",
        )
    else:
        fig = px.bar(
            df,
            x=x,
            y=y,
            color=color,
            color_continuous_scale="Tealgrn",
            title=title,
            text_auto=".1f",
        )
    fig.update_layout(height=420, coloraxis_showscale=False)
    return fig


df = load_dashboard_data()

st.title("Tablero de diagnóstico de agentes de IA")
st.caption(
    "Vista ejecutiva para analizar desempeño, relevancia operativa y brechas de los agentes usados en distintas áreas de negocio."
)

with st.sidebar:
    st.header("Filtros")
    areas = st.multiselect(
        "Área responsable",
        options=sorted(df["Área responsable"].dropna().unique().tolist()),
        default=sorted(df["Área responsable"].dropna().unique().tolist()),
    )
    autonomy = st.multiselect(
        "Nivel de autonomía",
        options=[item for item in AUTONOMY_ORDER if item in df["Nivel de autonomía"].unique()],
        default=[item for item in AUTONOMY_ORDER if item in df["Nivel de autonomía"].unique()],
    )
    score_min = st.slider("Score general mínimo", min_value=1.0, max_value=5.0, value=1.0, step=0.1)
    relevance_cut = st.slider("Índice de relevancia mínimo", min_value=0, max_value=100, value=0, step=5)

filtered_df = df[
    df["Área responsable"].isin(areas)
    & df["Nivel de autonomía"].isin(autonomy)
    & (df["Score General"] >= score_min)
    & (df["Indice Relevancia"] >= relevance_cut)
].copy()

if filtered_df.empty:
    st.warning("No hay registros con los filtros seleccionados.")
    st.stop()

summary = build_summary(filtered_df)

st.markdown("### KPIs")
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Agentes evaluados", summary["total_agents"])
col2.metric("Áreas cubiertas", summary["total_areas"])
col3.metric("Score promedio", f"{summary['avg_score']:.2f}/5")
col4.metric("Relevancia promedio", f"{summary['avg_relevance']:.1f}/100")
col5.metric("Horas baseline", f"{summary['baseline_total']:.1f} h")

top_agent = filtered_df.sort_values("Indice Relevancia", ascending=False).iloc[0]
gap_agent = filtered_df.sort_values("Cobertura Brechas", ascending=False).iloc[0]

highlight_col1, highlight_col2 = st.columns(2)
with highlight_col1:
    st.markdown(
        f"""
        <div style="background:#e8f7f2;border:1px solid #b7e4d4;border-radius:16px;padding:18px 20px;">
            <div style="font-size:0.85rem;color:#2c6e5a;font-weight:600;text-transform:uppercase;letter-spacing:0.04em;">
                Agente más relevante
            </div>
            <div style="font-size:1.45rem;font-weight:700;color:#123b32;margin-top:6px;">
                {top_agent['Nombre del Agente']}
            </div>
            <div style="font-size:1rem;color:#245447;margin-top:6px;">
                Índice de relevancia: <strong>{top_agent['Indice Relevancia']}</strong>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with highlight_col2:
    st.markdown(
        f"""
        <div style="background:#fff3e8;border:1px solid #f2c79b;border-radius:16px;padding:18px 20px;">
            <div style="font-size:0.85rem;color:#9a5b12;font-weight:600;text-transform:uppercase;letter-spacing:0.04em;">
                Mayor necesidad de mejora
            </div>
            <div style="font-size:1.45rem;font-weight:700;color:#6f3f08;margin-top:6px;">
                {gap_agent['Nombre del Agente']}
            </div>
            <div style="font-size:1rem;color:#7a4a12;margin-top:6px;">
                Brechas declaradas: <strong>{int(gap_agent['Cobertura Brechas'])}</strong>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Resumen ejecutivo", "Capacidades y objetivos", "Desempeño", "Brechas y riesgo", "Datos base"]
)

with tab1:
    st.subheader("Mapa de relevancia de agentes")
    fig_scatter = px.scatter(
        filtered_df,
        x="Baseline Horas",
        y="Score General",
        size="Cobertura Capacidades",
        color="Indice Relevancia",
        hover_name="Nombre del Agente",
        hover_data={
            "Área responsable": True,
            "Nivel de autonomía": True,
            "Indice Relevancia": True,
            "Baseline Horas": ":.1f",
            "Score General": ":.2f",
            "Cobertura Capacidades": True,
        },
        color_continuous_scale="Tealgrn",
        title="Importancia inferida por impacto operativo y evaluación",
    )
    fig_scatter.update_layout(height=500)
    st.plotly_chart(fig_scatter, use_container_width=True)

    ranking_df = filtered_df.sort_values("Indice Relevancia", ascending=True)[
        ["Nombre del Agente", "Indice Relevancia", "Score General"]
    ]
    st.plotly_chart(
        color_scale_chart(
            ranking_df,
            x="Indice Relevancia",
            y="Nombre del Agente",
            color="Indice Relevancia",
            title="Ranking de relevancia",
            horizontal=True,
        ),
        use_container_width=True,
    )

    area_view = (
        filtered_df.groupby("Área responsable", as_index=False)
        .agg(
            agentes=("Nombre del Agente", "nunique"),
            relevancia_promedio=("Indice Relevancia", "mean"),
            score_promedio=("Score General", "mean"),
        )
        .sort_values("relevancia_promedio", ascending=False)
    )
    st.plotly_chart(
        px.bar(
            area_view,
            x="Área responsable",
            y="relevancia_promedio",
            color="agentes",
            text_auto=".1f",
            color_continuous_scale="Mint",
            title="Relevancia promedio por área",
        ).update_layout(height=420),
        use_container_width=True,
    )

with tab2:
    st.subheader("Capacidades más potenciadas")
    capacities = build_counts_table(filtered_df, "¿Qué capacidades aumenta?", "Capacidad")
    st.plotly_chart(
        px.bar(
            capacities,
            x="conteo",
            y="Capacidad",
            orientation="h",
            color="conteo",
            color_continuous_scale="Blues",
            text_auto=True,
            title="Capacidades que más fortalecen los agentes",
        ).update_layout(height=450, coloraxis_showscale=False),
        use_container_width=True,
    )

    objectives = build_counts_table(filtered_df, "¿Cuál es su objetivo principal?", "Objetivo")
    st.plotly_chart(
        px.bar(
            objectives,
            x="conteo",
            y="Objetivo",
            orientation="h",
            color="conteo",
            color_continuous_scale="Greens",
            text_auto=True,
            title="Objetivos principales declarados",
        ).update_layout(height=420, coloraxis_showscale=False),
        use_container_width=True,
    )

    type_matrix = build_heatmap_table(
        filtered_df,
        "Tipo de agente",
        "Tipo",
        transform=extract_agent_type_short,
    )
    fig_type_heatmap = px.imshow(
        type_matrix,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="Blues",
        title="Mapa de calor de tipos de agente por solución",
    )
    fig_type_heatmap.update_layout(
        height=420,
        xaxis_title="Tipo de agente",
        yaxis_title="Nombre del agente",
        coloraxis_showscale=False,
    )
    st.plotly_chart(fig_type_heatmap, use_container_width=True)

with tab3:
    st.subheader("Desempeño por agente")
    metric_scores = (
        filtered_df.groupby("Nombre del Agente")[METRIC_COLUMNS]
        .mean()
        .T.reset_index()
        .rename(columns={"index": "Métrica"})
    )
    metric_scores["Métrica"] = metric_scores["Métrica"].str.replace("\n", "", regex=False)
    heatmap_df = metric_scores.set_index("Métrica")
    st.dataframe(heatmap_df.style.format("{:.2f}"), use_container_width=True)

    agent_eval = (
        filtered_df[["Nombre del Agente", "Score General", "Ahorra tiempo\n", "Reduce costo operativo\n"]]
        .sort_values("Score General", ascending=False)
    )
    fig_eval = px.bar(
        agent_eval,
        x="Nombre del Agente",
        y=["Score General", "Ahorra tiempo\n", "Reduce costo operativo\n"],
        barmode="group",
        title="Comparativo de desempeño",
    )
    fig_eval.update_layout(height=450, yaxis_title="Puntaje promedio")
    st.plotly_chart(fig_eval, use_container_width=True)

    autonomy_view = (
        filtered_df.groupby("Nivel de autonomía", as_index=False)
        .agg(
            agentes=("Nombre del Agente", "nunique"),
            score_promedio=("Score General", "mean"),
        )
    )
    autonomy_view["Nivel de autonomía"] = pd.Categorical(
        autonomy_view["Nivel de autonomía"], categories=AUTONOMY_ORDER, ordered=True
    )
    autonomy_view = autonomy_view.sort_values("Nivel de autonomía")
    st.plotly_chart(
        px.line(
            autonomy_view,
            x="Nivel de autonomía",
            y="score_promedio",
            markers=True,
            title="Desempeño promedio por nivel de autonomía",
        ).update_layout(height=380),
        use_container_width=True,
    )

with tab4:
    st.subheader("Brechas, pérdidas y dependencia humana")
    gaps = build_counts_table(filtered_df, "¿Qué le falta al agente?", "Brecha")
    losses = build_counts_table(
        filtered_df,
        "Si el agente desaparece mañana\n¿Qué perderíamos?",
        "Pérdida",
    )
    human = build_counts_table(
        filtered_df,
        "¿Dónde sigue siendo indispensable un humano?",
        "Intervención humana",
    )
    gaps_by_agent = (
        filtered_df[["Nombre del Agente", "Cobertura Brechas", "¿Qué le falta al agente?"]]
        .rename(columns={"¿Qué le falta al agente?": "Brechas"})
        .sort_values("Cobertura Brechas", ascending=False)
    )

    fig_gaps_by_agent = px.bar(
        gaps_by_agent,
        x="Cobertura Brechas",
        y="Nombre del Agente",
        orientation="h",
        color="Cobertura Brechas",
        color_continuous_scale="Sunset",
        text_auto=True,
        hover_data={"Brechas": True, "Cobertura Brechas": True},
        title="Número de brechas por agente",
    )
    fig_gaps_by_agent.update_traces(
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Número de brechas: %{x}<br><br>"
            "Brechas detectadas:<br>%{customdata[0]}<extra></extra>"
        )
    )
    fig_gaps_by_agent.update_layout(height=420, coloraxis_showscale=False)
    st.plotly_chart(fig_gaps_by_agent, use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.plotly_chart(
            px.bar(
                gaps,
                x="conteo",
                y="Brecha",
                orientation="h",
                color="conteo",
                color_continuous_scale="Oranges",
                text_auto=True,
                title="Brechas más repetidas",
            ).update_layout(height=450, coloraxis_showscale=False),
            use_container_width=True,
        )
    with col_b:
        st.plotly_chart(
            px.bar(
                losses,
                x="conteo",
                y="Pérdida",
                orientation="h",
                color="conteo",
                color_continuous_scale="Reds",
                text_auto=True,
                title="Qué se perdería si el agente desaparece",
            ).update_layout(height=450, coloraxis_showscale=False),
            use_container_width=True,
        )

    st.plotly_chart(
        px.bar(
            human,
            x="conteo",
            y="Intervención humana",
            orientation="h",
            color="conteo",
            color_continuous_scale="Purples",
            text_auto=True,
            title="Dónde sigue siendo clave el humano",
        ).update_layout(height=420, coloraxis_showscale=False),
        use_container_width=True,
    )

    risk_table = filtered_df[
        [
            "Nombre del Agente",
            "Área responsable",
            "Indice Relevancia",
            "Dependencias Humanas",
            "Cobertura Brechas",
            "¿Qué le falta al agente?",
        ]
    ].sort_values(["Indice Relevancia", "Cobertura Brechas"], ascending=[False, False])
    st.dataframe(risk_table, use_container_width=True)

with tab5:
    st.subheader("Detalle de la data procesada")
    display_columns = [
        "Nombre del Agente",
        "Área responsable",
        "Nivel de autonomía",
        "Baseline Horas",
        "Score General",
        "Indice Relevancia",
        "¿Cuál es su objetivo principal?",
        "¿Qué capacidades aumenta?",
        "¿Qué le falta al agente?",
        "¿Dónde sigue siendo indispensable un humano?",
    ]
    st.dataframe(
        filtered_df[display_columns].sort_values("Indice Relevancia", ascending=False),
        use_container_width=True,
    )
