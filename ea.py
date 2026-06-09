from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd


DATA_FILE = "Diagnóstico de Agentes de IA y Automatización(Sheet1) (1).csv"

METRIC_COLUMNS = [
    "Resuelve el problema para el que fue creado\n",
    "Entrega resultados útiles\n",
    "Reduce retrabajo\n",
    "La calidad es adecuada\n",
    "Los usuarios volverían a usarlo\n",
    "¿Cumple su objetivo?\n",
    "Ahorra tiempo\n",
    "Reduce esfuerzo humano\n",
    "Reduce costo operativo\n",
    "Escala mejor que el proceso anterior\n",
    "Reduce tiempos de respuesta\n",
]

MULTI_VALUE_COLUMNS = {
    "Tipo de agente": "tipo_agente",
    "¿Qué capacidades aumenta?": "capacidades",
    "¿Cuál es su objetivo principal?": "objetivos",
    "Si el agente desaparece mañana\n¿Qué perderíamos?": "perdidas",
    "¿Dónde sigue siendo indispensable un humano?": "humano_indispensable",
    "¿Qué le falta al agente?": "brechas",
}

AUTONOMY_ORDER = [
    "Solo responde",
    "Sugiere",
    "Trabaja y espera aprobación",
    "Ejecuta algunas acciones automáticamente.",
]

AUTONOMY_MAP = {label: index + 1 for index, label in enumerate(AUTONOMY_ORDER)}

TYPE_PATTERNS = [
    "Knowledge Agent",
    "Analyst Agent",
    "Synthesis Agent",
    "Recommendation Agent",
    "Content Agent",
    "Orchestrator Agent",
    "Innovation Agent",
]


def _project_root() -> Path:
    return Path(__file__).resolve().parent


def _normalized_filename(value: str) -> str:
    return unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii").lower()


def get_data_path(data_path: str | Path | None = None) -> Path:
    if data_path is not None:
        return Path(data_path)

    root = _project_root()
    exact_path = root / DATA_FILE
    if exact_path.exists():
        return exact_path

    target_name = _normalized_filename(DATA_FILE)
    csv_files = sorted(root.glob("*.csv"))

    for candidate in csv_files:
        if _normalized_filename(candidate.name) == target_name:
            return candidate

    for candidate in csv_files:
        normalized = _normalized_filename(candidate.name)
        if "diagnostico" in normalized and "agentes" in normalized:
            return candidate

    if len(csv_files) == 1:
        return csv_files[0]

    raise FileNotFoundError(
        "No se encontró el archivo CSV de la encuesta en el directorio del proyecto."
    )


def load_raw_data(data_path: str | Path | None = None) -> pd.DataFrame:
    path = get_data_path(data_path)
    return pd.read_csv(path, encoding="latin1")


def normalize_text(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    text = text.replace("Nerdadore", "NerdAfore")
    text = text.replace("\x96", "-")
    return re.sub(r"\s+", " ", text)


def slugify(value: object) -> str:
    text = normalize_text(value)
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    return text


def split_multivalue(value: object) -> list[str]:
    if pd.isna(value):
        return []
    parts = [normalize_text(part) for part in str(value).split(";")]
    return [part for part in parts if part]


def extract_baseline_hours(value: object) -> float:
    if pd.isna(value):
        return np.nan

    text = normalize_text(value)
    numbers = re.findall(r"\d+\.?\d*", text.replace(",", "."))
    if not numbers:
        return np.nan

    if "de" in text.lower() and len(numbers) >= 2:
        return float(numbers[0])

    return float(numbers[0])


def extract_agent_type_short(value: object) -> list[str]:
    types = split_multivalue(value)
    short_types = []
    for agent_type in types:
        matched = next((pattern for pattern in TYPE_PATTERNS if pattern in agent_type), agent_type)
        short_types.append(matched)
    return short_types


def explode_column(
    df: pd.DataFrame,
    source_col: str,
    item_name: str,
    transform=None,
) -> pd.DataFrame:
    working = df[["Nombre del Agente", "Área responsable", source_col]].copy()
    if transform is None:
        working[item_name] = working[source_col].apply(split_multivalue)
    else:
        working[item_name] = working[source_col].apply(transform)

    working = working.drop(columns=[source_col]).explode(item_name)
    working[item_name] = working[item_name].fillna("").astype(str).str.strip()
    working = working[working[item_name] != ""]
    return working.reset_index(drop=True)


def build_agent_dataset(data_path: str | Path | None = None) -> pd.DataFrame:
    df = load_raw_data(data_path).copy()

    text_columns = df.select_dtypes(include="object").columns
    for column in text_columns:
        df[column] = df[column].apply(normalize_text)

    df["Área responsable"] = df["Área responsable"].replace(
        {
            "Modelos de negocio": "Modelos de Negocio",
        }
    )

    for column in METRIC_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df["Baseline Horas"] = df[
        "¿Cuál es el baseline del agente?(Cuántas horas demoraría la misma tarea en hacerse si no contáramos con el agente)"
    ].apply(extract_baseline_hours)
    df["Nivel de autonomía"] = df["Nivel de autonomía"].replace("", np.nan)
    df["Autonomía Score"] = df["Nivel de autonomía"].map(AUTONOMY_MAP)
    df["Score General"] = df[METRIC_COLUMNS].mean(axis=1)

    df["Tipos de agente list"] = df["Tipo de agente"].apply(extract_agent_type_short)
    df["Capacidades list"] = df["¿Qué capacidades aumenta?"].apply(split_multivalue)
    df["Objetivos list"] = df["¿Cuál es su objetivo principal?"].apply(split_multivalue)
    df["Perdidas list"] = df["Si el agente desaparece mañana\n¿Qué perderíamos?"].apply(split_multivalue)
    df["Humano indispensable list"] = df["¿Dónde sigue siendo indispensable un humano?"].apply(split_multivalue)
    df["Brechas list"] = df["¿Qué le falta al agente?"].apply(split_multivalue)

    df["Cobertura Tipos"] = df["Tipos de agente list"].apply(len)
    df["Cobertura Capacidades"] = df["Capacidades list"].apply(len)
    df["Cobertura Objetivos"] = df["Objetivos list"].apply(len)
    df["Cobertura Brechas"] = df["Brechas list"].apply(len)
    df["Dependencias Humanas"] = df["Humano indispensable list"].apply(len)

    score_norm = (df["Score General"] / 5).fillna(0)
    baseline_norm = (
        np.log1p(df["Baseline Horas"].fillna(df["Baseline Horas"].median()))
        / np.log1p(df["Baseline Horas"].fillna(df["Baseline Horas"].median()).max())
    ).replace([np.inf, -np.inf], 0).fillna(0)
    capabilities_norm = (
        df["Cobertura Capacidades"] / max(df["Cobertura Capacidades"].max(), 1)
    ).fillna(0)

    df["Indice Relevancia"] = (
        100 * (0.5 * score_norm + 0.3 * baseline_norm + 0.2 * capabilities_norm)
    ).round(1)
    df["Estatus Relevancia"] = pd.cut(
        df["Indice Relevancia"],
        bins=[0, 50, 70, 85, 100],
        labels=["Baja", "Media", "Alta", "Crítica"],
        include_lowest=True,
    )
    df["agent_slug"] = df["Nombre del Agente"].apply(slugify)

    return df


def build_counts_table(df: pd.DataFrame, source_col: str, item_name: str, transform=None) -> pd.DataFrame:
    exploded = explode_column(df, source_col, item_name, transform=transform)
    counts = (
        exploded[item_name]
        .value_counts()
        .rename_axis(item_name)
        .reset_index(name="conteo")
    )
    return counts


def build_heatmap_table(df: pd.DataFrame, source_col: str, item_name: str, transform=None) -> pd.DataFrame:
    exploded = explode_column(df, source_col, item_name, transform=transform)
    return pd.crosstab(exploded["Nombre del Agente"], exploded[item_name])


def build_summary(df: pd.DataFrame) -> dict[str, float]:
    total_agents = int(df["Nombre del Agente"].nunique())
    total_areas = int(df["Área responsable"].nunique())
    avg_score = float(df["Score General"].mean()) if not df.empty else 0.0
    avg_relevance = float(df["Indice Relevancia"].mean()) if not df.empty else 0.0
    baseline_total = float(df["Baseline Horas"].fillna(0).sum()) if not df.empty else 0.0
    return {
        "total_agents": total_agents,
        "total_areas": total_areas,
        "avg_score": avg_score,
        "avg_relevance": avg_relevance,
        "baseline_total": baseline_total,
    }
