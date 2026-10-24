import os
import json
import time
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Optional, Dict, Any
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

# Rutas de archivos
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_FILE = os.path.join(BASE_DIR, "Temperaturas", "Global Land Temperatures", "GlobalLandTemperaturesByMajorCity.csv")
GLOBAL_CSV_FILE = os.path.join(BASE_DIR, "Temperaturas", "Global Land Temperatures", "GlobalTemperatures.csv")
CACHE_FILE = os.path.join(BASE_DIR, "open_meteo_cache.json")
STATIC_DIR = os.path.join(BASE_DIR, "static")

CURRENT_YEAR = datetime.now().year  # Año actual dinámico (ej. 2026)
HORIZONTE_PROYECCION = 2050         # Cobertura de proyección extendida hasta 2050
BASE_COMFORT_CDD = 18.0             # Estándar ASHRAE (18°C)

# ==============================================================================
# ESTRUCTURA GLOBAL EN MEMORIA RAM (PRE-COMPUTADA EN LIFESPAN)
# ==============================================================================
GLOBAL_CACHE: Dict[str, Any] = {
    "series_temporales": {},          # Series históricas y proyectadas (1850-2050) de anomalías, CDD y timeline anual
    "ciudades": {},                   # { "Ciudad": { "coords": [lat, lon], "cdd_historico": [...], "metricas": {...}, ... } }
    "ciudades_lower": {},             # Mapa O(1) case-insensitive -> clave canónica
    "matriz_interpolacion_base": {},  # Matriz base pre-computada para el mapa rasterizado de calor
    "config": {},                     # Configuración general y catálogo de ciudades
    "index_html": "",                 # HTML principal en RAM
    "bytes": {                        # JSON pre-serializados en bytes UTF-8 para latencia < 1 ms
        "climate_data": b"{}",
        "config": b"{}",
        "years": {},
        "cities": {}
    }
}
METADATA_CIUDADES: Dict[str, Dict[str, Any]] = {}
ANOS_HISTORICOS: list = []
PROYECCION_ANOS: list = list(range(CURRENT_YEAR + 1, HORIZONTE_PROYECCION + 1))
GLOBAL_HISTORIAL: Dict[int, float] = {}
GLOBAL_MODEL: Dict[str, Any] = {}


def parse_coord(coord_str: str) -> float:
    """Convierte formato '19.25N' o '99.10W' a float positivo/negativo."""
    if not isinstance(coord_str, str):
        return 0.0
    coord_str = coord_str.strip()
    direction = coord_str[-1].upper()
    val = float(coord_str[:-1])
    return -val if direction in ['S', 'W'] else val


def calcular_cdd(temperatura: float) -> float:
    """Calcula Cooling Degree Days (ASHRAE base 18°C)."""
    return round(max(0.0, float(temperatura) - BASE_COMFORT_CDD), 2)


def calcular_peso_normalizado(temp: float) -> float:
    """Normaliza la temperatura al rango [0.1, 1.0] para la capa térmica."""
    if temp < 10.0:
        return round(max(0.1, min(0.35, 0.1 + (temp + 5.0) / 15.0 * 0.25)), 2)
    elif temp <= 26.0:
        return round(0.36 + (temp - 10.0) / 16.0 * 0.32, 2)
    else:
        return round(min(1.0, 0.70 + (temp - 26.0) / 14.0 * 0.30), 2)


def clasificar_riesgo_energetico(calentamiento_decada: float, pct_aumento_cdd: float) -> dict:
    """
    Segmentación de vulnerabilidad mediante semáforo universal:
    - ALTO RIESGO (Rojo): > +0.25°C/década o > +25% en proyección de CDD.
    - RIESGO MODERADO (Amarillo): +10% a +25% en CDD.
    - BAJO RIESGO / ESTABLE (Verde): < +10% en CDD.
    """
    if calentamiento_decada > 0.25 or pct_aumento_cdd > 25.0:
        return {
            "nivel": "ALTO",
            "etiqueta": "ALTO RIESGO CRÍTICO",
            "color": "red",
            "hex": "#ef4444",
            "estres_red": "Crítico (>85% saturación en picos térmicos)",
            "descripcion": "Riesgo inminente de sobrecarga en transformadores y contingencias por demanda simultánea de climatización."
        }
    elif pct_aumento_cdd >= 10.0 or calentamiento_decada > 0.15:
        return {
            "nivel": "MODERADO",
            "etiqueta": "RIESGO MODERADO",
            "color": "yellow",
            "hex": "#f59e0b",
            "estres_red": "Moderado-Alto (60%-85% capacidad nominal)",
            "descripcion": "Incremento sostenido que requiere refuerzo preventivo en subestaciones y auditorías de eficiencia."
        }
    else:
        return {
            "nivel": "BAJO",
            "etiqueta": "ESTABLE / BAJO RIESGO",
            "color": "green",
            "hex": "#10b981",
            "estres_red": "Estable (<60% capacidad)",
            "descripcion": "Variación térmica dentro de los márgenes de resiliencia del sistema de distribución."
        }


def clasificar_perfil_geoclimatico(city: str, country: str, lat: float, lon: float, temp_media: float) -> str:
    """
    Clasifica la urbe en uno de tres perfiles geoclimáticos:
    - DESERTICO: Alta radiación solar, aridez extrema o latitudes secas.
    - COSTERO_TROPICAL: Zonas costeras/tropicales de alta humedad o islas.
    - TEMPLADO_BOREAL: Europa Central/Norte, Norteamérica septentrional, zonas templadas frías.
    """
    city_l = city.lower()
    country_l = country.lower()
    lat_abs = abs(lat)

    desert_keywords = [
        "cairo", "baghdad", "riyadh", "dubai", "alexandria", "damascus", "amman", "jeddah",
        "kuwait", "doha", "casablanca", "tunis", "algiers", "hermosillo", "mexicali", "monterrey",
        "phoenix", "las vegas", "karachi", "lahore", "delhi", "jaipur", "ahmadabad"
    ]
    desert_countries = ["egypt", "iraq", "saudi arabia", "iran", "morocco", "algeria", "tunisia", "united arab emirates", "kuwait", "qatar"]

    if any(k in city_l for k in desert_keywords) or any(c in country_l for c in desert_countries):
        return "DESERTICO"

    if lat_abs >= 40.0:
        return "TEMPLADO_BOREAL"

    cold_transition_countries = ["germany", "united kingdom", "france", "russia", "poland", "canada", "ukraine", "japan", "sweden", "norway"]
    if any(c in country_l for c in cold_transition_countries):
        return "TEMPLADO_BOREAL"

    if lat_abs <= 28.0 or temp_media >= 22.0:
        return "COSTERO_TROPICAL"

    return "TEMPLADO_BOREAL"


def generar_medidas_prescriptivas(riesgo_info: dict, delta_temp: float, pct_cdd: float, perfil_clima: str = "COSTERO_TROPICAL", city_name: str = "") -> list:
    """
    Genera el catálogo de 4 medidas prescriptivas diferenciadas y contextuales
    según el perfil geoclimático, el semáforo de riesgo y los valores dinámicos calculados.
    """
    if perfil_clima == "DESERTICO":
        return [
            {
                "categoria": "1. Infraestructura Eléctrica",
                "titulo": f"Mitigación de Sobrecarga Térmica en Transformadores ({city_name})",
                "descripcion": f"Instalación de blindajes contra radiación solar directa y refrigeración forzada (aceite/aire ONAF) en subestaciones críticas para soportar el incremento proyectado de +{pct_cdd}% en demanda simultánea de aire acondicionado continuo.",
                "icono": "⚡"
            },
            {
                "categoria": "2. Normativa y Construcción Urbana",
                "titulo": "Pavimentos y Techos Reflectivos contra Isla de Calor",
                "descripcion": f"Modificación al reglamento urbano para exigir albedo superficial mínimo de 0.70 en techumbres (Cool Roofs) y pavimentos permeables/reflectivos para amortiguar el aumento térmico estimado de +{delta_temp}°C.",
                "icono": "🏜️"
            },
            {
                "categoria": "3. Eficiencia Energética",
                "titulo": "Sustitución HVAC Inverter para Climas Extremos",
                "descripcion": "Incentivos fiscales para migrar a equipos con compresores de velocidad variable certificados para operar hasta 52°C sin caída de eficiencia, gestionando la curva de demanda en horas pico de insolación.",
                "icono": "❄️"
            },
            {
                "categoria": "4. Alerta Temprana & Resiliencia",
                "titulo": "Respuesta Dinámica a la Demanda & Microredes Solares",
                "descripcion": "Acoplamiento de generación solar fotovoltaica con almacenamiento BESS en nodos industriales/residenciales para sincronizar el pico de generación solar con el pico de demanda de enfriamiento.",
                "icono": "☀️"
            }
        ]
    elif perfil_clima == "COSTERO_TROPICAL":
        return [
            {
                "categoria": "1. Infraestructura Eléctrica",
                "titulo": "Protección Anticorrosiva & Resiliencia de Subestaciones Costeras",
                "descripcion": f"Encapsulamiento blindado (tecnología GIS) y recubrimientos anti-salinos en líneas y transformadores para evitar fallas dieléctricas ante la combinación de alta humedad costera y sobrecarga de +{pct_cdd}% en climatización.",
                "icono": "⚡"
            },
            {
                "categoria": "2. Normativa y Construcción Urbana",
                "titulo": "Envolventes con Control de Humedad & Ventilación Pasiva",
                "descripcion": f"Requisitos obligatorios de barreras de vapor continuas, aislamiento térmico ASHRAE y diseño de ventilación cruzada para mitigar la carga latente provocada por el alza térmica de +{delta_temp}°C.",
                "icono": "🌊"
            },
            {
                "categoria": "3. Eficiencia Energética",
                "titulo": "Climatización con Deshumidificación Dedicada (DOAS)",
                "descripcion": "Adopción prioritaria de sistemas de aire exterior dedicados con recuperación entálpica para desacoplar el control de humedad del enfriamiento sensible, reduciendo hasta un 30% el consumo eléctrico.",
                "icono": "💧"
            },
            {
                "categoria": "4. Alerta Temprana & Resiliencia",
                "titulo": "Microredes Híbridas para Eventos Meteorológicos Extremos",
                "descripcion": "Protocolo de respaldo de energía descentralizada y desconexión controlada en zonas vulnerables durante tormentas tropicales u olas de calor simultáneas.",
                "icono": "🚨"
            }
        ]
    else:  # TEMPLADO_BOREAL
        return [
            {
                "categoria": "1. Infraestructura Eléctrica",
                "titulo": "Reconversión y Refuerzo de Redes para Picos Estivales",
                "descripcion": f"Modernización de transformadores envejecidos —diseñados históricamente sólo para calefacción invernal— para soportar la nueva demanda masiva estival de climatización proyectada en +{pct_cdd}%.",
                "icono": "⚡"
            },
            {
                "categoria": "2. Normativa y Construcción Urbana",
                "titulo": "Transición de Calefacción a Redes de Distrito de Frío (District Cooling)",
                "descripcion": f"Despliegue de canalizaciones de refrigeración centralizada por agua de lago/río y actualización de códigos de edificación ante el calentamiento estival sostenido de +{delta_temp}°C.",
                "icono": "🏢"
            },
            {
                "categoria": "3. Eficiencia Energética",
                "titulo": "Bombas de Calor Reversibles de Alta Eficiencia",
                "descripcion": "Reemplazo integral de calderas de gas por bombas de calor aerotérmicas y geotérmicas reversibles (SCOP/SEER > 4.5) con control horario automatizado.",
                "icono": "🔄"
            },
            {
                "categoria": "4. Alerta Temprana & Resiliencia",
                "titulo": "Protocolo de Emergencia para Ciudades en Transición Climática",
                "descripcion": "Monitoreo preventivo y habilitación de refugios urbanos de enfriamiento pasivo ante episodios inéditos de olas de calor en poblaciones vulnerables no aclimatadas.",
                "icono": "🛡️"
            }
        ]


def cargar_cache_meteo() -> dict:
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def calcular_baseline_anomalia(historial: dict) -> float:
    """Línea base climatológica 1951-1980 (fallback: media completa) para anomalías térmicas."""
    vals = [v for y, v in historial.items() if 1951 <= y <= 1980]
    if not vals:
        vals = list(historial.values())
    return round(float(np.mean(vals)), 2) if vals else 0.0


def construir_payload_ciudad(city_label: str, meta: dict, historial_dict: dict) -> dict:
    """Construye y estructura el payload completo de una ciudad o del consolidado Global para GLOBAL_CACHE en RAM."""
    base = calcular_baseline_anomalia(historial_dict)
    serie_historica = [
        {
            "year": y,
            "temperatura": historial_dict[y],
            "anomalia": round(historial_dict[y] - base, 2),
            "cdd": calcular_cdd(historial_dict[y])
        }
        for y in sorted(historial_dict.keys())
        if y <= CURRENT_YEAR
    ]

    slope = meta["slope"]
    intercept = meta["intercept"]
    serie_proyeccion = []
    for y in PROYECCION_ANOS:
        pred_val = round(intercept + slope * y, 2)
        diff_years = y - CURRENT_YEAR
        serie_proyeccion.append({
            "year": y,
            "prediccion": pred_val,
            "anomalia_proy": round(pred_val - base, 2),
            "cdd_proy": calcular_cdd(pred_val),
            "lim_sup": round(pred_val + (0.04 * diff_years), 2),
            "lim_inf": round(pred_val - (0.04 * diff_years), 2)
        })

    perfil_clima = clasificar_perfil_geoclimatico(
        city_label,
        meta["country"],
        meta["lat"],
        meta["lon"],
        meta["media_hist"]
    )
    medidas = generar_medidas_prescriptivas(
        meta["riesgo"],
        meta["delta_temp_2040"],
        meta["pct_aumento_cdd"],
        perfil_clima,
        city_label if city_label else "Global"
    )

    metricas = {
        "calentamiento_decada": meta["calentamiento_decada"],
        "media_historica": meta["media_hist"],
        "baseline_anomalia": base,
        "cdd_hist_base": meta["cdd_hist_base"],
        "temp_actual": meta["temp_actual"],
        "temp_proy_2040": meta["temp_proy_2040"],
        "delta_temp_2040": meta["delta_temp_2040"],
        "cdd_proy_2040": meta["cdd_proy_2040"],
        "pct_aumento_cdd": meta["pct_aumento_cdd"],
        "riesgo": meta["riesgo"]
    }

    return {
        "city": city_label,
        "country": meta["country"],
        "coords": [meta["lat"], meta["lon"]],
        "lat": meta["lat"],
        "lon": meta["lon"],
        "perfil_clima": perfil_clima,
        "current_year": CURRENT_YEAR,
        "horizonte": HORIZONTE_PROYECCION,
        "cdd_historico": [item["cdd"] for item in serie_historica],
        "metricas": metricas,
        "calentamiento_decada": meta["calentamiento_decada"],
        "media_historica": meta["media_hist"],
        "cdd_hist_base": meta["cdd_hist_base"],
        "temp_actual": meta["temp_actual"],
        "temp_proy_2040": meta["temp_proy_2040"],
        "delta_temp_2040": meta["delta_temp_2040"],
        "cdd_proy_2040": meta["cdd_proy_2040"],
        "pct_aumento_cdd": meta["pct_aumento_cdd"],
        "riesgo": meta["riesgo"],
        "medidas_prescriptivas": medidas,
        "historico": serie_historica,
        "proyeccion": serie_proyeccion
    }


def _precomputar_matriz_interpolacion_base(grid_h: int = 90, grid_w: int = 180) -> dict:
    """Pre-computa en RAM la grilla geoclimática base para el mapa rasterizado de calor (IDW)."""
    lats = np.linspace(85.0, -85.0, grid_h, dtype=np.float32)
    lons = np.linspace(-180.0, 180.0, grid_w, dtype=np.float32)
    lat_rad = np.deg2rad(lats)
    temp_base_lat = np.round(-16.0 + 46.0 * np.power(np.cos(lat_rad), 1.35), 2)
    return {
        "grid_shape": [grid_h, grid_w],
        "lat_bounds": [85.0, -85.0],
        "lon_bounds": [-180.0, 180.0],
        "lats": np.round(lats, 2).tolist(),
        "lons": np.round(lons, 2).tolist(),
        "temp_base_lat": temp_base_lat.tolist(),
        "idw_params": {"power": 2.0, "radius_sq": 1050.0, "smoothing": 8.0, "blend": 0.90}
    }


def inicializar_dataset_en_memoria():
    """
    Carga única en memoria RAM al arranque del servidor (lifespan):
    1. Lee el CSV de ciudades una sola vez con extracción vectorizada de año.
    2. Pre-computa en GLOBAL_CACHE las series 1850-2050, el índice por ciudad y la matriz base de interpolación.
    3. Pre-serializa en bytes UTF-8 todas las respuestas para garantizar latencia < 1 ms (cero I/O en endpoints).
    """
    global METADATA_CIUDADES, ANOS_HISTORICOS, GLOBAL_HISTORIAL, GLOBAL_MODEL, PROYECCION_ANOS, GLOBAL_CACHE

    t0 = time.perf_counter()
    print("[LIFESPAN] Cargando y pre-computando datasets climáticos en GLOBAL_CACHE (RAM)...")

    if not os.path.exists(CSV_FILE):
        raise FileNotFoundError(f"No se encontró el dataset en: {CSV_FILE}")

    open_meteo_cache = cargar_cache_meteo()

    # Lectura única del dataset desde disco
    df = pd.read_csv(
        CSV_FILE,
        usecols=['dt', 'AverageTemperature', 'City', 'Country', 'Latitude', 'Longitude']
    )
    df = df.dropna(subset=['AverageTemperature', 'dt'])
    df['year'] = df['dt'].str.slice(0, 4).astype(int)
    df = df[(df['year'] >= 1850) & (df['year'] <= 2013)]

    df_annual = (
        df.groupby(['City', 'Country', 'Latitude', 'Longitude', 'year'], as_index=False)['AverageTemperature']
        .mean()
    )
    df_annual['AverageTemperature'] = df_annual['AverageTemperature'].round(2)

    ciudades_coords = df[['City', 'Country', 'Latitude', 'Longitude']].drop_duplicates('City').copy()
    ciudades_coords['lat'] = ciudades_coords['Latitude'].apply(parse_coord)
    ciudades_coords['lon'] = ciudades_coords['Longitude'].apply(parse_coord)
    coords_dict = ciudades_coords.set_index('City')[['Country', 'lat', 'lon']].to_dict('index')

    meta_ciudades = {}
    todos_los_anos_reales = list(range(1850, CURRENT_YEAR + 1))

    for city, group in df_annual.groupby('City', sort=False):
        coord_info = coords_dict.get(city, {'Country': 'Unknown', 'lat': 0.0, 'lon': 0.0})
        lat = float(coord_info['lat'])
        lon = float(coord_info['lon'])
        group_sorted = group.sort_values('year')
        historial = dict(zip(group_sorted['year'].astype(int), group_sorted['AverageTemperature'].astype(float)))

        cache_key = f"{city}_{round(lat, 2)}_{round(lon, 2)}"
        if cache_key in open_meteo_cache:
            historial.update({int(k): float(v) for k, v in open_meteo_cache[cache_key].items()})

        x_prev = np.array([y for y in historial.keys() if y >= 1950], dtype=float)
        y_prev = np.array([historial[int(y)] for y in x_prev], dtype=float)
        if len(x_prev) >= 5:
            fit_prev = np.polyfit(x_prev, y_prev, deg=1)
            for y_fill in range(2014, CURRENT_YEAR + 1):
                if y_fill not in historial:
                    historial[y_fill] = round(float(fit_prev[1] + fit_prev[0] * y_fill), 2)

        x_all = np.array(sorted(historial.keys()), dtype=float)
        y_all = np.array([historial[int(y)] for y in x_all], dtype=float)
        mask_recent = x_all >= 1950
        coeffs = np.polyfit(x_all[mask_recent], y_all[mask_recent], deg=1)
        slope, intercept = float(coeffs[0]), float(coeffs[1])
        calentamiento_decada = round(slope * 10, 3)

        temps_hist = [historial[int(y)] for y in x_all if y <= CURRENT_YEAR]
        media_hist = round(float(np.mean(temps_hist)), 2) if temps_hist else 0.0
        cdd_hist_base = round(float(np.mean([calcular_cdd(t) for t in temps_hist])), 2)

        temp_proy_2040 = round(float(intercept + slope * HORIZONTE_PROYECCION), 2)
        temp_actual = historial.get(CURRENT_YEAR, round(float(intercept + slope * CURRENT_YEAR), 2))
        delta_temp_2040 = round(temp_proy_2040 - temp_actual, 2)
        cdd_proy_2040 = calcular_cdd(temp_proy_2040)

        if cdd_hist_base > 0.01:
            pct_aumento_cdd = round(((cdd_proy_2040 - cdd_hist_base) / cdd_hist_base) * 100.0, 1)
        else:
            pct_aumento_cdd = round((cdd_proy_2040 * 10.0), 1)

        riesgo_info = clasificar_riesgo_energetico(calentamiento_decada, pct_aumento_cdd)

        meta_ciudades[city] = {
            "country": coord_info['Country'],
            "lat": lat,
            "lon": lon,
            "historial": historial,
            "slope": slope,
            "calentamiento_decada": calentamiento_decada,
            "intercept": intercept,
            "media_hist": media_hist,
            "baseline_anom": calcular_baseline_anomalia(historial),
            "cdd_hist_base": cdd_hist_base,
            "temp_actual": temp_actual,
            "temp_proy_2040": temp_proy_2040,
            "delta_temp_2040": delta_temp_2040,
            "cdd_proy_2040": cdd_proy_2040,
            "pct_aumento_cdd": pct_aumento_cdd,
            "riesgo": riesgo_info,
            "riesgo_compacto": {
                "nivel": riesgo_info["nivel"],
                "etiqueta": riesgo_info["etiqueta"],
                "hex": riesgo_info["hex"]
            }
        }

    METADATA_CIUDADES = meta_ciudades
    ANOS_HISTORICOS = todos_los_anos_reales
    PROYECCION_ANOS = list(range(CURRENT_YEAR + 1, HORIZONTE_PROYECCION + 1))

    # Consolidado global
    global_historial = {}
    for y in ANOS_HISTORICOS:
        vals = [data["historial"].get(y) for data in METADATA_CIUDADES.values() if data["historial"].get(y) is not None]
        if vals:
            global_historial[y] = round(float(np.mean(vals)), 2)

    GLOBAL_HISTORIAL = global_historial

    gx = np.array([y for y in GLOBAL_HISTORIAL.keys() if y >= 1950], dtype=float)
    gy = np.array([GLOBAL_HISTORIAL[int(y)] for y in gx], dtype=float)
    gcoeffs = np.polyfit(gx, gy, deg=1)
    g_slope = float(gcoeffs[0])
    g_intercept = float(gcoeffs[1])
    g_cal_dec = round(g_slope * 10, 3)
    g_temp_actual = GLOBAL_HISTORIAL.get(CURRENT_YEAR, round(g_intercept + g_slope * CURRENT_YEAR, 2))
    g_temp_2040 = round(g_intercept + g_slope * HORIZONTE_PROYECCION, 2)
    g_cdd_base = round(float(np.mean([calcular_cdd(t) for t in GLOBAL_HISTORIAL.values()])), 2)
    g_cdd_2040 = calcular_cdd(g_temp_2040)
    g_pct_cdd = round(((g_cdd_2040 - g_cdd_base) / g_cdd_base) * 100.0, 1) if g_cdd_base > 0.01 else 15.0

    GLOBAL_MODEL = {
        "country": "Global",
        "lat": 20.0,
        "lon": 0.0,
        "slope": g_slope,
        "intercept": g_intercept,
        "calentamiento_decada": g_cal_dec,
        "media_hist": round(float(np.mean(list(GLOBAL_HISTORIAL.values()))), 2),
        "cdd_hist_base": g_cdd_base,
        "temp_actual": g_temp_actual,
        "temp_proy_2040": g_temp_2040,
        "delta_temp_2040": round(g_temp_2040 - g_temp_actual, 2),
        "cdd_proy_2040": g_cdd_2040,
        "pct_aumento_cdd": g_pct_cdd,
        "riesgo": clasificar_riesgo_energetico(g_cal_dec, g_pct_cdd)
    }

    # 1. Pre-computar matriz de interpolación base para el mapa rasterizado de calor
    matriz_base = _precomputar_matriz_interpolacion_base()
    GLOBAL_CACHE["matriz_interpolacion_base"] = matriz_base

    # 2. Pre-cachear configuración (/api/config)
    ciudades_lista = [
        {
            "city": c,
            "country": data["country"],
            "lat": data["lat"],
            "lon": data["lon"],
            "riesgo": data["riesgo"]
        }
        for c, data in sorted(METADATA_CIUDADES.items())
    ]
    config_payload = {
        "current_year": CURRENT_YEAR,
        "horizonte_proyeccion": HORIZONTE_PROYECCION,
        "anos_historicos": ANOS_HISTORICOS,
        "min_ano": min(ANOS_HISTORICOS),
        "max_ano_hist": CURRENT_YEAR,
        "anos_proyeccion": PROYECCION_ANOS,
        "max_ano_proy": HORIZONTE_PROYECCION,
        "ciudades": ciudades_lista
    }
    GLOBAL_CACHE["config"] = config_payload
    GLOBAL_CACHE["bytes"]["config"] = json.dumps(config_payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")

    # 3. Pre-computar índice rápido por ciudad en GLOBAL_CACHE["ciudades"]
    GLOBAL_CACHE["ciudades"].clear()
    GLOBAL_CACHE["ciudades_lower"].clear()
    GLOBAL_CACHE["bytes"]["cities"].clear()

    global_payload = construir_payload_ciudad("", GLOBAL_MODEL, GLOBAL_HISTORIAL)
    global_bytes = json.dumps(global_payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    GLOBAL_CACHE["ciudades"][""] = global_payload
    GLOBAL_CACHE["ciudades"]["Global"] = global_payload
    GLOBAL_CACHE["ciudades_lower"][""] = ""
    GLOBAL_CACHE["ciudades_lower"]["global"] = "Global"
    GLOBAL_CACHE["bytes"]["cities"][""] = global_bytes
    GLOBAL_CACHE["bytes"]["cities"]["Global"] = global_bytes

    for city, meta in METADATA_CIUDADES.items():
        city_payload = construir_payload_ciudad(city, meta, meta["historial"])
        city_bytes = json.dumps(city_payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        GLOBAL_CACHE["ciudades"][city] = city_payload
        GLOBAL_CACHE["ciudades_lower"][city.lower()] = city
        GLOBAL_CACHE["bytes"]["cities"][city] = city_bytes

    # 4. Pre-computar series temporales (1850 - 2050) y timeline térmico en GLOBAL_CACHE["series_temporales"]
    timeline_dict = {}
    GLOBAL_CACHE["bytes"]["years"].clear()
    rango_completo = ANOS_HISTORICOS + PROYECCION_ANOS

    for y in rango_completo:
        puntos_ano = []
        es_proyeccion = y > CURRENT_YEAR
        temps_y = []
        cdds_y = []

        for city, data in METADATA_CIUDADES.items():
            if not es_proyeccion:
                t = data["historial"].get(y)
                if t is None:
                    t = round(data["intercept"] + data["slope"] * y, 2)
            else:
                t = round(data["intercept"] + data["slope"] * y, 2)

            cdd_val = calcular_cdd(t)
            peso = calcular_peso_normalizado(t)
            temps_y.append(t)
            cdds_y.append(cdd_val)

            puntos_ano.append({
                "lat": data["lat"],
                "lon": data["lon"],
                "temperatura": t,
                "anomalia": round(t - data["baseline_anom"], 2),
                "cdd": cdd_val,
                "intensidad": peso,
                "city": city,
                "country": data["country"],
                "riesgo": data["riesgo_compacto"]
            })

        timeline_dict[str(y)] = puntos_ano

        kpis_y = {
            "year": y,
            "current_year": CURRENT_YEAR,
            "es_proyeccion": es_proyeccion,
            "total_ciudades": len(puntos_ano),
            "temp_promedio": round(float(np.mean(temps_y)), 2) if temps_y else 0.0,
            "cdd_promedio": round(float(np.mean(cdds_y)), 2) if cdds_y else 0.0,
            "temp_maxima": round(float(np.max(temps_y)), 2) if temps_y else 0.0,
            "ciudad_mas_calida": max(puntos_ano, key=lambda x: x["temperatura"]) if puntos_ano else None
        }
        year_payload = {
            "year": y,
            "current_year": CURRENT_YEAR,
            "es_proyeccion": es_proyeccion,
            "kpis": kpis_y,
            "puntos": puntos_ano
        }
        GLOBAL_CACHE["bytes"]["years"][y] = json.dumps(year_payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")

    series_temporales_payload = {
        "min_year": min(rango_completo),
        "max_year": max(rango_completo),
        "current_year": CURRENT_YEAR,
        "global_historico": global_payload["historico"],
        "global_proyeccion": global_payload["proyeccion"],
        "matriz_interpolacion_base": matriz_base,
        "timeline": timeline_dict
    }
    GLOBAL_CACHE["series_temporales"] = series_temporales_payload
    GLOBAL_CACHE["bytes"]["climate_data"] = json.dumps(series_temporales_payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")

    # 5. Pre-cargar index.html en memoria RAM
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            GLOBAL_CACHE["index_html"] = f.read()

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    print(f"[OK] GLOBAL_CACHE listo en {elapsed_ms:.1f} ms ({len(METADATA_CIUDADES)} ciudades, {len(rango_completo)} años en RAM).")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manejador de ciclo de vida FastAPI: carga única y pre-computación total en GLOBAL_CACHE (RAM)."""
    inicializar_dataset_en_memoria()
    yield
    GLOBAL_CACHE.clear()
    METADATA_CIUDADES.clear()


app = FastAPI(
    title="ClimateAI: Analítica Predictiva para Climatización y Resiliencia Energética",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def _build_dynamic_year_bytes(year: int) -> bytes:
    """Construye en memoria un año fuera del rango precalculado (1850-2050) sin tocar disco."""
    es_proyeccion = year > CURRENT_YEAR
    puntos = []
    temps = []
    cdds = []
    for city, data in METADATA_CIUDADES.items():
        temp = data["historial"].get(year) if not es_proyeccion else None
        if temp is None:
            temp = round(data["intercept"] + data["slope"] * year, 2)
        cdd_val = calcular_cdd(temp)
        temps.append(temp)
        cdds.append(cdd_val)
        puntos.append({
            "city": city,
            "country": data["country"],
            "lat": data["lat"],
            "lon": data["lon"],
            "temperatura": temp,
            "anomalia": round(temp - data["baseline_anom"], 2),
            "cdd": cdd_val,
            "es_proyeccion": es_proyeccion,
            "riesgo": data["riesgo_compacto"]
        })
    payload = {
        "year": year,
        "current_year": CURRENT_YEAR,
        "es_proyeccion": es_proyeccion,
        "kpis": {
            "year": year,
            "current_year": CURRENT_YEAR,
            "es_proyeccion": es_proyeccion,
            "total_ciudades": len(puntos),
            "temp_promedio": round(float(np.mean(temps)), 2) if temps else 0.0,
            "cdd_promedio": round(float(np.mean(cdds)), 2) if cdds else 0.0,
            "temp_maxima": round(float(np.max(temps)), 2) if temps else 0.0,
            "ciudad_mas_calida": max(puntos, key=lambda x: x["temperatura"]) if puntos else None
        },
        "puntos": puntos
    }
    return json.dumps(payload, ensure_ascii=False).encode("utf-8")


@app.get("/", response_class=HTMLResponse)
async def serve_index():
    """Entrega index.html directamente desde GLOBAL_CACHE en RAM."""
    html = GLOBAL_CACHE.get("index_html")
    if not html:
        raise HTTPException(status_code=404, detail="index.html no cargado en memoria")
    return HTMLResponse(content=html)


@app.get("/api/config")
async def get_config():
    """Lookup O(1) en GLOBAL_CACHE para configuración inicial y catálogo de ciudades (< 1 ms)."""
    return Response(content=GLOBAL_CACHE["bytes"]["config"], media_type="application/json")


@app.get("/api/climate-data")
@app.get("/api/thermal-timeline")
async def get_climate_data(year: Optional[int] = Query(None)):
    """
    Lookup O(1) sobre GLOBAL_CACHE en RAM (< 1 ms):
    Retorna las series históricas/proyectadas (1850-2050), timeline anual y matriz base de interpolación,
    o el corte específico de un año si se pasa ?year=.
    """
    if year is not None:
        year_bytes = GLOBAL_CACHE["bytes"]["years"].get(year) or _build_dynamic_year_bytes(year)
        return Response(content=year_bytes, media_type="application/json")
    return Response(content=GLOBAL_CACHE["bytes"]["climate_data"], media_type="application/json")


@app.get("/api/heatmap-year")
async def get_heatmap_year(year: int = Query(...)):
    """Lookup O(1) en GLOBAL_CACHE para un año específico del mapa térmico."""
    year_bytes = GLOBAL_CACHE["bytes"]["years"].get(year) or _build_dynamic_year_bytes(year)
    return Response(content=year_bytes, media_type="application/json")


@app.get("/api/city/{name}")
async def get_city_by_name(name: str):
    """
    Lookup directo O(1) sobre GLOBAL_CACHE["bytes"]["cities"] (< 1 ms):
    Retorna coords, cdd_historico, metricas, serie histórica (1850-actual) y proyección (hasta 2050).
    """
    key = name.strip()
    cities_bytes = GLOBAL_CACHE["bytes"]["cities"]
    if key in cities_bytes:
        return Response(content=cities_bytes[key], media_type="application/json")

    canonical = GLOBAL_CACHE["ciudades_lower"].get(key.lower())
    if canonical is not None and canonical in cities_bytes:
        return Response(content=cities_bytes[canonical], media_type="application/json")

    raise HTTPException(status_code=404, detail=f"Ciudad '{name}' no encontrada en GLOBAL_CACHE")


@app.get("/api/city-detail")
async def get_city_detail(city: str = Query("")):
    """Lookup directo O(1) sobre GLOBAL_CACHE (retrocompatible con ?city=)."""
    key = city.strip()
    cities_bytes = GLOBAL_CACHE["bytes"]["cities"]
    if key in cities_bytes:
        return Response(content=cities_bytes[key], media_type="application/json")

    canonical = GLOBAL_CACHE["ciudades_lower"].get(key.lower())
    if canonical is not None and canonical in cities_bytes:
        return Response(content=cities_bytes[canonical], media_type="application/json")

    return Response(content=cities_bytes["Global"], media_type="application/json")


if __name__ == "__main__":
    import os
    import time
    import threading
    import webbrowser
    import uvicorn

    HOST = "127.0.0.1"
    PORT = 8080
    URL = f"http://{HOST}:{PORT}"

    def launch_browser():
        time.sleep(1.2)
        try:
            webbrowser.open(URL)
        except Exception:
            pass

    try:
        threading.Thread(target=launch_browser, daemon=True).start()
        uvicorn.run(app, host=HOST, port=PORT, reload=False, workers=1)
    except KeyboardInterrupt:
        pass
    finally:
        os._exit(0)
