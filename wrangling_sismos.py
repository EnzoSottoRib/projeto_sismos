#!/usr/bin/env python3
"""
Wrangling: indicador de potencial de estrago sísmico (USGS + NOAA + Banco Mundial)

Dependências:  pip install pandas numpy geopandas
Natural Earth (países, 10m):
  https://naciscdn.org/naturalearth/10m/cultural/ne_10m_admin_0_countries.zip
  (a versão 50m é bem mais rápida e basta para um teste)

Saídas (pasta ./saida):
  events_modelo.csv            -> PRONTO p/ regressão/classificação (mainshocks M>=6 com covariáveis)
  countries_cluster.csv        -> PRONTO p/ clustering (países com M6+ e covariáveis)
  events.csv                   -> todos os eventos USGS (com flag eh_mainshock), p/ exploração
  countries.csv                -> todos os países do BM (inclusive sem sismos)
  noaa_sem_match.csv           -> eventos NOAA com mortes que não casaram com o USGS (inspecionar!)
  events_sem_covariaveis.csv   -> eventos sem PIB/densidade (ex.: Taiwan)
"""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# CONFIGURAÇÃO
# --------------------------------------------------------------------------- #

PASTA_PROJETO = Path(__file__).resolve().parent
PASTA_DADOS = PASTA_PROJETO / "dados"
PASTA_SAIDA = PASTA_PROJETO / "saida"

CFG = dict(
    usgs=str(PASTA_DADOS / "usgs_1990_2023_m55.csv"),
    noaa=str(PASTA_DADOS / "noaa_earthquakes.tsv"),

    wb_gdp=str(PASTA_DADOS / "API_NY.GDP.PCAP.CD_DS2_en_csv_v2_1529.csv"),

    wb_dens=str(PASTA_DADOS / "API_EN.POP.DNST_DS2_en_csv_v2_7671.csv"),

    wb_meta=str(PASTA_DADOS / "Metadata_Country_API_NY.GDP.PCAP.CD_DS2_en_csv_v2_1529.csv"),

    countries_shp=str(PASTA_DADOS / "ne_50m_admin_0_countries.zip"),

    wb_extra=str(PASTA_DADOS / "covariaveis_extra.csv"),

    y0=1990,
    y1=2023,
    min_mag=5.5,
    max_offshore_km=250,
    match_km=200,
    match_dmag=1.0,
    max_gap_borda=12,
    janela_covar=5,
    mag_pais=6.0,
    decl_dias=30,
    decl_km=100,
    vulcao_excluir=[("1991-06-15", "PHL")]
)

OUT = PASTA_SAIDA

# NOAA: códigos de descrição (usados quando o número de mortes está em branco)
DESC_RANGE = {0: (0, 0), 1: (1, 50), 2: (51, 100), 3: (101, 1000), 4: (1001, np.inf)}

# Natural Earth -> código do Banco Mundial (confira o relatório de ISO3 sem match)
ISO_FIX = {"KOS": "XKX", "SDS": "SSD", "PSX": "PSE"}

# Casos conhecidos para validar visualmente o match
CASOS_CONHECIDOS = {
    "Sumatra/Oceano Índico 2004": "2004-12-26",
    "Sichuan 2008": "2008-05-12",
    "Haiti 2010": "2010-01-12",
    "Tohoku 2011": "2011-03-11",
    "Nepal 2015": "2015-04-25",
    "Turquia/Síria 2023": "2023-02-06",
}


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


# --------------------------------------------------------------------------- #
# 1. USGS
# --------------------------------------------------------------------------- #
def load_usgs():
    cols = ["time", "latitude", "longitude", "depth", "mag", "magType", "place", "id"]
    df = pd.read_csv(CFG["usgs"], usecols=cols, encoding_errors="replace")
    df["time"] = pd.to_datetime(df["time"], utc=True).dt.tz_convert(None)  # UTC naive
    df = df.drop_duplicates("id").dropna(subset=["time", "latitude", "longitude", "mag"])
    print(f"[USGS] arquivo cobre {df.time.min()} -> {df.time.max()} ({len(df)} eventos)")
    df = df[
        df.time.dt.year.between(CFG["y0"], CFG["y1"]) & (df.mag >= CFG["min_mag"])
    ].sort_values("time").reset_index(drop=True)
    por_ano = df.groupby(df.time.dt.year).size()
    faltando = sorted(set(range(CFG["y0"], CFG["y1"] + 1)) - set(por_ano.index))
    if faltando:
        print("  ATENCAO: anos sem nenhum evento (download incompleto?):", faltando)
    print(f"  eventos/ano (min / mediana / max): {por_ano.min()} / {por_ano.median():.0f} / {por_ano.max()}")
    return df


def geocodificar(df):
    """lat/lon -> ISO3. Dentro do polígono = país; no mar = país mais próximo (<= max_offshore_km)."""
    import geopandas as gpd
    from shapely.ops import nearest_points

    world = gpd.read_file(CFG["countries_shp"]).to_crs("EPSG:4326")
    # ISO_A3 vem como -99 para França, Noruega etc.; ISO_A3_EH corrige a maioria
    iso = world["ISO_A3_EH"].where(world["ISO_A3_EH"].notna() & (world["ISO_A3_EH"] != "-99"), world["ADM0_A3"])
    world = world.assign(iso3=iso.replace(ISO_FIX))[["iso3", "geometry"]].reset_index(drop=True)

    pts = gpd.GeoDataFrame(
        df[["latitude", "longitude"]].copy(),
        geometry=gpd.points_from_xy(df["longitude"], df["latitude"]),
        crs="EPSG:4326",
    )
    j = gpd.sjoin(pts, world, how="left", predicate="within")
    j = j[~j.index.duplicated(keep="first")]
    df = df.copy()
    df["iso3"] = j["iso3"]
    df["dist_costa_km"] = np.where(df["iso3"].notna(), 0.0, np.nan)

    off = df.index[df["iso3"].isna()]
    print(f"[GEO] {len(df) - len(off)} eventos em terra; {len(off)} no mar (buscando país mais próximo; pode demorar)")
    if len(off):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            nn = gpd.sjoin_nearest(pts.loc[off], world, how="left")
        nn = nn[~nn.index.duplicated(keep="first")].copy()
        dist = []
        for _, row in nn.iterrows():
            g = world.geometry.iloc[int(row["index_right"])]
            q = nearest_points(row.geometry, g)[1]
            dist.append(haversine_km(row.geometry.y, row.geometry.x, q.y, q.x))
        nn["d_km"] = dist
        ok = nn["d_km"] <= CFG["max_offshore_km"]
        df.loc[nn.index[ok], "iso3"] = nn.loc[ok, "iso3"]
        df.loc[nn.index, "dist_costa_km"] = nn["d_km"]
        print(f"  atribuídos a país: {ok.sum()} | mar aberto (descartáveis): {(~ok).sum()}")
    return df


# --------------------------------------------------------------------------- #
# 1b. Declustering (mainshock x réplicas/precursores)
# --------------------------------------------------------------------------- #
def declusterizar(df):
    """Marca como mainshock o maior evento de cada sequência: qualquer evento menor ocorrido nos
    decl_dias SEGUINTES e dentro de um raio R de um evento maior é marcado como réplica.
    (Eventos anteriores ao maior NÃO são fundidos: um precursor que matou gente continua sendo evento próprio.) Não apaga linhas.
    R = max(decl_km, comprimento de ruptura esperado) com log10(L) = -2.44 + 0.59*M (Wells & Coppersmith 1994),
    para que réplicas grandes de terremotos gigantes (ex.: M7.9 após Tohoku M9.1) não virem 'mainshocks'."""
    df = df.copy()
    t = df["time"].values
    la, lo, mg = df["latitude"].values, df["longitude"].values, df["mag"].values
    n = len(df)
    estado = np.zeros(n, dtype=np.int8)      # 0 livre | 1 mainshock | 2 réplica/precursor
    pai = np.full(n, -1)
    janela = np.timedelta64(CFG["decl_dias"], "D")
    for i in np.argsort(-mg, kind="stable"):  # do maior para o menor
        if estado[i] != 0:
            continue
        estado[i] = 1
        a = np.searchsorted(t, t[i], side="left")
        b = np.searchsorted(t, t[i] + janela, side="right")
        idx = np.arange(a, b)
        idx = idx[estado[idx] == 0]
        if len(idx) == 0:
            continue
        raio = max(CFG["decl_km"], 10 ** (-2.44 + 0.59 * mg[i]))
        d = haversine_km(la[i], lo[i], la[idx], lo[idx])
        viz = idx[d <= raio]
        estado[viz] = 2
        pai[viz] = i
    ids = df["id"].values
    df["eh_mainshock"] = estado == 1
    df["id_mainshock"] = [ids[p] if p >= 0 else None for p in pai]
    print(f"[DECLUSTER] {int((estado == 1).sum())} mainshocks | {int((estado == 2).sum())} réplicas/precursores "
          f"(até {CFG['decl_dias']} dias após o evento maior, raio >= {CFG['decl_km']} km, crescente com a magnitude)")
    return df


# --------------------------------------------------------------------------- #
# 2. NOAA
# --------------------------------------------------------------------------- #
def load_noaa():
    df = pd.read_csv(CFG["noaa"], sep="\t", encoding_errors="replace")
    df = df.drop(columns=["Search Parameters"], errors="ignore")  # coluna lixo (linha "[]")
    df.columns = [c.strip() for c in df.columns]
    for c in df.columns:
        if c != "Location Name":
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df[df["Year"].between(CFG["y0"], CFG["y1"])].copy().reset_index(drop=True)

    df = df.rename(columns={
        "Location Name": "local", "Vol": "vulcao_id", "Latitude": "lat", "Longitude": "lon", "Mag": "mag_noaa",
        "Deaths": "mortes_diretas", "Total Deaths": "mortes_total",
        "Death Description": "desc_dir", "Total Death Description": "desc_total",
        "Total Damage ($Mil)": "dano_total_musd",
    })
    for c in ["mortes_total", "mortes_diretas", "desc_total", "desc_dir", "dano_total_musd", "mag_noaa", "vulcao_id"]:
        if c not in df:
            df[c] = np.nan

    # Mortes: preferir TOTAL (inclui tsunami/incêndio, que são consequência do sismo)
    df["mortes"] = df["mortes_total"].fillna(df["mortes_diretas"])
    df["desc"] = df["desc_total"].fillna(df["desc_dir"])
    df.loc[df["mortes"].isna() & (df["desc"] == 0), "mortes"] = 0
    lo = df["desc"].map(lambda k: DESC_RANGE.get(k, (np.nan, np.nan))[0])
    hi = df["desc"].map(lambda k: DESC_RANGE.get(k, (np.nan, np.nan))[1])
    df["mortes_min"] = df["mortes"].fillna(lo)
    df["mortes_max"] = df["mortes"].fillna(hi)

    # Evento listado, sem número e sem descrição -> assume 0 (com flag, para teste de sensibilidade)
    sem_info = df["mortes_min"].isna() & df["mortes_max"].isna()
    df["mortes_assumida_zero"] = sem_info
    df.loc[sem_info, ["mortes", "mortes_min", "mortes_max"]] = 0

    # Alta mortalidade: >=100 certo; <100 certo; faixa 51-100 (desc=2) fica ambígua (NaN)
    df["alta_mortalidade"] = np.select(
        [df["mortes_min"] >= 100, df["mortes_max"] < 100], [1.0, 0.0], default=np.nan
    )

    # Janela de datas (NOAA tem datas incompletas)
    def janela(r):
        y = int(r["Year"])
        try:
            if pd.isna(r["Mo"]):
                a, b = pd.Timestamp(y, 1, 1), pd.Timestamp(y + 1, 1, 1)
            elif pd.isna(r["Dy"]) or r["Dy"] == 0:
                a = pd.Timestamp(y, int(r["Mo"]), 1)
                b = a + pd.offsets.MonthBegin(1)
            else:
                a = pd.Timestamp(y, int(r["Mo"]), int(r["Dy"]))
                b = a + pd.Timedelta(days=1)
        except ValueError:
            a, b = pd.Timestamp(y, 1, 1), pd.Timestamp(y + 1, 1, 1)
        return pd.Series({"t_lo": a - pd.Timedelta(hours=12), "t_hi": b + pd.Timedelta(hours=12)})

    df = pd.concat([df, df.apply(janela, axis=1)], axis=1)
    print(f"[NOAA] {len(df)} eventos em {CFG['y0']}-{CFG['y1']} | "
          f"com mortes>0: {(df.mortes_max.fillna(0) > 0).sum()} | "
          f"mortes assumidas 0: {int(df.mortes_assumida_zero.sum())}")
    return df


# --------------------------------------------------------------------------- #
# 3. Match USGS <-> NOAA (sem ID comum: tempo + distância + magnitude)
# --------------------------------------------------------------------------- #
def match_noaa_usgs(noaa, usgs):
    t = usgs["time"].values
    la, lo, mg = usgs["latitude"].values, usgs["longitude"].values, usgs["mag"].values
    usadas = set()
    res = {}
    # os eventos mais letais escolhem primeiro (evita que uma réplica "roube" o principal)
    ordem = noaa.sort_values("mortes_max", ascending=False, na_position="last").index
    for i in ordem:
        r = noaa.loc[i]
        res[i] = (np.nan, np.nan, np.nan)
        if pd.isna(r["lat"]) or pd.isna(r["lon"]):
            continue
        a = np.searchsorted(t, r["t_lo"].to_datetime64(), side="left")
        b = np.searchsorted(t, r["t_hi"].to_datetime64(), side="right")
        if b <= a:
            continue
        cand = np.arange(a, b)
        d = haversine_km(r["lat"], r["lon"], la[cand], lo[cand])
        dm = np.abs(mg[cand] - r["mag_noaa"]) if pd.notna(r["mag_noaa"]) else np.zeros(len(cand))
        ok = (d <= CFG["match_km"]) & (dm <= CFG["match_dmag"]) & ~np.isin(cand, list(usadas))
        score = np.where(ok, d + 100 * dm, np.inf)
        k = int(score.argmin())
        if np.isfinite(score[k]):
            usadas.add(int(cand[k]))
            res[i] = (int(cand[k]), float(d[k]), float(dm[k]))
    m = pd.DataFrame.from_dict(res, orient="index", columns=["usgs_idx", "match_dist_km", "match_dmag"])
    return noaa.join(m)


def relatorio_match(noaa_m):
    n, ok = len(noaa_m), noaa_m["usgs_idx"].notna().sum()
    print(f"[MATCH] NOAA casados com USGS: {ok}/{n} ({ok / n:.0%})")
    nao = noaa_m[noaa_m["usgs_idx"].isna() & (noaa_m["mortes_max"].fillna(0) > 0)]
    nao = nao.sort_values("mortes_max", ascending=False)
    nao.to_csv(OUT / "noaa_sem_match.csv", index=False)
    print(f"  NOAA com mortes e SEM match: {len(nao)} (salvo em noaa_sem_match.csv). Maiores:")
    print(nao[["Year", "Mo", "Dy", "local", "lat", "lon", "mag_noaa", "mortes_max"]].head(12).to_string(index=False))
    print("  (esperado: quase todos com mag < 5.5, fora do universo; M>=5.5 aqui = investigar)")


# --------------------------------------------------------------------------- #
# 4. Banco Mundial
# --------------------------------------------------------------------------- #
def load_wb(path, nome):
    with open(path, encoding="utf-8-sig") as f:        # o CSV tem linhas de metadados antes do cabeçalho
        skip = next(i for i, ln in enumerate(f) if "Country Name" in ln and "Country Code" in ln)
    df = pd.read_csv(path, skiprows=skip, encoding="utf-8-sig")
    df = df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed")]   # vírgula final cria coluna vazia
    anos = [c for c in df.columns if str(c).isdigit()]
    long = df.melt(id_vars=["Country Code"], value_vars=anos, var_name="ano", value_name=nome)
    long["ano"] = long["ano"].astype(int)
    return long.rename(columns={"Country Code": "iso3"})


def adicionar_extra(wb):
    """Acrescenta países ausentes no BM (ex.: Taiwan) a partir de CSV manual. Colunas: iso3, ano,
    pib_pc, dens_pop (+ opcionais pais, regiao, grupo_renda). Anos em branco são interpolados."""
    caminho = Path(CFG["wb_extra"])

    if not caminho.exists():
        modelo = PASTA_DADOS / "covariaveis_extra_MODELO.csv"
        modelo = Path("covariaveis_extra_MODELO.csv")
        if not modelo.exists():
            pd.DataFrame({
                "iso3": "TWN", "pais": "Taiwan", "regiao": "East Asia & Pacific",
                "grupo_renda": "High income",
                "ano": range(CFG["y0"], CFG["y1"] + 1), "pib_pc": np.nan, "dens_pop": np.nan,
            }).to_csv(modelo, index=False)
            print(f"[BM] modelo p/ Taiwan criado: {modelo} (preencha, salve como {caminho} e rode de novo)")
        return wb
    ex = pd.read_csv(caminho)
    ex = ex.dropna(subset=["pib_pc", "dens_pop"], how="all")
    if ex.empty:
        print(f"[BM] {caminho} está sem valores; ignorado")
        return wb
    for c in ["pais", "regiao", "grupo_renda"]:
        if c not in ex:
            ex[c] = np.nan
    anos = sorted(wb["ano"].unique())
    grade = pd.MultiIndex.from_product([ex["iso3"].unique(), anos], names=["iso3", "ano"]).to_frame(index=False)
    ex = grade.merge(ex, on=["iso3", "ano"], how="left")
    for c in ["pais", "regiao", "grupo_renda"]:
        ex[c] = ex.groupby("iso3")[c].transform("first")
    wb = wb[~wb["iso3"].isin(ex["iso3"].unique())]       # a série manual substitui a do BM
    print(f"[BM] covariáveis extras incorporadas: {sorted(ex['iso3'].unique())}")
    return pd.concat([wb, ex[wb.columns.intersection(ex.columns)]], ignore_index=True).sort_values(["iso3", "ano"])


def build_wb():
    wb = load_wb(CFG["wb_gdp"], "pib_pc").merge(
        load_wb(CFG["wb_dens"], "dens_pop"), on=["iso3", "ano"], how="outer"
    )
    meta = pd.read_csv(CFG["wb_meta"], encoding="utf-8-sig")
    # Agregados (World, Euro area, AFE...) têm Region vazio -> ficam de fora
    paises = meta.loc[meta["Region"].notna(), ["Country Code", "TableName", "Region", "IncomeGroup"]]
    paises.columns = ["iso3", "pais", "regiao", "grupo_renda"]
    wb = wb.merge(paises, on="iso3", how="inner").sort_values(["iso3", "ano"])
    wb = adicionar_extra(wb)

    lim = CFG["max_gap_borda"]
    for c in ["pib_pc", "dens_pop"]:
        original_nan = wb[c].isna()
        wb[c] = wb.groupby("iso3")[c].transform(
            lambda s: s.interpolate(limit_area="inside").ffill(limit=lim).bfill(limit=lim)
        )
        wb[c + "_imputado"] = original_nan & wb[c].notna()
    wb["log_pib_pc"] = np.log(wb["pib_pc"])
    wb["log_dens_pop"] = np.log(wb["dens_pop"])
    print(f"[BM] {wb.iso3.nunique()} países | NaN restante em {CFG['y0']}-{CFG['y1']}: "
          f"PIB {wb[wb.ano.between(CFG['y0'], CFG['y1'])].pib_pc.isna().mean():.1%}, "
          f"dens {wb[wb.ano.between(CFG['y0'], CFG['y1'])].dens_pop.isna().mean():.1%}")
    return wb


# --------------------------------------------------------------------------- #
# 5. Datasets finais
# --------------------------------------------------------------------------- #
def tratar_mortes(ev):
    """(a) zera mortes de eventos listados em vulcao_excluir (mortes de erupção, ex.: Pinatubo 1991);
    (b) soma ao mainshock as mortes de réplicas/precursores que tenham registro próprio no NOAA
        (senão elas sumiriam do dataset de modelagem, que só usa mainshocks);
    (c) recalcula alta_mortalidade e log_mortes."""
    cols_m = ["mortes", "mortes_min", "mortes_max"]
    volc = ev["vulcao_id"].notna() & (ev["mortes_max"].fillna(0) > 0)
    ev["mortes_removidas_vulcao"] = 0.0
    if volc.any():
        print("[VULCÃO] eventos NOAA com campo Vol preenchido e mortes (REVISE: Vol não implica que as mortes sejam vulcânicas):")
        print(ev[volc].sort_values("mortes", ascending=False)[["time", "mag", "iso3", "mortes"]].head(10).to_string(index=False))
        chave = ev["time"].dt.strftime("%Y-%m-%d") + "|" + ev["iso3"].fillna("")
        lista = {f"{d}|{c}" for d, c in CFG["vulcao_excluir"]}
        zerar = volc & chave.isin(lista)
        if zerar.any():
            ev.loc[zerar, "mortes_removidas_vulcao"] = ev.loc[zerar, "mortes"].fillna(0)
            ev.loc[zerar, cols_m] = 0
            print(f"  -> mortes zeradas (lista vulcao_excluir): {int(zerar.sum())} evento(s), "
                  f"{int(ev.loc[zerar, 'mortes_removidas_vulcao'].sum())} mortes")

    ev = ev.set_index("usgs_id", drop=False)
    ev["mortes_recebidas"] = 0.0
    mov = ev[~ev["eh_mainshock"] & ev["noaa_match"] & (ev["mortes_max"].fillna(0) > 0) & ev["id_mainshock"].notna()]
    if len(mov):
        soma = mov.groupby("id_mainshock")[cols_m].sum()
        for c in cols_m:
            ev.loc[soma.index, c] = ev.loc[soma.index, c] + soma[c]
        ev.loc[soma.index, "mortes_recebidas"] = soma["mortes"]
        ev.loc[mov.index, cols_m] = 0
        print(f"[MORTES] {len(mov)} réplicas/precursores com mortes próprias somadas ao mainshock da sequência. Maiores:")
        top = soma["mortes"].sort_values(ascending=False).head(8).index
        print(ev.loc[top, ["time", "mag", "iso3", "mortes_recebidas", "mortes"]].to_string(index=False))
    ev = ev.reset_index(drop=True)
    ev["alta_mortalidade"] = np.select(
        [ev["mortes_min"] >= 100, ev["mortes_max"] < 100], [1.0, 0.0], default=np.nan
    )
    ev["log_mortes"] = np.log1p(ev["mortes"])         # NaN quando só há faixa (desc), sem número
    return ev


def build_events(usgs, noaa_m, wb):
    n = noaa_m[noaa_m["usgs_idx"].notna()].copy()
    n["usgs_idx"] = n["usgs_idx"].astype(int)
    n = n.set_index("usgs_idx")
    n["noaa_match"] = True
    cols = ["noaa_match", "mortes", "mortes_min", "mortes_max", "alta_mortalidade",
            "mortes_assumida_zero", "dano_total_musd", "match_dist_km", "match_dmag", "local", "vulcao_id"]
    ev = usgs.join(n[cols], how="left")
    ev["noaa_match"] = ev["noaa_match"].fillna(False).astype(bool)

    # Sem registro no NOAA = terremoto "insignificante" -> 0 mortes (premissa do desenho)
    sem = ~ev["noaa_match"]
    ev.loc[sem, ["mortes", "mortes_min", "mortes_max", "alta_mortalidade"]] = 0
    ev["sem_registro_noaa"] = sem
    ev["ano"] = ev["time"].dt.year
    ev = ev.rename(columns={"id": "usgs_id", "depth": "prof_km"})
    ev = tratar_mortes(ev)        # vulcões, réplicas -> mainshock; recalcula alta_mortalidade e log_mortes

    ev = ev.merge(
        wb[["iso3", "ano", "pib_pc", "dens_pop", "log_pib_pc", "log_dens_pop",
            "pib_pc_imputado", "dens_pop_imputado", "regiao", "grupo_renda"]],
        on=["iso3", "ano"], how="left",
    )
    return ev


def build_paises(ev, wb):
    n_anos = CFG["y1"] - CFG["y0"] + 1
    m = ev[(ev["mag"] >= CFG["mag_pais"]) & ev["iso3"].notna()]
    tot = m.groupby("iso3").agg(
        n_eventos_bruto=("usgs_id", "size"),
        mortes_totais=("mortes", "sum"),
        n_alta_mortalidade=("alta_mortalidade", "sum"),
    )
    ms = m[m["eh_mainshock"]].groupby("iso3").agg(      # contagens só com mainshocks (sem réplicas)
        n_eventos=("usgs_id", "size"),
        anos_com_evento=("ano", "nunique"),
        mag_max=("mag", "max"),
    )
    agg = tot.join(ms, how="left")
    agg["mortes_medias_evento"] = np.where(agg["n_eventos"] > 0, agg["mortes_totais"] / agg["n_eventos"], np.nan)
    agg["taxa_anual"] = agg["n_eventos"] / n_anos
    agg["prob_anual_poisson"] = 1 - np.exp(-agg["taxa_anual"])          # assume eventos independentes
    agg["prob_anual_empirica"] = agg["anos_com_evento"] / n_anos         # robusta a clusters de réplicas

    rec = wb[wb["ano"] > CFG["y1"] - CFG["janela_covar"]].groupby("iso3").agg(
        pais=("pais", "first"), regiao=("regiao", "first"), grupo_renda=("grupo_renda", "first"),
        pib_pc=("pib_pc", "mean"), dens_pop=("dens_pop", "mean"),
    )
    p = rec.join(agg, how="left")
    for c in ["n_eventos", "n_eventos_bruto", "anos_com_evento", "mortes_totais", "n_alta_mortalidade",
              "taxa_anual", "prob_anual_poisson", "prob_anual_empirica"]:
        p[c] = p[c].fillna(0)
    p["tem_evento"] = p["n_eventos"] > 0
    p["log_pib_pc"], p["log_dens_pop"] = np.log(p["pib_pc"]), np.log(p["dens_pop"])
    return p.reset_index()


def validar(ev):
    print("\n[VALIDAÇÃO] casos conhecidos (mag >= 6.5, +-1 dia). Colunas: tempo | mag | país | mortes | eh_mainshock | noaa_match")
    for nome, d in CASOS_CONHECIDOS.items():
        d = pd.Timestamp(d)
        s = ev[(ev.time.between(d - pd.Timedelta(days=1), d + pd.Timedelta(days=2))) & (ev.mag >= 6.5)]
        print(f"  {nome}:")
        if s.empty:
            print("    (nenhum evento encontrado)")
        else:
            print(s[["time", "mag", "iso3", "mortes", "eh_mainshock", "noaa_match"]].to_string(index=False, header=False))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    usgs = declusterizar(geocodificar(load_usgs()))
    noaa_m = match_noaa_usgs(load_noaa(), usgs)
    relatorio_match(noaa_m)
    wb = build_wb()

    ev = build_events(usgs, noaa_m, wb)
    ev = ev[ev["iso3"].notna()].copy()                     # remove mar aberto

    fora_bm = ev[ev["pib_pc"].isna() | ev["dens_pop"].isna()]
    fora_bm.to_csv(OUT / "events_sem_covariaveis.csv", index=False)
    print(f"\n[JOIN] eventos sem PIB/densidade: {len(fora_bm)} -> principais países:")
    print(fora_bm.groupby("iso3").agg(n=("usgs_id", "size"), mortes=("mortes", "sum"))
          .sort_values("mortes", ascending=False).head(10).to_string())

    # mortes que o match atribuiu a um evento que NÃO é o mainshock da sequência (idealmente ~0)
    desv = ev[ev["noaa_match"] & ~ev["eh_mainshock"] & (ev["mortes"] > 0)]
    print(f"\n[DECLUSTER] eventos com mortes atribuídas a réplica/precursor: {len(desv)}")
    if len(desv):
        print(desv[["time", "mag", "iso3", "mortes", "id_mainshock"]].sort_values("mortes", ascending=False)
              .head(8).to_string(index=False))

    validar(ev)
    paises = build_paises(ev, wb)

    ev.to_csv(OUT / "events.csv", index=False)
    paises.to_csv(OUT / "countries.csv", index=False)

    # datasets prontos para análise
    mod = ev[ev["eh_mainshock"] & (ev["mag"] >= CFG["mag_pais"]) & ev["pib_pc"].notna() & ev["dens_pop"].notna()]
    mod.to_csv(OUT / "events_modelo.csv", index=False)
    clu = paises[paises["tem_evento"]].dropna(subset=["pib_pc", "dens_pop"])
    clu.to_csv(OUT / "countries_cluster.csv", index=False)

    print(f"\n[RESUMO] events.csv: {len(ev)} | countries.csv: {len(paises)}")
    print(f"  events_modelo.csv: {len(mod)} mainshocks M>={CFG['mag_pais']} | "
          f"Alta_Mortalidade=1: {int(mod.alta_mortalidade.fillna(0).sum())} "
          f"({mod.alta_mortalidade.fillna(0).mean():.1%}) | log_mortes==0: {(mod.log_mortes == 0).mean():.0%}")
    print(f"  countries_cluster.csv: {len(clu)} países")

    return ev, paises, mod, clu


if __name__ == "__main__":
    main()