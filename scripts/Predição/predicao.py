# rondon florest — predição de gênero + atualização em SQL Server (ROW_ID)
# - RF contextual + Modelo de NOME (TF-IDF+LR calibrada) + Bayes (nome/UF, nome BR)
# - Ajuste adaptativo de limiares até a meta
# - Fallback para 100% cobertura (marca IMPUTACAO_FORCADA)
# - LOG inclui QTD_FORCADOS e PCT_FORCADOS
# - Staging com to_sql(method=None) + fast_executemany=True (sem erro de parameter markers)
# - FIX: coalescência e deduplicação de colunas para evitar erro "Selected columns are not unique"

import os, json, uuid, datetime, numpy as np, pandas as pd, unicodedata, re
from sqlalchemy import create_engine, text
from sqlalchemy import types as satypes
import urllib.parse

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

# ====== CONFIG GERAL ======
SERVER        = r"localhost"
DATABASE      = "IC" # nome do banco
SCHEMA        = "dbo"
TABELA_ORIG   = "STEM_Y" # nome da tabela
ODBC_DRIVER   = "ODBC Driver 17 for SQL Server"
KEY_COL       = "ROW_ID"
NAME_COL      = "NM_AUTOR"   # nome completo

# Rótulos
FEMALE_LABEL  = "FEMININO"
MALE_LABEL    = "MASCULINO"
INDET_LABEL   = "INDETERMINADO"

MODELO_PATH      = "modelo_genero.joblib"             # RF (contexto)
MODELO_NOME_PATH = "modelo_genero_nome.joblib"        # TF-IDF + LR calibrada
METRICS_PATH     = "modelo_genero_metrics.json"
METRICS_NOME     = "modelo_genero_nome_metrics.json"
CHECKPOINT_PATH  = "checkpoint_predicao.log"

FEATURES = [
    "NM_AREA_BASICA","NM_MODALIDADE_PROGRAMA","NM_GRAU_PROGRAMA","DS_SITUACAO_PROGRAMA",
    "NM_REGIAO","SG_UF_PROGRAMA","DS_DEPENDENCIA_ADMINISTRATIVA","NM_ENTIDADE_ENSINO",
    "NM_PROGRAMA_FOMENTO","NM_FINANCIADOR","NM_NATUREZA_FINANCIAMENTO",
    "NM_TIPO_PRODUCAO","NM_SUBTIPO_PRODUCAO","TP_AUTOR",
]
ALVO            = "GENERO"

# ====== ALVOS DE COBERTURA E LIMIARES ======
TARGET_COVERAGE        = 0.80   # meta da cascata (antes do fallback)
MIN_CONF_FLOOR         = 0.65
ADAPT_STEP             = 0.02

THRESH_RF_INIT         = 0.90
THRESH_NOME_ML_INIT    = 0.85
THRESH_BAYES_UF_INIT   = 0.82
THRESH_BAYES_BR_INIT   = 0.80

# ====== TREINO ======
TEST_SIZE       = 0.20
RANDOM_STATE    = 42
N_TREES         = 400

# Modelo de nome
TFIDF_NGRAM     = (2,4)
LR_MAX_ITER     = 2000

# Bayes suavização e suportes
ALPHA_PRIOR     = 1.0
MIN_SUP_UF      = 30
MIN_SUP_BR      = 10

# ====== STAGING / SQL ======
CHUNK_STAGE     = 10_000
STAGING_DTYPES  = {
    KEY_COL: satypes.BIGINT(),
    "GENERO_PRED": satypes.NVARCHAR(length=20),
    "PROB_GENERO_PRED": satypes.Float(precision=53),
    "FOI_IMPUTADO": satypes.Boolean(),
    "MODO_IMPUTACAO": satypes.NVARCHAR(length=16),
    "CONF_FONTE": satypes.NVARCHAR(length=16),
    "IMPUTACAO_FORCADA": satypes.Boolean(),
}

# ====== UTILS ======
def checkpoint(msg: str):
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    try:
        with open(CHECKPOINT_PATH, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass
    print(line, flush=True)

def strip_accents(s: str) -> str:
    if not isinstance(s, str):
        return s
    nfkd = unicodedata.normalize("NFKD", s)
    return "".join(c for c in nfkd if not unicodedata.combining(c))

def extract_first_name(name: str) -> str | None:
    if not isinstance(name, str): return None
    name = name.strip()
    if not name: return None
    up = strip_accents(name.upper())
    up = re.sub(r"[^A-Z]+", " ", up).strip()
    return up.split()[0] if up else None

def mode_series(s):
    try:
        return s.mode(dropna=True).iloc[0]
    except Exception:
        return np.nan

def compute_coverage(df_base, mask_scope):
    tot = int(mask_scope.sum())
    if tot == 0:
        return 1.0, 0
    ok = int((mask_scope & df_base["FOI_IMPUTADO"]).sum())
    return ok / tot, ok

def coalesce_duplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Une colunas com o mesmo nome (coalescendo valores não nulos) e garante nomes únicos.
    Mantém o primeiro valor não nulo da esquerda para a direita.
    """
    dup_names = [c for c, n in df.columns.value_counts().items() if n > 1]
    if dup_names:
        checkpoint(f"Atenção: detectadas colunas duplicadas -> {dup_names}")
    for col in dup_names:
        same_cols = [c for c in df.columns if c == col]
        base = df[same_cols[0]].copy()
        for c in same_cols[1:]:
            base = base.combine_first(df[c])
        # remove todas as duplicatas e reintroduz uma única coluna
        df.drop(columns=same_cols, inplace=True)
        df[col] = base
    # por garantia (se algo restar duplicado), dropa mantendo a 1ª
    if df.columns.duplicated().any():
        before = len(df.columns)
        df = df.loc[:, ~df.columns.duplicated()]
        after = len(df.columns)
        checkpoint(f"Removidas {before - after} colunas duplicadas residuais.")
    return df

# ====== MAIN ======
checkpoint("==== Início da execução ====")

# Conexão
params = {
    "Driver": ODBC_DRIVER, "Server": SERVER, "Database": DATABASE,
    "Trusted_Connection": "yes", "TrustServerCertificate": "yes",
}
engine_url = "mssql+pyodbc:///?odbc_connect=" + urllib.parse.quote_plus(
    ";".join([f"{k}={v}" for k,v in params.items()])
)
engine = create_engine(engine_url, fast_executemany=True, pool_pre_ping=True, future=True)

try:
    # ====== Ler tabela ======
    checkpoint(f"Lendo [{SCHEMA}].[{TABELA_ORIG}]...")
    with engine.begin() as conn:
        df_all = pd.read_sql(text(f"SELECT * FROM [{SCHEMA}].[{TABELA_ORIG}]"), conn)

    checkpoint(f"Linhas totais: {len(df_all):,}")

    # ====== FIX: coalescer/deduplicar colunas (evita erro do sklearn) ======
    df_all = coalesce_duplicate_columns(df_all)

    # Checagens de schema
    for needed in [KEY_COL, NAME_COL]:
        if needed not in df_all.columns:
            raise ValueError(f"Coluna obrigatória ausente: {needed}")
    faltantes = [c for c in FEATURES if c not in df_all.columns]
    if faltantes:
        raise ValueError(f"Features ausentes: {faltantes}")

    # ====== Preparar para treino ======
    checkpoint("Preparando dados para treino/validação (RF + Nome)...")
    cols_treino = list(dict.fromkeys(FEATURES + [ALVO, NAME_COL, "SG_UF_PROGRAMA"]))
    df_train = df_all[cols_treino].copy()

    # Normaliza rótulos de treino (F/M -> FEMININO/MASCULINO)
    alvo_raw = df_train[ALVO].astype(str).str.strip().str.upper()
    map_train = {"F": FEMALE_LABEL, "M": MALE_LABEL, FEMALE_LABEL: FEMALE_LABEL, MALE_LABEL: MALE_LABEL}
    df_train[ALVO] = alvo_raw.map(map_train)
    df_train = df_train.dropna(subset=[ALVO])

    # Primeiro nome (para regras por nome)
    df_train["FIRST_NAME_NORM"] = df_train[NAME_COL].apply(extract_first_name)

    # ====== RF (contexto) ======
    X = df_train[FEATURES]
    y = df_train[ALVO]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    checkpoint(f"Treino RF: {len(X_train):,} | Teste: {len(X_test):,}")

    preprocess = ColumnTransformer(
        [("cat", Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("oh", OneHotEncoder(handle_unknown="ignore"))
        ]), FEATURES)],
        remainder="drop"
    )

    rf = RandomForestClassifier(
        n_estimators=N_TREES, class_weight="balanced",
        random_state=RANDOM_STATE, n_jobs=-1
    )
    pipe_rf = Pipeline([("prep", preprocess), ("rf", rf)])

    checkpoint("Treinando RandomForest...")
    pipe_rf.fit(X_train, y_train)
    checkpoint("RF treinado.")

    # Métricas RF
    y_pred = pipe_rf.predict(X_test)
    classes_rf = pipe_rf.named_steps["rf"].classes_
    cm_rf = confusion_matrix(y_test, y_pred, labels=classes_rf, normalize="true")
    report_rf = classification_report(y_test, y_pred, labels=classes_rf, output_dict=True, zero_division=0)
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "when": datetime.datetime.now().isoformat(timespec="seconds"),
            "random_state": RANDOM_STATE, "test_size": TEST_SIZE, "n_trees": N_TREES,
            "features": FEATURES, "classes": list(classes_rf),
            "report_rf": report_rf, "confusion_matrix_norm_rf": cm_rf.tolist(),
        }, f, ensure_ascii=False, indent=2)
    joblib.dump(pipe_rf, MODELO_PATH)
    checkpoint(f"Modelo RF salvo em {os.path.abspath(MODELO_PATH)}")

    # ====== Modelo de NOME (TF-IDF + LR calibrada) ======
    checkpoint("Treinando modelo de NOME (TF-IDF char + LR calibrada)...")
    df_name_train = df_train.dropna(subset=["FIRST_NAME_NORM"]).copy()
    Xn = df_name_train["FIRST_NAME_NORM"].astype(str)
    yn = df_name_train[ALVO]
    tfidf = TfidfVectorizer(analyzer="char", ngram_range=TFIDF_NGRAM)
    lr = LogisticRegression(max_iter=LR_MAX_ITER, class_weight="balanced")
    name_pipe = Pipeline([("tfidf", tfidf), ("clf", CalibratedClassifierCV(lr, method="isotonic", cv=3))])

    Xn_tr, Xn_te, yn_tr, yn_te = train_test_split(
        Xn, yn, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=yn
    )
    name_pipe.fit(Xn_tr, yn_tr)
    yn_pred = name_pipe.predict(Xn_te)
    cm_nm = confusion_matrix(yn_te, yn_pred, labels=name_pipe.classes_, normalize="true")
    report_nm = classification_report(yn_te, yn_pred, labels=name_pipe.classes_, output_dict=True, zero_division=0)
    with open(METRICS_NOME, "w", encoding="utf-8") as f:
        json.dump({
            "classes": list(name_pipe.classes_),
            "tfidf_char_ngrams": TFIDF_NGRAM,
            "report_nome": report_nm,
            "confusion_matrix_norm_nome": cm_nm.tolist()
        }, f, ensure_ascii=False, indent=2)
    joblib.dump(name_pipe, MODELO_NOME_PATH)
    checkpoint(f"Modelo de nome salvo em {os.path.abspath(MODELO_NOME_PATH)}")

    # ====== Escopo de predição ======
    checkpoint("Preparando escopo INDETERMINADO...")
    df_out = df_all.copy()
    df_out["ALVO_RAW_UP"] = df_out[ALVO].astype(str).str.strip().str.upper()
    df_out["FIRST_NAME_NORM"] = df_out[NAME_COL].apply(extract_first_name)
    mask_indet = (df_out["ALVO_RAW_UP"] == INDET_LABEL)

    # Saídas
    df_out["GENERO_PRED"]       = np.where(mask_indet, np.nan, df_out[ALVO])
    df_out["PROB_GENERO_PRED"]  = np.nan
    df_out["FOI_IMPUTADO"]      = False
    df_out["MODO_IMPUTACAO"]    = None
    df_out["CONF_FONTE"]        = None
    df_out["IMPUTACAO_FORCADA"] = False

    total_indet = int(mask_indet.sum())
    checkpoint(f"INDETERMINADO: {total_indet:,}")

    # ====== Etapa A: RF por primeiro nome (agrega features por modo) ======
    df_pred_scope = df_out.loc[mask_indet & df_out["FIRST_NAME_NORM"].notna()].copy()
    if not df_pred_scope.empty:
        agg_dict = {feat: mode_series for feat in FEATURES}
        df_first_feats = df_pred_scope.groupby("FIRST_NAME_NORM", as_index=True)[FEATURES].agg(agg_dict).reset_index()
        checkpoint(f"Primeiros nomes distintos: {len(df_first_feats):,}")

        probas_rf = pipe_rf.predict_proba(df_first_feats[FEATURES])
        cls_rf = pipe_rf.named_steps["rf"].classes_
        idx_rf = probas_rf.argmax(axis=1)
        df_first_feats["PRED_RF"] = cls_rf[idx_rf]
        df_first_feats["CONF_RF"] = probas_rf.max(axis=1)
        map_pred_rf = df_first_feats.set_index("FIRST_NAME_NORM")["PRED_RF"]
        map_conf_rf = df_first_feats.set_index("FIRST_NAME_NORM")["CONF_RF"]

        idx_scope = df_out.index[mask_indet]
        fn_scope  = df_out.loc[idx_scope, "FIRST_NAME_NORM"]
        df_out.loc[idx_scope, "GENERO_PRED_RF"]      = fn_scope.map(map_pred_rf)
        df_out.loc[idx_scope, "PROB_GENERO_PRED_RF"] = fn_scope.map(map_conf_rf)
    else:
        df_out["GENERO_PRED_RF"] = np.nan
        df_out["PROB_GENERO_PRED_RF"] = np.nan

    # ====== Etapa B: Modelo de NOME (ML) ======
    mask_name_avail = mask_indet & df_out["FIRST_NAME_NORM"].notna()
    if mask_name_avail.any():
        probas_nm = name_pipe.predict_proba(df_out.loc[mask_name_avail, "FIRST_NAME_NORM"].astype(str))
        cls_nm = name_pipe.classes_
        idx_nm = probas_nm.argmax(axis=1)
        df_out.loc[mask_name_avail, "GENERO_PRED_NM"] = cls_nm[idx_nm]
        df_out.loc[mask_name_avail, "PROB_GENERO_PRED_NM"] = probas_nm.max(axis=1)
    else:
        df_out["GENERO_PRED_NM"] = np.nan
        df_out["PROB_GENERO_PRED_NM"] = np.nan

    # ====== Etapas C/D: Bayes (nome,UF) e (nome) ======
    checkpoint("Montando tabelas Bayes...")
    df_labeled = df_train.loc[df_train["FIRST_NAME_NORM"].notna(), ["FIRST_NAME_NORM","SG_UF_PROGRAMA", ALVO]].copy()
    df_labeled["ALVO_NORM"] = df_labeled[ALVO].astype(str).str.strip().str.upper()
    df_labeled = df_labeled[df_labeled["ALVO_NORM"].isin([FEMALE_LABEL, MALE_LABEL])]

    # (nome,UF)
    nu_counts = df_labeled.groupby(["FIRST_NAME_NORM","SG_UF_PROGRAMA","ALVO_NORM"]).size().unstack(fill_value=0)
    for col in [FEMALE_LABEL, MALE_LABEL]:
        if col not in nu_counts.columns: nu_counts[col] = 0
    nu_counts["TOT"] = nu_counts[FEMALE_LABEL] + nu_counts[MALE_LABEL]
    nu_counts["P_FEM"] = (nu_counts[FEMALE_LABEL] + ALPHA_PRIOR) / (nu_counts["TOT"] + 2*ALPHA_PRIOR)
    nu_counts["P_MASC"]= (nu_counts[MALE_LABEL]  + ALPHA_PRIOR) / (nu_counts["TOT"] + 2*ALPHA_PRIOR)
    nu_counts["CLASSE_UF"] = np.where(nu_counts["P_FEM"] >= nu_counts["P_MASC"], FEMALE_LABEL, MALE_LABEL)
    nu_counts["CONF_UF"]   = nu_counts[["P_FEM","P_MASC"]].max(axis=1)

    # (nome) BR
    n_counts = df_labeled.groupby(["FIRST_NAME_NORM","ALVO_NORM"]).size().unstack(fill_value=0)
    for col in [FEMALE_LABEL, MALE_LABEL]:
        if col not in n_counts.columns: n_counts[col] = 0
    n_counts["TOT"] = n_counts[FEMALE_LABEL] + n_counts[MALE_LABEL]
    n_counts["P_FEM"] = (n_counts[FEMALE_LABEL] + ALPHA_PRIOR) / (n_counts["TOT"] + 2*ALPHA_PRIOR)
    n_counts["P_MASC"]= (n_counts[MALE_LABEL]  + ALPHA_PRIOR) / (n_counts["TOT"] + 2*ALPHA_PRIOR)
    n_counts["CLASSE_BR"] = np.where(n_counts["P_FEM"] >= n_counts["P_MASC"], FEMALE_LABEL, MALE_LABEL)
    n_counts["CONF_BR"]   = n_counts[["P_FEM","P_MASC"]].max(axis=1)

    # ====== Aplicação com ajuste adaptativo ======
    thr_rf, thr_ml, thr_uf, thr_br = THRESH_RF_INIT, THRESH_NOME_ML_INIT, THRESH_BAYES_UF_INIT, THRESH_BAYES_BR_INIT
    df_out.loc[mask_indet, ["GENERO_PRED","PROB_GENERO_PRED","FOI_IMPUTADO","MODO_IMPUTACAO","CONF_FONTE"]] = [np.nan, np.nan, False, None, None]

    def apply_once(trf, tml, tuf, tbr):
        idx_scope = df_out.index[mask_indet]

        # A) RF
        use_rf = df_out.loc[idx_scope, "PROB_GENERO_PRED_RF"].fillna(0.0) >= trf
        i_rf = df_out.loc[idx_scope].index[use_rf]
        df_out.loc[i_rf, "GENERO_PRED"]      = df_out.loc[i_rf, "GENERO_PRED_RF"]
        df_out.loc[i_rf, "PROB_GENERO_PRED"] = df_out.loc[i_rf, "PROB_GENERO_PRED_RF"]
        df_out.loc[i_rf, "FOI_IMPUTADO"]     = True
        df_out.loc[i_rf, "MODO_IMPUTACAO"]   = "RF"
        df_out.loc[i_rf, "CONF_FONTE"]       = "RF"

        # B) NOME_ML
        remain = df_out.loc[idx_scope & ~df_out["FOI_IMPUTADO"]].index
        prob_ml = df_out.loc[remain, "PROB_GENERO_PRED_NM"].fillna(0.0)
        use_ml = prob_ml >= tml
        i_ml = remain[use_ml]
        df_out.loc[i_ml, "GENERO_PRED"]      = df_out.loc[i_ml, "GENERO_PRED_NM"]
        df_out.loc[i_ml, "PROB_GENERO_PRED"] = df_out.loc[i_ml, "PROB_GENERO_PRED_NM"]
        df_out.loc[i_ml, "FOI_IMPUTADO"]     = True
        df_out.loc[i_ml, "MODO_IMPUTACAO"]   = "NOME_ML"
        df_out.loc[i_ml, "CONF_FONTE"]       = "ML"

        # C) NOME_UF (Bayes)
        remain = df_out.loc[idx_scope & ~df_out["FOI_IMPUTADO"]].index
        if len(remain) > 0:
            tmp = df_out.loc[remain, ["FIRST_NAME_NORM","SG_UF_PROGRAMA"]]
            nu = nu_counts[nu_counts["TOT"] >= MIN_SUP_UF].reset_index().set_index(["FIRST_NAME_NORM","SG_UF_PROGRAMA"])
            if not nu.empty:
                conf_list, cls_list, idx_list = [], [], []
                for idx, row in tmp.iterrows():
                    key = (row["FIRST_NAME_NORM"], row["SG_UF_PROGRAMA"])
                    if key in nu.index:
                        conf = float(nu.loc[key, "CONF_UF"])
                        if conf >= tuf:
                            conf_list.append(conf)
                            cls_list.append(nu.loc[key, "CLASSE_UF"])
                            idx_list.append(idx)
                if idx_list:
                    i_uf = pd.Index(idx_list)
                    df_out.loc[i_uf, "GENERO_PRED"]      = cls_list
                    df_out.loc[i_uf, "PROB_GENERO_PRED"] = conf_list
                    df_out.loc[i_uf, "FOI_IMPUTADO"]     = True
                    df_out.loc[i_uf, "MODO_IMPUTACAO"]   = "NOME_UF"
                    df_out.loc[i_uf, "CONF_FONTE"]       = "BAYES_UF"

        # D) NOME_BR (Bayes nacional)
        remain = df_out.loc[idx_scope & ~df_out["FOI_IMPUTADO"]].index
        if len(remain) > 0:
            tmp = df_out.loc[remain, ["FIRST_NAME_NORM"]]
            nb = n_counts[n_counts["TOT"] >= MIN_SUP_BR]
            if not nb.empty:
                conf_map = nb["CONF_BR"]
                class_map= nb["CLASSE_BR"]
                conf_series = tmp["FIRST_NAME_NORM"].map(conf_map)
                class_series= tmp["FIRST_NAME_NORM"].map(class_map)
                use_br = conf_series.fillna(0.0) >= tbr
                i_br = tmp.index[use_br]
                df_out.loc[i_br, "GENERO_PRED"]      = class_series.loc[i_br].values
                df_out.loc[i_br, "PROB_GENERO_PRED"] = conf_series.loc[i_br].values
                df_out.loc[i_br, "FOI_IMPUTADO"]     = True
                df_out.loc[i_br, "MODO_IMPUTACAO"]   = "NOME_BR"
                df_out.loc[i_br, "CONF_FONTE"]       = "BAYES_BR"

        cov, n_ok = compute_coverage(df_out, mask_indet)
        return cov, n_ok

    checkpoint(f"Limiar inicial: RF={THRESH_RF_INIT}, ML={THRESH_NOME_ML_INIT}, UF={THRESH_BAYES_UF_INIT}, BR={THRESH_BAYES_BR_INIT}")
    cov, n_ok = apply_once(THRESH_RF_INIT, THRESH_NOME_ML_INIT, THRESH_BAYES_UF_INIT, THRESH_BAYES_BR_INIT)
    checkpoint(f"Cobertura inicial: {cov:.3%} ({n_ok:,}/{total_indet:,})")

    guard = 0
    thr_rf, thr_ml, thr_uf, thr_br = THRESH_RF_INIT, THRESH_NOME_ML_INIT, THRESH_BAYES_UF_INIT, THRESH_BAYES_BR_INIT
    while cov < TARGET_COVERAGE and guard < 20:
        if thr_br > MIN_CONF_FLOOR:   thr_br = max(MIN_CONF_FLOOR, thr_br - ADAPT_STEP)
        elif thr_uf > MIN_CONF_FLOOR: thr_uf = max(MIN_CONF_FLOOR, thr_uf - ADAPT_STEP)
        elif thr_ml > MIN_CONF_FLOOR: thr_ml = max(MIN_CONF_FLOOR, thr_ml - ADAPT_STEP)
        elif thr_rf > MIN_CONF_FLOOR: thr_rf = max(MIN_CONF_FLOOR, thr_rf - ADAPT_STEP)
        else: break

        # reset apenas para o escopo indeterminado
        df_out.loc[mask_indet, ["GENERO_PRED","PROB_GENERO_PRED","FOI_IMPUTADO","MODO_IMPUTACAO","CONF_FONTE"]] = [np.nan, np.nan, False, None, None]
        cov, n_ok = apply_once(thr_rf, thr_ml, thr_uf, thr_br)
        checkpoint(f"Ajuste {guard+1:02d} => RF={thr_rf:.2f}, ML={thr_ml:.2f}, UF={thr_uf:.2f}, BR={thr_br:.2f} | Cobertura: {cov:.3%} ({n_ok:,}/{total_indet:,})")
        guard += 1

    # ====== Fallback 100% ======
    checkpoint("Aplicando Fallback (100%)...")
    remain_mask = (mask_indet & ~df_out["FOI_IMPUTADO"])
    remain_idx  = df_out.index[remain_mask]
    qtd_forcados = 0
    if len(remain_idx) > 0:
        # classe majoritária global (treino)
        major_class = df_train[ALVO].value_counts().idxmax()
        prob_nome_remain = df_out.loc[remain_idx, "PROB_GENERO_PRED_NM"]
        pred_nome_remain = df_out.loc[remain_idx, "GENERO_PRED_NM"]
        fallback_class = np.where(pred_nome_remain.notna(), pred_nome_remain, major_class)
        fallback_conf  = np.where(prob_nome_remain.notna(), prob_nome_remain.astype(float), 0.51)
        df_out.loc[remain_idx, "GENERO_PRED"]        = fallback_class
        df_out.loc[remain_idx, "PROB_GENERO_PRED"]   = fallback_conf
        df_out.loc[remain_idx, "FOI_IMPUTADO"]       = True
        df_out.loc[remain_idx, "MODO_IMPUTACAO"]     = "FALLBACK"
        df_out.loc[remain_idx, "CONF_FONTE"]         = "FALLBACK"
        df_out.loc[remain_idx, "IMPUTACAO_FORCADA"]  = True
        qtd_forcados = len(remain_idx)

    qtd_imput = int((mask_indet & df_out["FOI_IMPUTADO"]).sum())
    total_indet = int(mask_indet.sum())
    pct_forcados = (qtd_forcados / total_indet) if total_indet else 0.0
    cov_final = (qtd_imput / total_indet) if total_indet else 1.0
    checkpoint(f"Cobertura final: {cov_final:.3%} ({qtd_imput:,}/{total_indet:,}) | Forçados: {qtd_forcados:,} ({pct_forcados:.2%})")

    # ====== Atualização no banco ======
    with engine.begin() as db:
        checkpoint("Garantindo LOG e colunas...")

        # LOG table
        db.execute(text(f"""
        IF OBJECT_ID(N'[{SCHEMA}].[LOG_IMPUTACAO_GENERO]') IS NULL
        BEGIN
          CREATE TABLE [{SCHEMA}].[LOG_IMPUTACAO_GENERO] (
            ID INT IDENTITY(1,1) PRIMARY KEY,
            DT_IMPUTACAO DATETIME2 NOT NULL DEFAULT SYSDATETIME(),
            THRESHOLD_RF FLOAT NULL,
            THRESHOLD_NOME_ML FLOAT NULL,
            THRESHOLD_BAYES_UF FLOAT NULL,
            THRESHOLD_BAYES_BR FLOAT NULL,
            TARGET_COVERAGE FLOAT NULL,
            COVERAGE_FINAL FLOAT NULL,
            QTD_NULOS INT NOT NULL,
            QTD_IMPUTADOS INT NOT NULL,
            QTD_FORCADOS INT NOT NULL,
            PCT_FORCADOS FLOAT NULL,
            MODELO_PATH NVARCHAR(260) NULL,
            MODELO_NOME_PATH NVARCHAR(260) NULL,
            METRICS_PATH NVARCHAR(260) NULL,
            METRICS_NOME NVARCHAR(260) NULL
          );
        END
        """))
        # adiciona colunas caso tabela já exista (idempotente)
        for add_col, add_type in [
            ("QTD_FORCADOS","INT"), ("PCT_FORCADOS","FLOAT"),
            ("THRESHOLD_RF","FLOAT"), ("THRESHOLD_NOME_ML","FLOAT"),
            ("THRESHOLD_BAYES_UF","FLOAT"), ("THRESHOLD_BAYES_BR","FLOAT"),
            ("TARGET_COVERAGE","FLOAT"), ("COVERAGE_FINAL","FLOAT"),
            ("MODELO_PATH","NVARCHAR(260)"), ("MODELO_NOME_PATH","NVARCHAR(260)"),
            ("METRICS_PATH","NVARCHAR(260)"), ("METRICS_NOME","NVARCHAR(260)"),
        ]:
            db.execute(text(f"""
                IF COL_LENGTH(N'[{SCHEMA}].[LOG_IMPUTACAO_GENERO]', '{add_col}') IS NULL
                BEGIN
                    ALTER TABLE [{SCHEMA}].[LOG_IMPUTACAO_GENERO] ADD [{add_col}] {add_type} NULL;
                END
            """))
        # colunas na tabela alvo
        for col_name, sql_type in [
            ("GENERO_PRED","NVARCHAR(20)"),
            ("PROB_GENERO_PRED","FLOAT"),
            ("FOI_IMPUTADO","BIT"),
            ("MODO_IMPUTACAO","NVARCHAR(16)"),
            ("CONF_FONTE","NVARCHAR(16)"),
            ("IMPUTACAO_FORCADA","BIT"),
        ]:
            db.execute(text(f"""
                IF COL_LENGTH(N'[{SCHEMA}].[{TABELA_ORIG}]', '{col_name}') IS NULL
                BEGIN
                    ALTER TABLE [{SCHEMA}].[{TABELA_ORIG}] ADD [{col_name}] {sql_type} NULL;
                END
            """))

        # staging
        staging_name = f"{TABELA_ORIG}_STAGING_{uuid.uuid4().hex[:8]}"
        df_stage = df_out.loc[:, [
            KEY_COL,"GENERO_PRED","PROB_GENERO_PRED","FOI_IMPUTADO",
            "MODO_IMPUTACAO","CONF_FONTE","IMPUTACAO_FORCADA"
        ]].copy()
        df_stage["FOI_IMPUTADO"]      = df_stage["FOI_IMPUTADO"].astype(int)
        df_stage["IMPUTACAO_FORCADA"] = df_stage["IMPUTACAO_FORCADA"].astype(int)

        checkpoint(f"Carregando staging [{SCHEMA}].[{staging_name}]...")
        df_stage.to_sql(
            name=staging_name, con=db, schema=SCHEMA,
            if_exists="replace", index=False,
            chunksize=CHUNK_STAGE, method=None, dtype=STAGING_DTYPES
        )
        checkpoint(f"Staging concluída ({len(df_stage):,} linhas).")

        checkpoint("Executando UPDATE com JOIN...")
        db.execute(text(f"""
            UPDATE tgt
            SET
                tgt.[GENERO_PRED]       = src.[GENERO_PRED],
                tgt.[PROB_GENERO_PRED]  = src.[PROB_GENERO_PRED],
                tgt.[FOI_IMPUTADO]      = src.[FOI_IMPUTADO],
                tgt.[MODO_IMPUTACAO]    = src.[MODO_IMPUTACAO],
                tgt.[CONF_FONTE]        = src.[CONF_FONTE],
                tgt.[IMPUTACAO_FORCADA] = src.[IMPUTACAO_FORCADA]
            FROM [{SCHEMA}].[{TABELA_ORIG}] AS tgt
            INNER JOIN [{SCHEMA}].[{staging_name}] AS src
                ON tgt.[{KEY_COL}] = src.[{KEY_COL}];
        """))
        db.execute(text(f"DROP TABLE [{SCHEMA}].[{staging_name}];"))

        checkpoint("Registrando LOG...")
        db.execute(
            text(f"""
                INSERT INTO [{SCHEMA}].[LOG_IMPUTACAO_GENERO]
                    (THRESHOLD_RF, THRESHOLD_NOME_ML, THRESHOLD_BAYES_UF, THRESHOLD_BAYES_BR,
                     TARGET_COVERAGE, COVERAGE_FINAL,
                     QTD_NULOS, QTD_IMPUTADOS, QTD_FORCADOS, PCT_FORCADOS,
                     MODELO_PATH, MODELO_NOME_PATH, METRICS_PATH, METRICS_NOME)
                VALUES (:trf, :tml, :tuf, :tbr,
                        :tgt, :cov,
                        :nulos, :imp, :forc, :pctf,
                        :mpath, :npath, :kpath, :kname);
            """),
            {"trf": float(thr_rf), "tml": float(thr_ml), "tuf": float(thr_uf), "tbr": float(thr_br),
             "tgt": float(TARGET_COVERAGE), "cov": float(cov_final),
             "nulos": int(total_indet), "imp": int(qtd_imput),
             "forc": int(qtd_forcados), "pctf": float(pct_forcados),
             "mpath": os.path.abspath(MODELO_PATH), "npath": os.path.abspath(MODELO_NOME_PATH),
             "kpath": os.path.abspath(METRICS_PATH), "kname": os.path.abspath(METRICS_NOME)}
        )

    print(f"\n✅ Atualização concluída.")
    print(f"INDETERMINADO: {total_indet:,} | Imputados: {qtd_imput:,} (cobertura {cov_final:.1%})")
    print(f"Forçados (fallback): {qtd_forcados:,} ({pct_forcados:.1%} dos INDETERMINADO)")
    print("Colunas gravadas: GENERO_PRED, PROB_GENERO_PRED, FOI_IMPUTADO, MODO_IMPUTACAO, CONF_FONTE, IMPUTACAO_FORCADA")
    print(f"Modelos: RF={os.path.abspath(MODELO_PATH)} | NOME={os.path.abspath(MODELO_NOME_PATH)}")
    print(f"Métricas: RF={os.path.abspath(METRICS_PATH)} | NOME={os.path.abspath(METRICS_NOME)}")
    checkpoint("==== Fim da execução (sucesso) ====")

except Exception as e:
    print("\n[ERRO]", e)
    checkpoint(f"[ERRO] {repr(e)}")
    checkpoint("==== Fim da execução (falha) ====")
finally:
    try: engine.dispose()
    except Exception: pass
