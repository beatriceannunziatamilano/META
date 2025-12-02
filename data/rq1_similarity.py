#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
rq1_similarity.py

Script unico in Python che:
- legge i file PSY_* e SUD.xlsx da data/raw
- calcola Spearman, Cosine e Euclidean (come similitudine) tra
  ogni mappa PSY e ogni mappa SUD
- converte ogni matrice in Z-score
- calcola un combined Z (Stouffer) per cortex e subcortex
- salva le matrici Z_* e i ranking RANK_* in ALL_outputs_RQ1

Struttura attesa dei file (tutti in data/raw):
- PSY_adults.xlsx
- PSY_adults_ctx.xlsx
- PSY_adolescents.xlsx
- PSY_adolescents_ctx.xlsx
- SUD.xlsx

Assunzioni:
- Le righe sono regioni (stessa ordering in PSY e SUD).
- Le colonne sono disturbi / sottotipi.
- Le prime 68 righe = corteccia; le eventuali righe in più = subcorteccia.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


# -------------------------------------------------------------------------
# PATH DI BASE (repo root = cartella che contiene "data")
# -------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "data" / "raw"
OUT_DIR = ROOT / "ALL_outputs_RQ1"
OUT_DIR.mkdir(exist_ok=True)


# -------------------------------------------------------------------------
# HELPER: caricamento e conversione tabelle
# -------------------------------------------------------------------------
def safe_table_to_numeric(df: pd.DataFrame) -> np.ndarray:
    """
    Converte un DataFrame misto (stringhe/numeri) in matrice float32,
    con NaN per i valori non convertibili.
    """
    numeric_df = df.apply(pd.to_numeric, errors="coerce")
    return numeric_df.to_numpy(dtype=float)


def zscore_matrix(mat: np.ndarray) -> np.ndarray:
    """
    Z-score across all entries di una matrice 2D.
    Ignora NaN.
    """
    m = np.nanmean(mat)
    s = np.nanstd(mat, ddof=1)
    if s == 0 or np.isnan(s):
        # evita divisione per zero: ritorna z=0
        return np.zeros_like(mat, dtype=float)
    return (mat - m) / s


# -------------------------------------------------------------------------
# METRICHE DI SIMILITUDINE
# -------------------------------------------------------------------------
def spearman_similarity(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman correlation tra due vettori (ignora NaN)."""
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2:
        return np.nan
    r, _ = stats.spearmanr(x[mask], y[mask])
    return float(r)


def cosine_similarity(x: np.ndarray, y: np.ndarray) -> float:
    """Cosine similarity tra due vettori (ignora NaN)."""
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() == 0:
        return np.nan
    x_m = x[mask]
    y_m = y[mask]
    denom = np.linalg.norm(x_m) * np.linalg.norm(y_m)
    if denom == 0:
        return np.nan
    return float(np.dot(x_m, y_m) / denom)


def euclidean_similarity(x: np.ndarray, y: np.ndarray) -> float:
    """
    Euclidean distance convertita in similitudine (con segno meno).
    D = ||x-y||_2 ; similarity = -D (più grande = più simile).
    """
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() == 0:
        return np.nan
    d = np.linalg.norm(x[mask] - y[mask])
    return float(-d)


# -------------------------------------------------------------------------
# FUNZIONE PRINCIPALE PER UN GRUPPO
# -------------------------------------------------------------------------
def process_group(group_label: str, psy_filename: str, n_cortex: int = 68) -> None:
    """
    Calcola le matrici Z_* per un gruppo (adults/adolescents, all/ctx).

    Parameters
    ----------
    group_label : str
        Nome cartella output, es. "adults_all".
    psy_filename : str
        Nome file Excel in data/raw, es. "PSY_adults.xlsx".
    n_cortex : int
        Numero di regioni corticale (default 68).
    """
    print(f"\n=== Group: {group_label} ===")

    # --- PATH E CARICAMENTO ---
    psy_path = RAW_DIR / psy_filename
    sud_path = RAW_DIR / "SUD.xlsx"

    if not psy_path.is_file():
        raise FileNotFoundError(f"{psy_path} non trovato")
    if not sud_path.is_file():
        raise FileNotFoundError(f"{sud_path} non trovato")

    # carica PSY e SUD (tutte le colonne numeriche)
    df_psy = pd.read_excel(psy_path)
    df_sud = pd.read_excel(sud_path)

    X_psy = safe_table_to_numeric(df_psy)  # [regioni x N_psy]
    X_sud = safe_table_to_numeric(df_sud)  # [regioni x N_sud]

    psy_names = list(df_psy.columns)
    sud_names = list(df_sud.columns)

    n_regions = min(X_psy.shape[0], X_sud.shape[0])
    X_psy = X_psy[:n_regions, :]
    X_sud = X_sud[:n_regions, :]

    print(f"  PSY: {psy_path.name} ({n_regions} regioni x {X_psy.shape[1]} disturbi)")
    print(f"  SUD: SUD.xlsx ({n_regions} regioni x {X_sud.shape[1]} sostanze)")

    # --- CORTEX / SUBCTX INDEX ---
    cortex_idx = np.arange(0, min(n_cortex, n_regions))
    if n_regions > n_cortex:
        subctx_idx = np.arange(n_cortex, n_regions)
    else:
        subctx_idx = np.array([], dtype=int)

    print(f"  Cortex rows: {len(cortex_idx)}; Subcortex rows: {len(subctx_idx)}")

    # --- OUTPUT DIR PER IL GRUPPO ---
    out_group = OUT_DIR / group_label
    out_group.mkdir(parents=True, exist_ok=True)

    # --- MATRICI METRICHE (cortex / subctx) ---
    n_psy = X_psy.shape[1]
    n_sud = X_sud.shape[1]

    def _init_matrices():
        return (
            np.full((n_psy, n_sud), np.nan, dtype=float),  # spearman
            np.full((n_psy, n_sud), np.nan, dtype=float),  # cosine
            np.full((n_psy, n_sud), np.nan, dtype=float),  # euclidean(sim)
        )

    spearman_cortex, cosine_cortex, euclid_cortex = _init_matrices()
    spearman_sub, cosine_sub, euclid_sub = _init_matrices() if subctx_idx.size else (None, None, None)

    # --- LOOP PRINCIPALE PSY x SUD ---
    for i in range(n_psy):
        psy_vec = X_psy[:, i]
        for j in range(n_sud):
            sud_vec = X_sud[:, j]

            # CORTEX
            x_c = psy_vec[cortex_idx]
            y_c = sud_vec[cortex_idx]
            spearman_cortex[i, j] = spearman_similarity(x_c, y_c)
            cosine_cortex[i, j] = cosine_similarity(x_c, y_c)
            euclid_cortex[i, j] = euclidean_similarity(x_c, y_c)

            # SUBCORTEX (se presente)
            if subctx_idx.size:
                x_s = psy_vec[subctx_idx]
                y_s = sud_vec[subctx_idx]
                spearman_sub[i, j] = spearman_similarity(x_s, y_s)
                cosine_sub[i, j] = cosine_similarity(x_s, y_s)
                euclid_sub[i, j] = euclidean_similarity(x_s, y_s)

    # --- Z-SCORE E COMBINED Z ---
    def build_Z_and_save(region_label: str,
                         spearman_mat: np.ndarray,
                         cosine_mat: np.ndarray,
                         euclid_mat: np.ndarray) -> np.ndarray:
        """Crea Z_* e combined, salva come CSV, ritorna Z_combined."""
        if spearman_mat is None:
            return None

        Z_spear = zscore_matrix(spearman_mat)
        Z_cos = zscore_matrix(cosine_mat)
        Z_euc = zscore_matrix(euclid_mat)

        Z_combined = (Z_spear + Z_cos + Z_euc) / math.sqrt(3.0)

        def _save(mat: np.ndarray, suffix: str):
            df = pd.DataFrame(mat, index=psy_names, columns=sud_names)
            df.index.name = "Psychiatric_Disorder"
            out_file = out_group / f"Z_{region_label}_{suffix}.csv"
            df.to_csv(out_file)
            print(f"  -> {out_file}")

        _save(Z_spear, "spearman")
        _save(Z_cos, "cosine")
        _save(Z_euc, "euclidean")
        _save(Z_combined, "combined")

        return Z_combined

    print("  Calcolo Z-score e salvataggio tabelle...")

    Zc = build_Z_and_save("cortex", spearman_cortex, cosine_cortex, euclid_cortex)
    Zs = None
    if subctx_idx.size:
        Zs = build_Z_and_save("subctx", spearman_sub, cosine_sub, euclid_sub)

    # --- RANKING RQ1.2 (usa combined Z, cortex + subcortex se presente) ---
    print("  Calcolo ranking RANK_overall_* ...")

    if Zs is not None:
        Ztot = (Zc + Zs) / math.sqrt(2.0)
    else:
        Ztot = Zc

    # rank per ogni sostanza
    for j, sud_name in enumerate(sud_names):
        z_col = Ztot[:, j]
        order = np.argsort(-z_col)  # decrescente
        df_rank = pd.DataFrame({
            "Psychiatric_Disorder": np.array(psy_names)[order],
            "Z_Stouffer": z_col[order],
        })
        out_rank = out_group / f"RANK_overall_by_{sud_name}.csv"
        df_rank.to_csv(out_rank, index=False)
        print(f"  -> {out_rank}")

    # rank su media across SUD
    Zmean = np.nanmean(Ztot, axis=1)
    order_mean = np.argsort(-Zmean)
    df_mean = pd.DataFrame({
        "Psychiatric_Disorder": np.array(psy_names)[order_mean],
        "Mean_Z_over_SUD": Zmean[order_mean],
    })
    out_rank_mean = out_group / "RANK_overall_mean_across_SUD.csv"
    df_mean.to_csv(out_rank_mean, index=False)
    print(f"  -> {out_rank_mean}")

    print(f"=== Group {group_label} DONE ===")


# -------------------------------------------------------------------------
# ENTRY-POINT GENERALE
# -------------------------------------------------------------------------
def main():
    """
    Esegue l'analisi per i 4 gruppi:

    - adults_all        -> PSY_adults.xlsx
    - adults_ctx        -> PSY_adults_ctx.xlsx
    - adolescents_all   -> PSY_adolescents.xlsx
    - adolescents_ctx   -> PSY_adolescents_ctx.xlsx
    """
    print(f"ROOT: {ROOT}")
    print(f"RAW_DIR: {RAW_DIR}")
    print(f"OUT_DIR: {OUT_DIR}")

    groups = [
        ("adults_all", "PSY_adults.xlsx"),
        ("adults_ctx", "PSY_adults_ctx.xlsx"),
        ("adolescents_all", "PSY_adolescents.xlsx"),
        ("adolescents_ctx", "PSY_adolescents_ctx.xlsx"),
    ]

    for label, fname in groups:
        process_group(label, fname, n_cortex=68)

    print("\n✅ Tutte le analisi completate.")


if __name__ == "__main__":
    main()
