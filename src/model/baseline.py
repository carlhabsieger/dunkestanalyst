"""Baseline : prédire le score Dunkest par moyenne glissante.
Backtest sans fuite : la prédiction d'un match n'utilise que les matchs précédents.
"""
import glob
import numpy as np
import pandas as pd

MIN_PRIOR_GAMES = 5       # historique minimal avant de prédire
MIN_PRIOR_MINUTES = 15.0  # filtre basé sur le passé (pas sur le match prédit)
WINDOWS = (3, 5, 10)


def load_all(pattern="data/player_logs*.csv"):
    files = sorted(glob.glob(pattern))
    if not files:
        raise SystemExit("Aucun CSV dans data/ : lance d'abord src.data.box_scores")
    df = pd.concat(
        [pd.read_csv(f, parse_dates=["GAME_DATE"]) for f in files],
        ignore_index=True,
    )
    df["MIN"] = pd.to_numeric(df["MIN"], errors="coerce")
    df = df.drop_duplicates(subset=["PLAYER_ID", "GAME_ID"])
    return df.sort_values(["PLAYER_ID", "GAME_DATE"]).reset_index(drop=True)


def add_predictions(df):
    g = df.groupby("PLAYER_ID")
    df["n_prior"] = g.cumcount()
    df["prior_min"] = g["MIN"].transform(
        lambda s: s.shift(1).rolling(10, min_periods=1).mean()
    )
    df["moy_cumulee"] = g["DUNKEST"].transform(
        lambda s: s.shift(1).expanding().mean()
    )
    preds = ["moy_cumulee"]
    for n in WINDOWS:
        df[f"moy_{n}"] = g["DUNKEST"].transform(
            lambda s, n=n: s.shift(1).rolling(n, min_periods=1).mean()
        )
        df[f"ewm_{n}"] = g["DUNKEST"].transform(
            lambda s, n=n: s.shift(1).ewm(span=n, adjust=False).mean()
        )
        preds += [f"moy_{n}", f"ewm_{n}"]
    return df, preds


def evaluate(df, preds):
    mask = (df["n_prior"] >= MIN_PRIOR_GAMES) & (df["prior_min"] >= MIN_PRIOR_MINUTES)
    y = df.loc[mask, "DUNKEST"]
    rows = []
    for p in preds:
        err = df.loc[mask, p] - y
        rows.append({
            "modele": p,
            "MAE": err.abs().mean(),
            "RMSE": np.sqrt((err ** 2).mean()),
            "biais": err.mean(),
        })
    ref = (y - y.mean()).abs().mean()
    return pd.DataFrame(rows).sort_values("MAE").round(2), int(mask.sum()), y.std(), ref


if __name__ == "__main__":
    df = load_all()
    df, preds = add_predictions(df)
    table, n, std, ref = evaluate(df, preds)
    print(f"{n} prédictions évaluées")
    print(f"Écart-type du score réel : {std:.2f}")
    print(f"MAE d'une constante (référence, légèrement optimiste) : {ref:.2f}")
    print(table.to_string(index=False))
