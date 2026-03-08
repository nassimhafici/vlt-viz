"""
Test console: chargement de SVIX / ^SHORTVOL et backfill
"""

import pandas as pd
from core.db import load_multi_prices


def _proxy_series_backward(base, proxy):
    """Extend SVIX history backward using ^SHORTVOL returns."""
    base, proxy = base.align(proxy, join="inner")
    rets_proxy = proxy.pct_change()
    first_idx = base.first_valid_index()
    if first_idx is None:
        return proxy.copy()
    start_val = base.loc[first_idx]
    backward_rets = rets_proxy.loc[:first_idx].iloc[:-1]
    if backward_rets.empty:
        return base
    reconstructed = start_val / (1 + backward_rets[::-1]).cumprod()[::-1]
    return reconstructed.combine_first(base)


def main():
    symbols = ["SVIX", "^SHORTVOL", "^VIX", "^VIX3M"]
    px = load_multi_prices(symbols)

    if px is None or px.empty:
        print("Aucune donnée renvoyée par load_multi_prices pour", symbols)
        return

    px.index = pd.to_datetime(px.index)
    px = px.sort_index()

    print("Colonnes disponibles :", list(px.columns))
    print(f"Nombre total de lignes : {len(px)}")

    # Vérifier présence des colonnes
    for sym in symbols:
        if sym in px.columns:
            n_valid = px[sym].notna().sum()
            print(f"{sym}: {n_valid} valeurs non nulles")
        else:
            print(f"{sym}: **ABSENT** des colonnes")

    # Tester le backfill SVIX avec SHORTVOL
    if "SVIX" in px.columns and "^SHORTVOL" in px.columns:
        svix_raw = px["SVIX"].dropna()
        shortvol_raw = px["^SHORTVOL"].dropna()

        print("\n--- Infos brutes ---")
        print("SVIX: dates min/max :", svix_raw.index.min(), "→", svix_raw.index.max())
        print("^SHORTVOL: dates min/max :", shortvol_raw.index.min(), "→", shortvol_raw.index.max())

        svix_proxy = _proxy_series_backward(svix_raw, shortvol_raw)

        print("\n--- Après backfill ---")
        print("SVIX_proxy: dates min/max :", svix_proxy.index.min(), "→", svix_proxy.index.max())
        print("SVIX_proxy: nb valeurs non nulles :", svix_proxy.notna().sum())

        # Vérifier que les valeurs sur la partie originale SVIX sont identiques
        common_idx = svix_raw.index.intersection(svix_proxy.index)
        diff = (svix_proxy.loc[common_idx] - svix_raw.loc[common_idx]).abs().max()
        print("Différence max sur la zone où SVIX existe déjà :", diff)

        # Aperçu des premières/dernières lignes
        print("\nHead SVIX_proxy:")
        print(svix_proxy.head(10))
        print("\nTail SVIX_proxy:")
        print(svix_proxy.tail(10))
    else:
        print("\nImpossible de tester le backfill: SVIX ou ^SHORTVOL manquant.")

    # Petit test sur les données VIX / VIX3M si tu veux vérifier le signal carry
    if "^VIX" in px.columns and "^VIX3M" in px.columns:
        vix_spot = px["^VIX"].dropna()
        vix3m = px["^VIX3M"].dropna()
        aligned = vix_spot.align(vix3m, join="inner")
        vix_spot, vix3m = aligned

        svix_signal = (vix_spot < vix3m).astype(int)
        print("\nSignal carry (VIX < VIX3M):")
        print("Nb de points :", len(svix_signal))
        print("Proportion de 1 :", svix_signal.mean())
        print("Aperçu:")
        print(svix_signal.tail(10))


if __name__ == "__main__":
    main()
