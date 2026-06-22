"""Check linearity of price-duration relationship and whether
a quadratic specification changes the gender slope finding."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
import pandas as pd
import numpy as np
import unicodedata
from utils import filter_words, ppp_per_usd
import reverse_geocoder as rg
from statsmodels.formula.api import ols

def normalize(text):
    if not isinstance(text, str):
        return ''
    text = text.lower()
    text = unicodedata.normalize('NFKD', text)
    text = ''.join(c for c in text if not unicodedata.combining(c))
    return text

filter_words_n = list(set([normalize(word) for word in filter_words]))

# Load and preprocess
df = pd.read_csv(ROOT / 'data' / 'snapshots' / 'treatwell_without_raw-all-2025-06-02.csv', low_memory=False)
coords = df[['lat', 'lon']].dropna()
results = rg.search(list(coords.itertuples(index=False, name=None)), mode=1, verbose=False)
geo_df = pd.DataFrame(results, index=coords.index)[['cc']]
geo_df.columns = ['country']
df['country'] = geo_df['country']

df['duration'] = (df['simpleCutDurationMin'] + df['simpleCutDurationMax']) / 2
df = df[(df['duration'] > 10) & (df['duration'] < 120)]
df = df[~df['country'].isin(['LU', 'SM'])]
df = df[df['simpleCutSalePrice'] < 200]
df = df[df['name'] != "Test Salon TEST PURPOSE'S ONLY"]

USD_TO_EUR = ppp_per_usd['DE']
df['price'] = df.apply(
    lambda row: row['simpleCutSalePrice'] / ppp_per_usd.get(row['country'], 1.0) * USD_TO_EUR,
    axis=1
)

df = df[df['simpleCutName'].apply(lambda x: not any(word in normalize(str(x)) for word in filter_words_n))]
df = df[(df['is_male'] == True) | (df['is_female'] == True)]
df['is_female_bool'] = df['is_female'] == True
df['duration_sq'] = df['duration'] ** 2

print(f"Sample: {len(df)} obs, {df['id'].nunique()} salons")
print(f"Women mean duration: {df.loc[df['is_female_bool'], 'duration'].mean():.1f}")
print(f"Men mean duration: {df.loc[~df['is_female_bool'], 'duration'].mean():.1f}")

# Duration distribution by gender
print("\n=== Duration distribution by gender ===")
for gender, label in [(True, 'Women'), (False, 'Men')]:
    d = df.loc[df['is_female_bool'] == gender, 'duration']
    print(f"{label}: p10={d.quantile(.1):.0f}, p25={d.quantile(.25):.0f}, "
          f"p50={d.quantile(.5):.0f}, p75={d.quantile(.75):.0f}, p90={d.quantile(.9):.0f}")

# --- Model 1: Linear (baseline, replicating paper) ---
# Using entity-demeaned OLS as proxy for FE (C(id) too large)
# Instead, demean within salon
print("\n=== Demeaning within salon ===")
salon_means = df.groupby('id')[['price', 'duration', 'duration_sq']].transform('mean')
df['price_dm'] = df['price'] - salon_means['price']
df['duration_dm'] = df['duration'] - salon_means['duration']
df['duration_sq_dm'] = df['duration_sq'] - salon_means['duration_sq']

# Female is between-group within salon, so we need the interaction demeaned
df['fem_dur'] = df['is_female_bool'].astype(float) * df['duration']
df['fem_dur_sq'] = df['is_female_bool'].astype(float) * df['duration_sq']
salon_means2 = df.groupby('id')[['fem_dur', 'fem_dur_sq']].transform('mean')
df['fem_dur_dm'] = df['fem_dur'] - salon_means2['fem_dur']
df['fem_dur_sq_dm'] = df['fem_dur_sq'] - salon_means2['fem_dur_sq']

# Female indicator demeaned
fem_mean = df.groupby('id')['is_female_bool'].transform('mean')
df['female_dm'] = df['is_female_bool'].astype(float) - fem_mean

# Model 1: Linear
m1 = ols('price_dm ~ female_dm + duration_dm + fem_dur_dm - 1', data=df).fit(
    cov_type='cluster', cov_kwds={'groups': df['id']})
print("\n=== Model 1: Linear (paper specification) ===")
print(f"female:        {m1.params['female_dm']:.3f} (SE={m1.bse['female_dm']:.3f})")
print(f"duration:      {m1.params['duration_dm']:.3f} (SE={m1.bse['duration_dm']:.3f})")
print(f"female×dur:    {m1.params['fem_dur_dm']:.3f} (SE={m1.bse['fem_dur_dm']:.3f})")

# Model 2: Add quadratic duration
m2 = ols('price_dm ~ female_dm + duration_dm + duration_sq_dm + fem_dur_dm - 1', data=df).fit(
    cov_type='cluster', cov_kwds={'groups': df['id']})
print("\n=== Model 2: + Duration² ===")
print(f"female:        {m2.params['female_dm']:.3f} (SE={m2.bse['female_dm']:.3f})")
print(f"duration:      {m2.params['duration_dm']:.3f} (SE={m2.bse['duration_dm']:.3f})")
print(f"duration²:     {m2.params['duration_sq_dm']:.4f} (SE={m2.bse['duration_sq_dm']:.4f})")
print(f"female×dur:    {m2.params['fem_dur_dm']:.3f} (SE={m2.bse['fem_dur_dm']:.3f})")

# Model 3: Add quadratic + female×duration²
m3 = ols('price_dm ~ female_dm + duration_dm + duration_sq_dm + fem_dur_dm + fem_dur_sq_dm - 1', data=df).fit(
    cov_type='cluster', cov_kwds={'groups': df['id']})
print("\n=== Model 3: + Duration² + Female×Duration² ===")
print(f"female:        {m3.params['female_dm']:.3f} (SE={m3.bse['female_dm']:.3f})")
print(f"duration:      {m3.params['duration_dm']:.3f} (SE={m3.bse['duration_dm']:.3f})")
print(f"duration²:     {m3.params['duration_sq_dm']:.4f} (SE={m3.bse['duration_sq_dm']:.4f})")
print(f"female×dur:    {m3.params['fem_dur_dm']:.3f} (SE={m3.bse['fem_dur_dm']:.3f})")
print(f"female×dur²:   {m3.params['fem_dur_sq_dm']:.5f} (SE={m3.bse['fem_dur_sq_dm']:.5f})")

# Now repeat for unisex salons only
print("\n" + "="*60)
print("UNISEX SALONS ONLY")
print("="*60)

female_ids = set(df.loc[df['is_female_bool'], 'id'].unique())
male_ids = set(df.loc[~df['is_female_bool'], 'id'].unique())
both_ids = female_ids & male_ids
dfu = df[df['id'].isin(both_ids)].copy()
print(f"Sample: {len(dfu)} obs, {dfu['id'].nunique()} salons")

# Re-demean for unisex subsample
salon_means_u = dfu.groupby('id')[['price', 'duration', 'duration_sq']].transform('mean')
dfu['price_dm'] = dfu['price'] - salon_means_u['price']
dfu['duration_dm'] = dfu['duration'] - salon_means_u['duration']
dfu['duration_sq_dm'] = dfu['duration_sq'] - salon_means_u['duration_sq']
salon_means2_u = dfu.groupby('id')[['fem_dur', 'fem_dur_sq']].transform('mean')
dfu['fem_dur_dm'] = dfu['fem_dur'] - salon_means2_u['fem_dur']
dfu['fem_dur_sq_dm'] = dfu['fem_dur_sq'] - salon_means2_u['fem_dur_sq']
fem_mean_u = dfu.groupby('id')['is_female_bool'].transform('mean')
dfu['female_dm'] = dfu['is_female_bool'].astype(float) - fem_mean_u

# Model 1u: Linear
m1u = ols('price_dm ~ female_dm + duration_dm + fem_dur_dm - 1', data=dfu).fit(
    cov_type='cluster', cov_kwds={'groups': dfu['id']})
print("\n=== Model 1u: Linear ===")
print(f"female:        {m1u.params['female_dm']:.3f} (SE={m1u.bse['female_dm']:.3f})")
print(f"duration:      {m1u.params['duration_dm']:.3f} (SE={m1u.bse['duration_dm']:.3f})")
print(f"female×dur:    {m1u.params['fem_dur_dm']:.3f} (SE={m1u.bse['fem_dur_dm']:.3f})")

# Model 2u: + Duration²
m2u = ols('price_dm ~ female_dm + duration_dm + duration_sq_dm + fem_dur_dm - 1', data=dfu).fit(
    cov_type='cluster', cov_kwds={'groups': dfu['id']})
print("\n=== Model 2u: + Duration² ===")
print(f"female:        {m2u.params['female_dm']:.3f} (SE={m2u.bse['female_dm']:.3f})")
print(f"duration:      {m2u.params['duration_dm']:.3f} (SE={m2u.bse['duration_dm']:.3f})")
print(f"duration²:     {m2u.params['duration_sq_dm']:.4f} (SE={m2u.bse['duration_sq_dm']:.4f})")
print(f"female×dur:    {m2u.params['fem_dur_dm']:.3f} (SE={m2u.bse['fem_dur_dm']:.3f})")

# Model 3u: + Duration² + Female×Duration²
m3u = ols('price_dm ~ female_dm + duration_dm + duration_sq_dm + fem_dur_dm + fem_dur_sq_dm - 1', data=dfu).fit(
    cov_type='cluster', cov_kwds={'groups': dfu['id']})
print("\n=== Model 3u: + Duration² + Female×Duration² ===")
print(f"female:        {m3u.params['female_dm']:.3f} (SE={m3u.bse['female_dm']:.3f})")
print(f"duration:      {m3u.params['duration_dm']:.3f} (SE={m3u.bse['duration_dm']:.3f})")
print(f"duration²:     {m3u.params['duration_sq_dm']:.4f} (SE={m3u.bse['duration_sq_dm']:.4f})")
print(f"female×dur:    {m3u.params['fem_dur_dm']:.3f} (SE={m3u.bse['fem_dur_dm']:.3f})")
print(f"female×dur²:   {m3u.params['fem_dur_sq_dm']:.5f} (SE={m3u.bse['fem_dur_sq_dm']:.5f})")
