# ============================================================
# 🦠 COVID-19 Data Analysis — Advanced Project
# Tools: Python | Pandas | Matplotlib | Seaborn
# Author: Mahalakshmi P | Data Analytics
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import seaborn as sns
import numpy as np
import warnings
warnings.filterwarnings('ignore')

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.facecolor': '#0D1117',
    'axes.facecolor': '#161B22',
    'text.color': '#E6EDF3',
    'axes.labelcolor': '#8B949E',
    'xtick.color': '#8B949E',
    'ytick.color': '#8B949E',
    'axes.edgecolor': '#30363D',
    'grid.color': '#21262D',
    'grid.alpha': 0.5,
})

COLORS = ['#58A6FF','#3FB950','#FF7B72','#D2A8FF','#FFA657','#79C0FF','#F78166','#56D364']

# ── 1. Dataset ───────────────────────────────────────────────
print("🦠 Loading COVID-19 dataset...")

countries = ['India','USA','Brazil','UK','Russia','France','Germany','Italy']
months    = ['Jan','Feb','Mar','Apr','May','Jun',
             'Jul','Aug','Sep','Oct','Nov','Dec']

np.random.seed(42)
rows = []
for country in countries:
    base = np.random.randint(5000, 50000)
    for i, month in enumerate(months):
        wave = np.sin(i / 3) * 0.4 + 1
        cases  = int(base * wave * np.random.uniform(0.85, 1.15))
        deaths = int(cases * np.random.uniform(0.01, 0.04))
        recovered = int(cases * np.random.uniform(0.75, 0.92))
        vaccinated = int(np.random.randint(100000, 5000000) * (i + 1) / 12)
        rows.append([country, month, i+1, cases, deaths, recovered, vaccinated])

df = pd.DataFrame(rows, columns=[
    'Country','Month','Month_Num','Cases','Deaths','Recovered','Vaccinated'])

df['Active']         = df['Cases'] - df['Deaths'] - df['Recovered']
df['Death_Rate']     = (df['Deaths'] / df['Cases'] * 100).round(2)
df['Recovery_Rate']  = (df['Recovered'] / df['Cases'] * 100).round(2)
df['Active_Rate']    = (df['Active'] / df['Cases'] * 100).round(2)

print(f"✅ Dataset: {len(df)} rows × {len(df.columns)} columns")
print(f"   Countries: {df['Country'].nunique()} | Months: {df['Month'].nunique()}\n")

# ── 2. Summary ───────────────────────────────────────────────
total_cases     = df['Cases'].sum()
total_deaths    = df['Deaths'].sum()
total_recovered = df['Recovered'].sum()
total_vaccinated= df['Vaccinated'].sum()
global_death_rate    = (total_deaths / total_cases * 100)
global_recovery_rate = (total_recovered / total_cases * 100)

print("── Global Summary ──────────────────────────────────────")
print(f"  🦠 Total Cases     : {total_cases:,}")
print(f"  💀 Total Deaths    : {total_deaths:,}")
print(f"  💚 Total Recovered : {total_recovered:,}")
print(f"  💉 Total Vaccinated: {total_vaccinated:,}")
print(f"  📉 Death Rate      : {global_death_rate:.2f}%")
print(f"  📈 Recovery Rate   : {global_recovery_rate:.2f}%\n")

# ── 3. Dashboard ─────────────────────────────────────────────
fig = plt.figure(figsize=(20, 16), facecolor='#0D1117')
fig.suptitle('🦠  COVID-19 Global Data Analysis Dashboard',
             fontsize=24, fontweight='bold', color='#E6EDF3', y=0.98)

gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

# -- Chart 1: Total Cases by Country --------------------------
ax1 = fig.add_subplot(gs[0, 0])
country_cases = df.groupby('Country')['Cases'].sum().sort_values(ascending=True)
bars = ax1.barh(country_cases.index, country_cases.values,
                color=COLORS[:len(country_cases)], height=0.6, edgecolor='#0D1117')
ax1.set_title('Total Cases by Country', color='#E6EDF3', fontweight='bold', pad=10)
ax1.set_xlabel('Total Cases', color='#8B949E')
for bar in bars:
    ax1.text(bar.get_width() + 1000, bar.get_y() + bar.get_height()/2,
             f'{bar.get_width():,.0f}', va='center', fontsize=7, color='#8B949E')

# -- Chart 2: Monthly Cases Trend (Multi-line) ----------------
ax2 = fig.add_subplot(gs[0, 1:])
for i, country in enumerate(countries[:5]):
    cdf = df[df['Country'] == country].sort_values('Month_Num')
    ax2.plot(cdf['Month'], cdf['Cases'], marker='o', markersize=4,
             label=country, color=COLORS[i], linewidth=2)
ax2.set_title('Monthly Cases Trend (Top 5 Countries)', color='#E6EDF3', fontweight='bold', pad=10)
ax2.set_xlabel('Month', color='#8B949E')
ax2.set_ylabel('Cases', color='#8B949E')
ax2.legend(fontsize=8, facecolor='#161B22', labelcolor='#E6EDF3', edgecolor='#30363D')
ax2.grid(True, alpha=0.3)

# -- Chart 3: Death Rate by Country (Bar) ---------------------
ax3 = fig.add_subplot(gs[1, 0])
death_rate = df.groupby('Country')['Death_Rate'].mean().sort_values(ascending=False)
colors_d = ['#FF7B72' if v > death_rate.mean() else '#58A6FF' for v in death_rate.values]
ax3.bar(death_rate.index, death_rate.values, color=colors_d, width=0.6, edgecolor='#0D1117')
ax3.set_title('Avg Death Rate by Country (%)', color='#E6EDF3', fontweight='bold', pad=10)
ax3.set_ylabel('Death Rate %', color='#8B949E')
ax3.tick_params(axis='x', rotation=45)
high = mpatches.Patch(color='#FF7B72', label='Above Average')
low  = mpatches.Patch(color='#58A6FF', label='Below Average')
ax3.legend(handles=[high, low], fontsize=7, facecolor='#161B22',
           labelcolor='#E6EDF3', edgecolor='#30363D')

# -- Chart 4: Recovery Rate Trend (Area) ----------------------
ax4 = fig.add_subplot(gs[1, 1])
monthly_recovery = df.groupby('Month_Num')['Recovery_Rate'].mean()
ax4.fill_between(monthly_recovery.index, monthly_recovery.values,
                 alpha=0.3, color='#3FB950')
ax4.plot(monthly_recovery.index, monthly_recovery.values,
         color='#3FB950', linewidth=2.5, marker='o', markersize=5)
ax4.set_title('Global Recovery Rate Trend', color='#E6EDF3', fontweight='bold', pad=10)
ax4.set_xlabel('Month', color='#8B949E')
ax4.set_ylabel('Recovery Rate %', color='#8B949E')
ax4.set_xticks(range(1, 13))
ax4.set_xticklabels(months, rotation=45, fontsize=8)
ax4.grid(True, alpha=0.3)

# -- Chart 5: Vaccination Progress ----------------------------
ax5 = fig.add_subplot(gs[1, 2])
vax = df.groupby('Country')['Vaccinated'].sum().sort_values(ascending=False)
ax5.bar(vax.index, vax.values / 1e6, color=COLORS, width=0.6, edgecolor='#0D1117')
ax5.set_title('Total Vaccinated (Millions)', color='#E6EDF3', fontweight='bold', pad=10)
ax5.set_ylabel('Vaccinated (M)', color='#8B949E')
ax5.tick_params(axis='x', rotation=45)

# -- Chart 6: Cases vs Deaths Scatter -------------------------
ax6 = fig.add_subplot(gs[2, 0])
for i, country in enumerate(countries):
    cdf = df[df['Country'] == country]
    ax6.scatter(cdf['Cases'], cdf['Deaths'], color=COLORS[i],
                alpha=0.7, s=60, label=country, edgecolors='#0D1117')
ax6.set_title('Cases vs Deaths Correlation', color='#E6EDF3', fontweight='bold', pad=10)
ax6.set_xlabel('Cases', color='#8B949E')
ax6.set_ylabel('Deaths', color='#8B949E')
ax6.legend(fontsize=7, facecolor='#161B22', labelcolor='#E6EDF3', edgecolor='#30363D')

# -- Chart 7: KPI Cards ---------------------------------------
ax7 = fig.add_subplot(gs[2, 1])
ax7.axis('off')
ax7.set_facecolor('#0D1117')
kpis = [
    ('Total Cases',      f'{total_cases:,}',         '#58A6FF'),
    ('Total Deaths',     f'{total_deaths:,}',         '#FF7B72'),
    ('Total Recovered',  f'{total_recovered:,}',      '#3FB950'),
    ('Death Rate',       f'{global_death_rate:.2f}%', '#FFA657'),
]
for i, (label, value, color) in enumerate(kpis):
    y = 0.85 - i * 0.22
    ax7.add_patch(mpatches.FancyBboxPatch(
        (0.05, y - 0.08), 0.9, 0.18,
        boxstyle='round,pad=0.02',
        facecolor=color + '22', edgecolor=color, linewidth=1.5,
        transform=ax7.transAxes))
    ax7.text(0.5, y + 0.02, value, transform=ax7.transAxes,
             ha='center', fontsize=12, fontweight='bold', color=color)
    ax7.text(0.5, y - 0.04, label, transform=ax7.transAxes,
             ha='center', fontsize=9, color='#8B949E')
ax7.set_title('Key KPIs', color='#E6EDF3', fontweight='bold', pad=10)

# -- Chart 8: Monthly Deaths Heatmap --------------------------
ax8 = fig.add_subplot(gs[2, 2])
pivot = df.pivot_table(values='Deaths', index='Country', columns='Month_Num', aggfunc='sum')
pivot.columns = months
sns.heatmap(pivot, ax=ax8, cmap='RdYlGn_r', linewidths=0.5,
            linecolor='#0D1117', annot=False, fmt='.0f',
            cbar_kws={'shrink': 0.8})
ax8.set_title('Deaths Heatmap (Country × Month)', color='#E6EDF3', fontweight='bold', pad=10)
ax8.tick_params(colors='#8B949E', labelsize=7)
ax8.set_xlabel('Month', color='#8B949E')
ax8.set_ylabel('Country', color='#8B949E')

output_path = '/mnt/user-data/outputs/covid_dashboard.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='#0D1117')
print(f"✅ Dashboard saved → {output_path}")

# ── 4. Advanced Analysis ─────────────────────────────────────
print("\n── 💡 Advanced Insights ────────────────────────────────")
worst  = death_rate.idxmax()
best   = death_rate.idxmin()
best_recovery = df.groupby('Country')['Recovery_Rate'].mean().idxmax()
peak_month = df.groupby('Month')['Cases'].sum().idxmax()
print(f"  ☠️  Highest Death Rate  : {worst} ({death_rate[worst]:.2f}%)")
print(f"  ✅ Lowest Death Rate   : {best} ({death_rate[best]:.2f}%)")
print(f"  💚 Best Recovery Rate  : {best_recovery}")
print(f"  📅 Peak Cases Month    : {peak_month}")
print("────────────────────────────────────────────────────────\n")

# ── 5. Export CSV ─────────────────────────────────────────────
summary = df.groupby('Country').agg(
    Total_Cases=('Cases','sum'),
    Total_Deaths=('Deaths','sum'),
    Total_Recovered=('Recovered','sum'),
    Avg_Death_Rate=('Death_Rate','mean'),
    Avg_Recovery_Rate=('Recovery_Rate','mean'),
    Total_Vaccinated=('Vaccinated','sum')
).round(2)
csv_path = '/mnt/user-data/outputs/covid_summary.csv'
summary.to_csv(csv_path)
print(f"✅ Summary CSV → {csv_path}")
print(summary.to_string())
