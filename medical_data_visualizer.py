import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1
df = pd.read_csv(r"C:\Users\MOHAMMED SHOAIB B\OneDrive\Desktop\test\boilerplate-medical-data-visualizer\medical_examination.csv")

# 2
bmi = df['weight'] / ((df['height'] / 100) ** 2)
df['overweight'] = (bmi > 25).astype(int)

# 3
df['cholesterol'] = (df['cholesterol'] > 1).astype(int)
df['gluc'] = (df['gluc'] > 1).astype(int)

# 4
def draw_cat_plot():
    # 5
    df_cat = pd.melt(df,
                     id_vars=['cardio'],
                     value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight'])

    # 6
    df_cat = df_cat.groupby(['cardio', 'variable', 'value']).size().reset_index(name='total')

    # 7
    g = sns.catplot(data=df_cat,
                    x='variable',
                    y='total',
                    hue='value',
                    col='cardio',
                    kind='bar',
                    height=5,
                    aspect=1)

    g.set_axis_labels("variable", "total")
    g.set_titles("cardio = {col_name}")
    g._legend.set_title('value')

    # 8
    fig = g.fig

    # 9
    fig.savefig('catplot.png')
    return fig


# 10
def draw_heat_map():
    # 11
    df_heat = df.copy()

    # Strict filtering: correct blood pressure, height, weight
    df_heat = df_heat[
        (df_heat['ap_lo'] <= df_heat['ap_hi']) &
        (df_heat['height'] >= df_heat['height'].quantile(0.025)) &
        (df_heat['height'] <= df_heat['height'].quantile(0.975)) &
        (df_heat['weight'] >= df_heat['weight'].quantile(0.025)) &
        (df_heat['weight'] <= df_heat['weight'].quantile(0.975))
    ]

    # 12
    corr = df_heat.corr()  # do not round, full precision

    # 13
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # 14
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr,
                mask=mask,
                annot=True,
                fmt=".1f",
                linewidths=.5,
                square=True,
                cbar_kws={"shrink": 0.5},
                center=0,
                ax=ax)

    # 15
    # nothing extra needed

    # 16
    fig.savefig('heatmap.png')
    return fig


# Optional: run directly
if __name__ == "__main__":
    draw_cat_plot()
    draw_heat_map()
