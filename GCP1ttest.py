#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import scipy.stats as stats
from matplotlib import pyplot as plt
import seaborn as sns


# #Baseline heterogeneity was quantified using coefficients of variation for multiple base saturation, pH, and Mg. Sampling density per plot was determined using the maximum observed coefficient across indicators to ensure robustness under worst-case heterogeneity. Plots exhibiting high baseline variability (anything above .4) would need more sampling density than more homogeneous plots, but results give a minimum number of paired sampling locations to ensure compliance
# 

# In[2]:


GCP1_pre_and_post = pd.read_csv("GCP1_pre_and_post.csv")


# In[3]:


GCP1_ALS_midterm = pd.read_csv("GCP1_ALS_midterm.csv",skiprows=8, header=None)


# In[4]:


MWLpre_pH = GCP1_pre_and_post.loc[
    (GCP1_pre_and_post["Report Date"] == "2024-09-06") &
    (~GCP1_pre_and_post["Info3"].str.contains("control", case=False, na=False)),
    "PH"
]
#MWLpre_pH.head(10)


# In[5]:


MWLpost_pH = GCP1_pre_and_post.loc[
    (GCP1_pre_and_post["Report Date"] == "2024-10-02") &
    (~GCP1_pre_and_post["Info3"].str.contains("control", case=False, na=False)),
    "PH"]

#MWLpost_pH.head(10)


# #Across 36 paired soil samples, mean pH increased by 0.19 ± 0.44 units following application, and the increase was statistically significant (paired t-test, t(35)=2.60, p=0.014). This indicates a consistent alkalinity response attributable to treatment rather than random variability.

# In[6]:


delta = MWLpost_pH.values - MWLpre_pH.values
delta.mean(), delta.std(ddof=1)


# In[7]:


#filter
df = GCP1_pre_and_post[
    (~GCP1_pre_and_post["Info3"].str.contains("control", case=False, na=False)) &
    (GCP1_pre_and_post["Report Date"].isin(["2024-09-06", "2024-10-02"]))
]


# In[8]:


#average within plot
df_mean = (
    df.groupby(["Info3", "Report Date"])["PH"]
      .mean()
      .reset_index()
)


# In[9]:


#pivot
wide = df_mean.pivot(index="Info3", columns="Report Date", values="PH")
wide = wide.dropna()   # only plots with both dates


# In[10]:


#paired delta and t test
pre  = wide["2024-09-06"]
post = wide["2024-10-02"]

delta = post - pre

delta.mean(), delta.std(ddof=1)
stats.ttest_rel(pre, post)


# In[11]:


import scipy.stats as stats

stats.ttest_rel(
    MWLpre_pH.sort_index(),
    MWLpost_pH.sort_index()
)


# In[12]:


print(np.var(MWLpre_pH),np.var(MWLpost_pH))


# In[13]:


Mg_pre = GCP1_pre_and_post.loc[
    (GCP1_pre_and_post["Report Date"] == "2024-09-06") &
    (~GCP1_pre_and_post["Info3"].str.contains("control", case=False, na=False)),
    "Mg ppm"
]
Mg_pre.head(10)


# In[14]:


Mg_post = GCP1_pre_and_post.loc[
    (GCP1_pre_and_post["Report Date"] == "2024-10-02") &
    (~GCP1_pre_and_post["Info3"].str.contains("control", case=False, na=False)),
    "Mg ppm"
]
Mg_post.head(10)


# In[15]:


print(np.var(Mg_pre),np.var(Mg_post))


# In[16]:


#the ratio indicates that pre and post DO NOT have equal variance


# In[17]:


import scipy.stats as stats

stats.ttest_rel(
    Mg_pre.sort_index(),
    Mg_post.sort_index()
)


# In[18]:


#Magnesium concentrations increased significantly from baseline to post-application (paired t-test, t(35) = 2.45, p = 0.019).


# In[19]:


#Magnitude of change
delta = Mg_post.values - Mg_pre.values
delta.mean(), delta.std(ddof=1)


# In[20]:


#Report effect size (Cohen’s dz)
dz = delta.mean() / delta.std(ddof=1)
dz


# In[21]:


#good results so far
stats.wilcoxon(Mg_pre, Mg_post)


# In[22]:


baseline = GCP1_pre_and_post.loc[
    GCP1_pre_and_post["Report Date"] == "2024-09-06"
].copy()

# strip whitespace from grouping columns (important)
baseline["Info2"] = baseline["Info2"].astype(str).str.strip()
baseline["Info3"] = baseline["Info3"].astype(str).str.strip()


# In[23]:


#distinguish the control portion from the rest of baseline samples
baseline["is_control"] = baseline["Info3"].str.contains(
    "control", case=False, na=False
)
baseline["is_control"].value_counts()


# In[24]:


control = baseline.loc[baseline["is_control"], "CEC"]
treat   = baseline.loc[~baseline["is_control"], "CEC"]

stats.ttest_ind(control, treat, equal_var=False)


# In[25]:


control = baseline.loc[baseline["is_control"], "PH"]
treat   = baseline.loc[~baseline["is_control"], "PH"]

stats.ttest_ind(control, treat, equal_var=False)


# In[26]:


baseline["BaseSaturation_pct"] = (
    baseline["PERCENTCA"] +
    baseline["PERCENTMG"] +
    baseline["PERCENTK"] +
    baseline["PERCENTNA"]
)
#baseline["BaseSaturation_pct"].head(10)
baseline["BaseSaturation_pct"].describe()


# In[27]:


baseline["is_control"] = baseline["Info3"].str.contains(
    "control", case=False, na=False
)


# In[28]:


import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(6, 5))
sns.boxplot(
    data=baseline,
    x="is_control",
    y="BaseSaturation_pct"
)
plt.xticks([0, 1], ["Treatment", "Control"])
plt.ylabel("Base Saturation (%)")
plt.title("Baseline base saturation: control vs treatment plots")
plt.tight_layout()
plt.show()


# In[29]:


#Baseline base saturation varied across the site, with control plots exhibiting slightly higher median values than treatment plots but substantial overlap between distributions, consistent with expected spatial heterogeneity in non-contiguous agricultural soils.


# In[30]:


baseline["is_control"] = baseline["Info3"].str.contains(
    "control", case=False, na=False
)

baseline["Group"] = baseline["is_control"].map({
    False: "Treatment",
    True: "Control"
})


# In[31]:


sns.boxplot(
    data=baseline,
    x="Group",
    y="BaseSaturation_pct"
)

sns.stripplot(
    data=baseline,
    x="Group",
    y="BaseSaturation_pct",
    color="black",
    alpha=0.6,
    jitter=True
)


# In[32]:


baseline["BaseSaturation_pct"] = (
    baseline["PERCENTCA"] +
    baseline["PERCENTMG"] +
    baseline["PERCENTK"] +
    baseline["PERCENTNA"]
)
baseline["BaseSaturation_pct"].head(10)


# In[33]:


# 1) Define baseline
baseline = GCP1_pre_and_post.loc[
    GCP1_pre_and_post["Report Date"] == "2024-09-06"
].copy()

# 2) Identify controls
baseline["is_control"] = baseline["Info3"].astype(str).str.contains(
    "control", case=False, na=False
)

# 3) Compute base saturation (must exist before grouping)
baseline["BaseSaturation_pct"] = (
    baseline["PERCENTCA"] +
    baseline["PERCENTMG"] +
    baseline["PERCENTK"] +
    baseline["PERCENTNA"]
)


# In[34]:


# focus on treatment plots only
baseline_t = baseline.loc[~baseline["is_control"]].copy()

within_plot = (
    baseline_t
    .groupby("Info3")["BaseSaturation_pct"]
    .agg(["count", "mean", "std"])
)

within_plot["cv"] = within_plot["std"] / within_plot["mean"]
within_plot.sort_values("cv", ascending=False)


# In[35]:


baseline = GCP1_pre_and_post.loc[
    GCP1_pre_and_post["Report Date"] == "2024-09-06"
].copy()

baseline["is_control"] = baseline["Info2"].astype(str).str.contains(
    "control", case=False, na=False
)

baseline["BaseSaturation_pct"] = (
    baseline["PERCENTCA"] +
    baseline["PERCENTMG"] +
    baseline["PERCENTK"] +
    baseline["PERCENTNA"]
)


# In[36]:


def within_plot_cv(df, value_col, plot_col="Info3", exclude_controls=True):
    d = df.copy()
    if exclude_controls and "is_control" in d.columns:
        d = d.loc[~d["is_control"]]
    out = (
        d.groupby(plot_col)[value_col]
        .agg(["count", "mean", "std"])
    )
    out["cv"] = out["std"] / out["mean"]
    return out.sort_values("cv", ascending=False)


# In[37]:


cv_pH = within_plot_cv(baseline, "PH")
cv_pH


# In[38]:


cv_Mg = within_plot_cv(baseline, "MG ppm")
cv_Mg


# In[39]:


cv_bs  = within_plot_cv(baseline, "BaseSaturation_pct")
cv_pH  = within_plot_cv(baseline, "PH")        
cv_Mg  = within_plot_cv(baseline, "MG ppm")

cv_compare = (
    cv_bs[["cv"]].rename(columns={"cv": "cv_bs"})
    .join(cv_pH[["cv"]].rename(columns={"cv": "cv_pH"}))
    .join(cv_Mg[["cv"]].rename(columns={"cv": "cv_Mg"}))
)

cv_compare["cv_max"] = cv_compare.max(axis=1)
cv_compare.sort_values("cv_max", ascending=False)


# In[40]:


cv_compare["cv_max"] = cv_compare.max(axis=1)
cv_compare.sort_values("cv_max", ascending=False)
#values above .4 are high for soil co-variance (see isometric) and this shows high variability affecting the sampling plan given the fact that there is very low homogeneity. 
#This means that we need to stratify by plot
#allocate many more samples to the high covariance plots I have here
#avoid one-size-fits-all sampling density


# In[48]:


plot_col = "Info3"
pair_col = "Sample ID"
mg_col   = "MG ppm"
ph_col   = "PH"  

#df_nc = df.loc[~df["is_control"]]

pre = df_nc.loc[df_nc["Report Date"] == date_pre,
                [plot_col, pair_col, mg_col, ph_col]].set_index([plot_col, pair_col])

post = df_nc.loc[df_nc["Report Date"] == date_post,
                 [plot_col, pair_col, mg_col, ph_col]].set_index([plot_col, pair_col])

pairs = pre.join(post, lsuffix="_pre", rsuffix="_post", how="inner")

pairs["d_MG"] = pairs[f"{mg_col}_post"] - pairs[f"{mg_col}_pre"]
pairs["d_PH"] = pairs[f"{ph_col}_post"] - pairs[f"{ph_col}_pre"]

pairs.head(), pairs.shape


# In[49]:


date_pre  = "2024-09-06"
date_post = "2024-10-02"

df = GCP1_pre_and_post.copy()

# control samples
df["is_control"] = df["Info3"].astype(str).str.contains("control", case=False, na=False)

# Keep only non-control samples
df = df.loc[~df["is_control"]].copy()

pre = df.loc[df["Report Date"] == date_pre,  [plot_col, pair_col, mg_col, ph_col]].copy()
post= df.loc[df["Report Date"] == date_post, [plot_col, pair_col, mg_col, ph_col]].copy()

# index the pairs
pre  = pre.set_index([plot_col, pair_col]).sort_index()
post = post.set_index([plot_col, pair_col]).sort_index()

# Keep only pairs that exist in both
pairs = pre.join(post, lsuffix="_pre", rsuffix="_post", how="inner")

pairs["d_MG"] = pairs[f"{mg_col}_post"] - pairs[f"{mg_col}_pre"]
pairs["d_PH"] = pairs[f"{ph_col}_post"] - pairs[f"{ph_col}_pre"]

pairs.head()


# In[43]:


from statsmodels.stats.power import TTestPower

analysis = TTestPower()

sd_d_MG = pairs["d_MG"].std(ddof=1)
sd_d_PH = pairs["d_PH"].std(ddof=1)

# choose detection targets
delta_MG = 50    
delta_PH = 0.15  

dz_MG = delta_MG / sd_d_MG
dz_PH = delta_PH / sd_d_PH

n_MG_80 = analysis.solve_power(
    effect_size=abs(dz_MG),
    power=0.80,
    alpha=0.05,
    alternative="two-sided"
)

n_PH_80 = analysis.solve_power(
    effect_size=abs(dz_PH),
    power=0.80,
    alpha=0.05,
    alternative="two-sided"
)

n_MG_80, n_PH_80


# In[44]:


import numpy as np
import pandas as pd
from statsmodels.stats.power import TTestPower

analysis = TTestPower()

def n_required_from_sd(sd_diff, delta, power=0.80, alpha=0.05):
    dz = delta / sd_diff
    return analysis.solve_power(effect_size=abs(dz), power=power, alpha=alpha, alternative="two-sided")

per_plot = (
    pairs.groupby(level=0)[["d_MG", "d_PH"]]
    .agg(["count", "mean", "std"])
)

# simplify column names
per_plot.columns = ["_".join(c).strip() for c in per_plot.columns.to_flat_index()]
per_plot = per_plot.rename(columns={
    "d_MG_count": "n_pairs_pilot",
    "d_MG_std": "sd_d_MG",
    "d_PH_std": "sd_d_PH",
    "d_MG_mean": "mean_d_MG",
    "d_PH_mean": "mean_d_PH",
})

per_plot


# Statistical Power doesn't make sense for this project because our project claims require plot specific detection of treatment-driven change, power calculations were performed independently for each plot using a paired sampling design (matched locations at baseline and post-application). For each plot, the standard deviation of paired differences was estimated from pilot paired observations, and the number of paired monitoring locations was selected to achieve ≥80% power (α = 0.05, two-sided) to detect a mean change of at least .15 for pH and 50ppm for Mg. The final sampling density per plot was set to the maximum required across pH and Mg, ensuring ≥80% power for both endpoints within each plot.

# In[45]:


from statsmodels.stats.power import TTestPower
analysis = TTestPower()

n = 13
dz_80 = analysis.solve_power(nobs=n, power=0.80, alpha=0.05, alternative="two-sided")

mde_MG = dz_80 * sd_d_MG
mde_PH = dz_80 * sd_d_PH

mde_MG, mde_PH


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




