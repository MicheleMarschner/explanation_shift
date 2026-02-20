# Plots & Tables


## RQ1: Effect of shifts on robustness of post-hoc explanations - Shifts destabilize explanations, and severity matters

**Report together:**
* Plot B4 (severity vs drift, one line per corruption) OR/AND Plot 1 (acc_vs_exp)
* Violin-by-severity of stability (ρ distribution) (optional but strong)
One violin (ρ distribution vs severity) or one short “heterogeneity” figure

**Why jointly:**
Plot B gives the trend (mean drift grows with severity, differs by corruption).
The violin adds “this is not just the mean”: drift may be heterogeneous (some samples blow up early).
A line graph (acc_vs_exp) only shows the mean drift. The Violin plot reveals the mechanism (RQ2). Does a severity of 3 mean all explanations degrade equally? Or does it mean 50% stay perfectly intact while 50% completely break (bimodal)? The violin plot reveals if the shift creates "heavy tails" of complete failures, which is vital context for RQ1.

**What’s redundant here:**
You don’t need all three similarity metrics in the main text. Pick one primary (I’d use Spearman ρ because you already interpret it heavily and you use “1−ρ” as drift) and put cosine/IoU in appendix.

**Why enough:**
* Why it is sufficient: Your acc_vs_exp lines, violin_by_severity plots, and pooled similarity tables provide an exhaustive empirical taxonomy of how explanations degrade. You aren't just saying "they break"; you are showing exactly at what severity they break, how much they break (magnitude), and whether the breaking is uniform or highly varied (violin plots).
* What to write: Use these plots to establish the baseline truth: explanations are fragile OOD, often more fragile than the model's accuracy itself.

**Overview Table:**
* Per-corruption summary: Corruption type, mean ρ (clean vs sev=1/3/5), Δρ from clean, accuracy drop at sev=5 ((grouped if too many) -> Quantifies “which corruptions cause fast drift” vs plots' visual trends. Add p‑values for trends if available.


## RQ2: Mechanisms/causes of decoupling between performance P and explanation quality Q - Decoupling: ΔP and ΔE do not track each other reliably

**Report together:**
* Acc vs explanation similarity over severity (dual axis) for 1–2 representative corruptions (not all):
* ΔP vs ΔE scatter (one point per corruption×severity) + Spearman ρ on plot
* Trust zones stacked bar (Robust / Silent Drift / Expected Failure / Stubborn Failure)

**Why jointly:**
* The dual-axis severity plot is the intuition (“here’s an example where accuracy stays okay but explanations drift” or vice versa). shows the longitudinal trend over severity (e.g., "Look, accuracy stays flat at 90% but explanation similarity drops to 0.4").
* The scatter is your global decoupling picture (“across conditions, they don’t align cleanly”): aggregates this into a global scatter, proving this isn't just one weird severity level, but a systemic decoupling across different corruptions.
* These are two views of the same object. plot_deltaP_vs_deltaE is the clearest for “decoupling” because it collapses to ΔP on one axis and ΔE on the other. acc_vs_exp helps the reader see how those points arise as severity increases per corruption. You can refer to the latter briefly when introducing ΔP and ΔE, and then use the delta‑scatter as the main RQ2 figure.
* Trust zones turn it into a decision-relevant story: “Silent Drift exists” = explanation changes even when accuracy is still correct (risk for trust claims). Translates this abstract math into interpretable categories. It explicitly names the decoupling mechanisms: "Silent Drift" (Explanation breaks, model doesn't) and "Stubborn Failure" (Model breaks, explanation doesn't).

**What’s redundant here:**
The grouped Plot C scatters per corruption×severity and the Spearman table largely report the same thing. Use:
Scatter (pooled) + Spearman table (best for reporting)
and keep the per-condition scatter plots only for appendix or debugging.
The correlation heatmap overlaps with “are signals redundant?” and can be appendix unless you explicitly discuss signal redundancy.

**Additionaly Overview Tables:**
Decoupling table: association between ΔP and ΔE -> the existing *Spearman table* (Corruption, Severity, ρ (ΔP vs ΔE), p‑value, N pairs)
Per corruption × severity:
* Spearman ρ(ΔP proxy, ΔE) (+ n)
* optionally: also ρ(|ΔEntropy|, drift)
-> Why: it’s the “global decoupling evidence” behind the scatter plots.


Trust-zones table (**or just include values on the stacked plot**)
Per severity: % Robust, % Silent Drift, % Expected Failure, % Stubborn Failure (and optionally per corruption if space allows)
--> Why: easy to cite in text (“Silent Drift rises from X%→Y% with severity”).


## RQ3: What shift-aware evaluation protocols should look like - Drift is operationally useful as a monitoring signal

**Report together:**
* ROC: failure detection (drift vs uncertainty vs confidence drift)
* Incremental value test: AUC(unc) vs AUC(unc+drift) + bootstrap CI for ΔAUC
* (Optional) Drift–uncertainty Spearman correlations (one line in text)
* ggf. Plot 3 (metric_vs_severity_faith_fog Quantus Faithfulness)
* ggf. Stacked Trust zones

Why jointly:
* ROC shows practical detectability: drift predicts corruption-induced failure well.
* The ΔAUC + CI answers the reviewer’s “is this just confidence?” question in one shot.
* Correlations help interpret why it’s not redundant.
These jointly motivate your guidelines: they show that (a) the same aggregate accuracy can hide very different mixes of zones, (b) drift can be a valuable early warning but also fails in “stubborn failure” regions, and (c) therefore protocols must explicitly characterize the shift, define slices (invariant, both‑correct, etc.), and report conditional metrics rather than global certificates

The ROC Curve proves the practical utility of your findings: You can use Explanation Drift as a monitoring tool to catch model failures under shift. This is a concrete recommendation for a "shift-aware protocol."
The Quantus Faithfulness Plot answers the crucial normative question: If an explanation shifts, is it wrong? If faithfulness also drops, it proves that post-hoc explainers lose their validity OOD (Protocol Invalidity, RQ2), meaning you cannot make global "Trust Claims" (RQ3).

This bundle is a great “mechanism/implication” piece for RQ2 and a direct motivator for RQ3 (monitoring + sanity checks).
Use the RQ2 bridge bundle as evidence for protocol components:
* Need paired design (clean-correct filtering, fixed target ŷ_clean)
* Need multiple signals (uncertainty + drift)
* Need slice reporting (invariant/both_correct)
* Need robustness checks (bootstrap CI, not single AUC)
* Need trust-zone style reporting (Silent Drift risks)

**Additionally Overview Table:**
Main summary table: drift vs baselines for failure detection
One row per condition (or per corruption, with severities in columns): 
* AUC(drift), AUC(uncertainty), AUC(conf-drift)
* ΔAUC = AUC(unc+drift) − AUC(unc) with 95% CI
* N pairs (after clean-correct filter) + failure rate

---------------------

## Redundant: 
* Correlation Heatmaps vs. Multiple Metric Lines (cos, IoU, rho)
→ pick ρ as main; cosine/IoU in appendix. / Action: Show the Correlation Heatmap early on (or in the appendix) to prove the metrics are redundant. Then, pick exactly ONE metric (e.g., Explanation Drift via 1−ρ) for all your main visualizations to keep the narrative clean.
* Plot B1-B3 (Similarity) vs. Plot B4 (Drift Magnitude) vs. Plot 1 (acc_vs_exp)
→ From B pick one orientation. I’d keep drift magnitude (1−ρ) because it reads naturally (“higher = worse”). Plot 1 (acc_vs_exp) could even be enough 
* Plot 2 (plot_deltaP_vs_deltaE) vs. Plot C (pooled_scatter Entropy vs Drift) vs. Spearman Table
Why they are redundant: They all attempt to answer: "Does model prediction change when the explanation changes?" plot_deltaP_vs_deltaE uses accuracy/error as the proxy for ΔP, while Plot C uses ΔEntropy.
Action: Use Plot 2 (plot_deltaP_vs_deltaE) in the main text for RQ2, as "Accuracy drop" is much more intuitive for readers to grasp than "Entropy shift". Move Plot C and the Spearman tables to the appendix to prove that the decoupling holds true even when looking at soft probabilities (entropy).

## Optional:
* Metric correlation heatmap (pooled)
→ supports “signals are not identical” but often belongs in appendix unless you explicitly discuss redundancy / proxies.
* Clean confidence vs vulnerability
→ great as an additional insight (“clean confidence doesn’t guarantee robustness”), but not central unless you make it part of RQ2 mechanism (“confidence ≠ stability”).


**The two things to add:**
* A tiny robustness check across multiple corruption types + severities for the ROC/ΔAUC story: a table/heatmap of ΔAUC per condition (+ maybe CI or just median + % positive) That makes the monitoring claim clearly general, not just “gaussian_noise sev3”.
* For the B1-3 or B4 plots - add metric sensitivity by combining all (p, iou, cos) in one plot (normalized): 
“Metric Sensitivity” comparison (ρ vs IoU across severities) -> single combined plot that directly compares ρ vs IoU on the same axes (or normalized).


**Quantus:**
* One compact result showing that your chosen QQ metrics behave sensibly under noise/adversarial perturbations (even on clean or mildly shifted data) makes your protocol suggestions more convincing, because you can argue that at least some of the observed decoupling is not just an artefact of a bad estimator but reflects real behaviour.
* The Missing Piece (Mechanisms): Showing that P and Q decouple is only step one. RQ2 asks why. This is where your Quantus (Faithfulness) plots are critical.
    * Mechanism A (Explainer Failure / Protocol Invalidity): If your Quantus faithfulness score drops significantly under shift, it means the explainer method itself is breaking down because it's being fed OOD data. The explanation drifted, but it's no longer faithfully reflecting the model.
    * Mechanism B (Model Strategy Shift): If the explanation drifts dramatically, BUT the Quantus faithfulness remains high, this is a profound finding. It means the explainer is working fine, but the model has completely changed its internal reasoning to solve the corrupted image, even if it still gets the right answer (Silent Drift).
Make sure you explicitly use the Quantus results to differentiate between Mechanism A and Mechanism B in your text.

--------------------------------

**Note Tables**
-> use tables to summarize across many conditions where plots would be too busy.
-> Use your pairs_table_from_pt.csv – it's perfect for computing these (pandas groupby/agg for means/ρ).
-> Format: 3 significant digits, bold significant p<0.01, footnotes for metric definitions.-