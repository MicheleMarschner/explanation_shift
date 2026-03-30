Additional Metrics:
* Randomization: Model Parameter Randomization (or Input Randomization): Tests if explanations collapse when model/input is noised. This is your "sanity check" baseline – must pass on clean data before trusting under shift.


1. How Q-metrics behave under shift, and
2. Whether they track ΔE and/or ΔP, and
3. Which metrics are “reliable” vs “protocol-sensitive” under shift (your Protocol Invalidity / MetaQuantus narrative).


* what says that protocol becomes OOD
* Clean, ID and OOD -> wie bestimme ich ID

A) in den perturbationen versuchen einmal ID und einmal OOD zu sein - wie messe ich dass
Compute with in-distribution perturbations (e.g., Gaussian noise at σ matching your clean data manifold) 


B) Plot: Q vs severity with accuracy overlay
x: severity
y1: accuracy (optional overlay)
y2: Quantus score Q
one line per corruption (like your Plot B)


C) The key “RQ2” Quantus result: correlation/decoupling
You already do ΔP vs ΔE. Add ΔP vs ΔQ (Quantus score change):
x: ΔP proxy (1 − acc) or ΔEntropy
y: ΔQ = Q(clean) − Q(corr) (or Q(corr) relative to clean baseline)
one point per corruption×severity (same as your ΔP vs ΔE scatter)
This is the cleanest way to say: “Even standard faithfulness metrics can decouple from performance under shift.”
-> This supports your “Q is not a proxy for P” claim with an established evaluation family, not just drift.


D) A small “protocol invalidity” illustration (optional but very on-theme)
Pick one metric that is known to be perturbation-sensitive (e.g., deletion/insertion) and show that:
its score changes a lot under shift even when ΔE is small or vice versa.
You can do this with a simple scatter:
x: ΔE (your drift)
y: ΔQ (Quantus)
If it’s messy/inconsistent across corruptions, that’s not “bad”—it’s exactly your point: protocol confounds.


C) **Gate:** 
Model Parameter Randomization (or Input Randomization): Tests if explanations collapse when model/input is noised. This is your "sanity check" baseline – must pass on clean data before trusting under shift.


D) **MetaQuantus Angle (Minimal, high impact)**
Since full MetaQuantus might be heavy, do a lite version (1 plot):

Noise-Resilience: Run one faithfulness metric (e.g., Deletion) with increasing noise levels on clean data. Reliable metrics stay stable until high noise.

Reactivity: Test on "adversarial" pairs (clean vs. semantically similar but perturbed inputs). Good metrics react appropriately.


Metric Family	Metric	Clean Score	Sev=3 Score	Meta: Noise-Resilient?	Ties to Failure Mode
Faithfulness	Deletion	0.85	0.62	Yes (stable to 5% noise)	Protocol invalidity (OOD perturbations)
Robustness	Max-Sens.	0.12	0.28	No (drops early)	Explainer sensitivity
Sanity	Param Rand.	0.92	0.88	Yes	Baseline reliability

Sanity checks (Quantus Param-Rand >0.9 required);




1. Der "Y-Conditioning" Plot (De-confounding von P und Q)
Zielt direkt auf: 5.2 Bausteine (Prediction Invariance Conditioning / Y-Conditioning)
Was du zeigst: Einen Line-Plot mit "Corruption Severity" auf der x-Achse und "Quantus Faithfulness Score" auf der y-Achse. Du zeichnest aber zwei Linien:
Unconditioned: Der durchschnittliche Faithfulness-Score über alle Testbilder der jeweiligen Severity.
Y-Conditioned (Invariant): Der durchschnittliche Faithfulness-Score nur für die Bilder, die das Modell sowohl in Clean als auch in Corrupted korrekt klassifiziert hat.
Warum das entscheidend für deine Story ist: In Kapitel 5.1/5.2 argumentierst du normativ, dass man Q entzerren (de-confound) muss. Wenn das Modell bei Severity 5 nur noch rät (Performance Drop), ist die Erklärung für diesen Müll-Output natürlich auch diffuser/unfaithful. Die blaue Linie (Unconditioned) wird stark abfallen. Die grüne Linie (Y-Conditioned) zeigt dir aber die wahre Degradierung des Explainers: Wie gut ist die Erklärung noch, wenn das Modell eigentlich noch weiß, was es tut?
Fazit für die Arbeit: "Dieser Plot beweist empirisch die Notwendigkeit von Guideline X aus meinem Protokoll: Wer nicht nach Performance stratifiziert, misst nur den Accuracy-Drop (P), nicht die wahre Erklärungsqualität (Q)."
2. Faithfulness vs. Explanation Drift (Die "Silent Drift" Diagnose)
Zielt direkt auf: 4.3 Mechanismen des Decoupling (Eigentliche Entkopplung vs. Protocol Invalidity)
Du hast in deinen vorherigen Plots das Phänomen "Silent Drift" gefunden (Modell hat Recht, aber Erklärung ändert sich stark). Kapitel 4.3 fragt nach dem Warum. Quantus gibt die Antwort.
Was du zeigst: Nimm die Population der "Silent Drift" Samples (hoher 1−ρ Drift, aber Vorhersage korrekt). Miss für diese spezifischen Samples den Quantus Faithfulness Score auf dem sauberen Bild (D_s) und dem korrumpierten Bild (D_t).
Die zwei möglichen Ausgänge (beide super für die Thesis!):
Szenario A (Protocol Invalidity): Der Faithfulness Score stürzt auf dem korrumpierten Bild massiv ab. Interpretation: Die Metrik (oder der Explainer) kommt mit dem Rauschen (z.B. Fog) nicht klar (OOD-Artefakt). Das Modell denkt noch immer dasselbe, aber der Explainer generiert Müll.
Szenario B (Model Strategy Shift): Der Faithfulness Score bleibt hoch, obwohl die Erklärung völlig anders aussieht als im Clean-Zustand. Interpretation: Der Explainer funktioniert perfekt! Das Modell hat tatsächlich intern seine Strategie geändert, um das Fog-Bild korrekt zu klassifizieren (es schaut jetzt z.B. auf Konturen statt auf Texturen).
Fazit für die Arbeit: Damit füllst du Kapitel 4.3.2 ("Echte Entkopplung") mit Leben und beweist, dass Q kein Proxy für P sein kann.
3. Der OOD-Sanity Check (Degeneration des Explainers)
Zielt direkt auf: 3.3 Familien definieren (Sanity) & 5.2 Bausteine (Sanity Checks)
Was du zeigst: Führe den ModelParameterRandomisation Test (Adebayo et al.) via Quantus durch. Miss, wie ähnlich (z.B. via SSIM oder HOG-Feature-Similarity) die Erklärung des untrainierten Modells zur Erklärung deines trainierten Modells ist. Mache das einmal für Clean-Daten (Severity 0) und einmal für stark korrumpierte Daten (Severity 4/5).
Warum das entscheidend für deine Story ist: Du erwähnst in 4.3.1 "Artefakt-Probleme". Oft degenerieren Saliency-Methoden bei stark verrauschten OOD-Bildern einfach zu Edge-Detectors (sie zeigen nur noch Kanten an, egal was das Modell denkt). Wenn der Sanity-Check auf D_s bestanden wird (Erklärung ändert sich bei Randomisierung), aber auf D_t fehlschlägt (Erklärung bleibt gleich, weil der Explainer nur noch auf den 'Fog' reagiert), hast du den ultimativen Beweis für Protocol Invalidity.
Fazit für die Arbeit: Stützt direkt deine Checkliste (5.3): "Man darf Sanity-Checks nicht nur auf In-Distribution-Daten machen, da Explainer ihr Verhalten OOD fundamental ändern können."
