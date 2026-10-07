# Model card: milk rejection risk

Purpose: warn the collector before the lab test. A person still decides.
Data: 62 clean deliveries (synthetic, for teaching).
Inputs: temp_c, hours_since_milking.
Threshold: 0.5
Precision: 0.67
Recall: 0.57
Confusion: tp=8 fp=4 fn=6 tn=44
Limits: small synthetic data; not checked on real milk.
