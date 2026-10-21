# v13 primary results

Extubation events: 15,970; analysable contrast: 4,630 (943 transition, 3,687 continued propofol)

48 h reintubation, weighted cumulative incidence: 7.5% vs 3.8%

Risk difference +3.69 pp (95% CI +1.78 to +5.73)

Risk ratio 1.96 (95% CI 1.43 to 2.60)

Effective sample size 867 treated / 2059 control; propensity-score AUC 0.787

E-value for the risk ratio 3.33; for the CI bound 2.20

Bootstrap replicates: 300


## Strata

 risk_treated  risk_control        rd       rr  ess_treated  ess_control  n_treated  n_control  events_treated  events_control    rd_low  rd_high   rr_low  rr_high  reps          stratum      auc
     0.043330      0.018483  0.024847 2.344334   590.534442   706.463289        631        816              24              20  0.005834 0.041764 1.252472 4.402783   200          cardiac 0.685057
     0.125484      0.060361  0.065123 2.078887   270.114656  2164.690252        271       2692              33             172  0.019862 0.102997 1.310807 2.887635   200 medical_surgical 0.661454
     0.047234      0.066340 -0.019106 0.712003    40.493785   149.820135         41        179               2              14 -0.085819 0.063230 0.000000 2.373893   200            neuro 0.725712

## Sensitivity

 risk_treated  risk_control       rd       rr  ess_treated  ess_control  n_treated  n_control  events_treated  events_control                          variant
     0.075412      0.038489 0.036923 1.959318   866.524480  2058.513808        943       3687              59             206                          primary
     0.075937      0.040650 0.035287 1.868082   858.848315  1975.219765        936       3465              59             206    excluding_terminal_extubation
     0.071373      0.037075 0.034298 1.925112   866.524480  2058.513808        943       3687              55             190 outcome_airway_reintubation_only

## Balance

         variable  smd_unweighted  smd_weighted  treated_mean  control_mean
              age       -0.031607     -0.004397     60.927890     61.439110
       vent_hours       -0.283762     -0.000949     28.841216     43.007531
         rass_min       -0.549841      0.011213     -3.761378     -2.902087
         fio2_max        0.569906      0.006691     75.254310     60.162258
         peep_max        0.161392      0.003623      6.470462      6.097023
          lactate        0.026192     -0.022758      2.530937      2.485889
       creatinine       -0.208075     -0.008051      1.109865      1.338683
log_propofol_rate        0.503770      0.005144      3.731266      3.491633
         vaso_any        0.410368      0.049916      0.599152      0.398156
       opioid_any        0.078929      0.078752      0.724284      0.688365
          sbt_any       -0.151872      0.130736      0.247084      0.315161
     delirium_24h       -0.097394      0.142752      0.165429      0.203146