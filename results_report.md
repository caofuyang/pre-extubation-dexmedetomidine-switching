# v13 primary results (common 48-hour risk scale)

Extubations analysed: 4,630 (943 transition, 3,687 continued propofol)

Primary (overlap weighted): 7.5% vs 3.8%; RD +3.66 pp (+1.55 to +5.70); RR 1.97 (1.37 to 2.68)

E-value 3.34 (CI bound 2.09); propensity-score AUC 0.787


## Estimators on the common scale

 risk_treated  risk_control       rd       rr        estimator    rd_low  rd_high   rr_low  rr_high  reps
     0.062566      0.055872 0.006694 1.119815            Crude -0.012325 0.025437 0.793591 1.504947 300.0
     0.074578      0.037934 0.036644 1.966008 Overlap weighted  0.015514 0.057001 1.372741 2.676201 300.0
     0.090704      0.051755 0.038950 1.752578    g-computation  0.013728 0.062404 1.239095 2.304955 300.0
     0.062566      0.032142 0.030424 1.946541        IPW (ATT)  0.012233 0.046974 1.344157 2.659838 300.0

## Strata

 risk_treated  risk_control        rd       rr    rd_low  rd_high   rr_low  rr_high  reps          stratum  n_treated  events_treated  n_control  events_control
     0.043286      0.018395  0.024891 2.353113  0.005561 0.043310 1.254955 5.029874   200          cardiac        631              24        816              20
     0.122270      0.058833  0.063437 2.078257  0.018901 0.098291 1.274658 2.819057   200 medical_surgical        271              33       2692             172
     0.047234      0.065224 -0.017990 0.724184 -0.088077 0.063166 0.000000 2.120460   200            neuro         41               2        179              14

## Sensitivity

 risk_treated  risk_control       rd       rr                             variant
     0.074578      0.037934 0.036644 1.966008                    primary_48h_risk
     0.075090      0.040077 0.035013 1.873629       excluding_terminal_extubation
     0.064332      0.029375 0.034956 2.189992    outcome_airway_reintubation_only
     0.075412      0.038489 0.036923 1.959318 competing_risk_cumulative_incidence

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