# Product Funnel & Growth Diagnostics (E-Commerce)

**Diagnosing conversion bottlenecks and sizing revenue impact through product analytics + experiment design**

---

## Project Overview

This project analyzes a 4-step conversion funnel using **90,400 user journeys (Q1 2015)**.  
I quantify where users drop off, identify high-friction segments, validate differences with statistical testing, and propose **prioritized experiments** with sample size + runtime estimates.

**Primary outcome:** End-to-end conversion is **0.50%**, with the largest revenue leakage occurring during **checkout completion**.

---

## Business Problem

Overall conversion has plateaued at **0.50%**, below expectations. Leadership needs answers to:

1. **Where are we losing users?** (biggest drop-off stage)
2. **What segments are underperforming?** (device, cohort, demographics)
3. **What is the revenue upside of fixing bottlenecks?**
4. **What experiments should we run first — and how long will they take?**

---

## Funnel Definition

This funnel is defined using page-level user events:

1. **Home** → entry point  
2. **Search** → engaged browsing  
3. **Payment Page** → checkout start  
4. **Payment Confirmation** → purchase completion  

> Notes:
> - Funnel metrics are computed at the **unique user** level (not event counts).
> - Users can appear in multiple steps, but each step is counted once per user.

---

## Key Findings

### Funnel Performance (Overall)

| Step | Users |
|------|-------|
| Home | 90,400 |
| Search | 45,200 |
| Payment (Checkout Start) | 6,030 |
| Confirmation (Purchase) | 452 |

**Stage Conversion Rates**
- Home → Search: **50.0%**
- Search → Payment: **13.34%**
- Payment → Confirmation: **7.50%**
- End-to-end (Home → Confirmation): **0.50%**

**Main diagnosis:** Discovery is strong, but **checkout completion leaks heavily**.

---

## Device Impact (Largest Opportunity)

Mobile materially outperforms Desktop throughout the funnel:

| Metric | Desktop | Mobile |
|--------|---------|--------|
| End-to-end conversion (Home → Confirm) | 0.25% | 1.00% |
| Checkout completion (Payment → Confirm) | 4.98% | 10.00% |

### Statistical Evidence
Two-proportion z-tests confirm these differences are **highly significant**:
- End-to-end conversion difference: **p ≪ 0.001**
- Checkout completion difference: **p ≪ 0.001**

**Interpretation:** Desktop checkout experience likely introduces major friction.

---

## Experiment Design (A/B Test Proposal)

### P0 Experiment: Desktop Checkout Redesign

**Hypothesis:** Simplifying Desktop checkout (modeled after Mobile patterns) will increase checkout completion.

**Primary Metric:** Payment → Confirmation conversion  
**Guardrails:** error rate, time-to-complete checkout, refund/cancel rate

**Baseline (Desktop):** 4.98%  
**Target lift examples:**
- Conservative: +2.0pp (4.98% → 6.98%)  
- Aggressive: close the gap toward Mobile (≈10%)

This repo includes:
- sample size calculations (power = 80%, α = 0.05)
- estimated runtime based on observed daily entrants into checkout

> Practical note: With low daily traffic at the payment stage, tests may require either
> (a) larger expected lift, (b) longer runtime, or (c) higher-traffic entry-point targeting.

---

## Revenue Impact Model (Scenario-Based)

We quantify opportunity by propagating funnel improvements to incremental purchases:

**Incremental revenue ≈**
`(Incremental users reaching checkout) × (Checkout completion rate) × (Average booking value)`

Scenarios included:
- Improving checkout completion (payment step fix)
- Closing Desktop vs Mobile performance gap
- Improving Search → Checkout Start conversion

> Revenue numbers are scenario-based and depend on traffic assumptions + AOV.

---

## Visualizations

Generated charts (saved to `/outputs`):
- `executive_dashboard.png` — funnel + conversion rates + device comparison + revenue scenarios
- `cohort_heatmap.png` — conversion by signup cohort and device
- `segment_analysis.png` — user segments: bounced, browsers, checkout starters, converters

---

## Project Structure

```text
product-funnel-growth-diagnostics/
│
├── ProductFunnel_Conversion_Diagnostics.ipynb        # End-to-end analysis notebook
├── funnel_growth_analysis.py                       # Script version (generates outputs)
├── README.md
├── experiment_proposal.txt
│
├── data/
│   ├── home_page_table.csv
│   ├── search_page_table.csv
│   ├── payment_page_table.csv
│   ├── payment_confirmation_table.csv
│   └── user_table.csv
│
└── outputs/
    ├── executive_dashboard.png
    ├── cohort_heatmap.png
    ├── segment_analysis.png
    ├── metrics_overall.csv
    ├── metrics_by_device.csv
    ├── segment_summary.csv
    └── cohort_metrics.csv
```

---

## Outputs

- Executive dashboard visualizations
- Funnel metrics exported as CSV files
- Console-based executive summary
- A/B test proposal saved to `experiment_proposal.txt`

---

## Methodology

### 1. Data Preparation & Validation
- Integrated **5 source tables** into a user-level funnel dataset
- Verified absence of duplicates and illogical flows
- Ensured each funnel stage is counted **once per user**

---

### 2. Funnel Analysis
- Calculated stage-level user counts and conversion rates
- Quantified drop-offs at each transition
- Defined practical behavioral segments:
  - **Bounced** — home only  
  - **Browsers** — searched but didn’t start checkout  
  - **Checkout Starters** — reached payment page but didn’t convert  
  - **Converters** — completed purchase  

---

### 3. Multi-Dimensional Analysis
- **Device analysis:** Desktop vs Mobile performance
- **Cohort analysis:** Conversion by signup month
- **Demographics:** Conversion differences by gender
- **Temporal diagnostics:** Weekly patterns and anomaly detection

---

### 4. Statistical Testing
- Two-proportion z-tests for device comparisons
- Wilson confidence intervals for conversion rates
- Power analysis for A/B test sample sizing and duration

---

### 5. Revenue Modeling
- Modeled incremental revenue using funnel propagation
- Compared improvement scenarios (current vs optimized)
- Used conservative assumptions for AOV and traffic

---

## Key Recommendations

### **P0 — Desktop Checkout Redesign**
**Rationale:** Mobile checkout completion is ~2× higher than Desktop (**p < 0.001**)

**Proposed Changes**
- Single-page checkout
- Autofill + smart defaults
- Apple Pay / Google Pay / PayPal
- Guest checkout option

**Expected Impact:** ~$100K–$180K incremental annual revenue  
**Estimated Test Duration:** ~5–6 weeks

---

### **P0 — Checkout Abandonment Investigation**
**Rationale:** 92.5% abandonment at checkout is unusually high

**Actions**
- Add checkout error & latency instrumentation
- Session replay analysis
- 10–15 targeted user interviews

**Expected Impact:** 1.5×–2× improvement in checkout completion

---

### **P1 — Improve Search → Checkout Start Conversion**
**Rationale:** 86.7% of searchers never begin checkout

**Tests**
- Improved ranking & filters
- Price transparency
- Social proof (reviews, popularity signals)

**Expected Impact:** ~45 additional purchases (~$11K incremental revenue per period)

---

## Visualizations

- **Executive Dashboard**
- **Cohort Performance Heatmap**
- **User Segmentation**

---

## Technical Highlights

### Analytics & Statistics
- Funnel diagnostics & segmentation
- Two-proportion hypothesis testing
- Confidence intervals & power analysis
- Experiment design & decision criteria

### Data Engineering
- User-level aggregation from event logs
- Data quality validation
- Reproducible metric exports

### Business Impact
- Revenue opportunity sizing
- Scenario-based ROI analysis
- Prioritized experimentation roadmap

---

## Business Impact

### Immediate Value
- Identified **six-figure annual revenue opportunity**
- Delivered **experiment-ready recommendations**
- Provided statistically grounded justification for roadmap changes

### Strategic Insights
- Mobile UX patterns should inform Desktop redesign
- Checkout is the highest-leverage bottleneck
- Funnel diagnostics can be reused for quarterly reviews

---

## Skills Demonstrated

### Technical
- Python (Pandas, NumPy, SciPy, Matplotlib, Seaborn)
- Statistical testing & experimentation
- Data visualization & storytelling
- ETL & validation pipelines

### Product & Business
- Product funnel analytics
- Growth opportunity sizing
- Executive communication
- Experiment prioritization

---

## Lessons Learned
- Data validation is non-negotiable
- Revenue framing drives prioritization
- Device splits often reveal UX debt
- Small conversion lifts can unlock large value
- Statistical rigor builds stakeholder trust

---

## Contact

**Taroon Ganesh**  
[LinkedIn](https://www.linkedin.com/in/taroon-ganesh-27b83b171/)

_Open to Data Science (Product, Growth, Marketing) and Analytics roles_
