# YvesBlue Climate Analytics Suite
**Portfolio Climate Risk & Emissions Attribution Platform**

*Barclays Best in Fintech 2022*

---

## Overview

A production-grade suite of climate analytics tools built for institutional equity investors, enabling CFA-style portfolio climate risk assessment across full holdings. The platform ingested equity portfolio identifiers, mapped each holding to multi-source emissions and physical risk data, and surfaced interactive analytics spanning emissions attribution, DCF impact modeling, bad-actor identification, and physical climate risk scoring.

I led all data architecture, analytical and statistical modeling, wireframing, and initial Python development. A team of engineers productionized the platform in Ruby on Rails.

---

## What It Does

**Portfolio Emissions Attribution**
Ingests equity portfolio holdings by identifier (ISIN, ticker, CUSIP), maps each to Scope 1–3 emissions data across multiple third-party providers, and attributes portfolio-level emissions exposure by holding weight, sector, and geography. Enables carbon footprint reporting, emissions intensity benchmarking, and portfolio decarbonization pathway modeling.

**Interactive DCF Climate Risk Modeling**
Integrates physical and transition risk factors into discounted cash flow models at the company level, allowing analysts to stress-test valuations against carbon price scenarios, regulatory transition timelines, and extreme weather exposure.

**Bad Actor Identification & Accountability Screening**
Flags holdings against emissions trajectories, SBTi commitment status, CDP disclosure quality, and physical risk exposure — surfacing companies with material climate liability misaligned with stated commitments.

**Physical Climate Risk Scoring**
Location-level extreme weather risk scoring for real asset-heavy sectors (REITs, utilities, industrials) using Core Ratings physical hazard data cross-referenced with NOAA climate projections and EPA facility-level data.

**Sector & Peer Analytics**
Interactive sector comparison dashboards and peer benchmarking tools enabling portfolio managers to identify relative climate performance across 40+ sectors.

---

## Data Sources Integrated

| Source | Data Type |
|--------|-----------|
| CDP | Company-disclosed Scope 1–3 emissions, targets, governance |
| MSCI ESG Research | ESG ratings, controversy flags, carbon metrics |
| Core Ratings | Physical climate hazard scores (flood, heat, wind, wildfire) |
| EPA | Facility-level GHG emissions (GHGRP), Superfund proximity |
| DOE | Energy consumption and efficiency data |
| NOAA | Historical and projected extreme weather data |
| Company-reported | Direct emissions disclosures, sustainability reports |

---

## My Role

- **Data architecture & wrangling:** Designed multi-source ingestion pipelines harmonizing inconsistent identifiers, units, boundary definitions, and disclosure years across seven data providers
- **Analytical & statistical modeling:** Emissions attribution methodology, portfolio carbon footprint calculations, physical risk scoring model, DCF stress-testing framework
- **Wireframing & UX:** Designed all dashboard layouts and user interaction flows for the analyst and portfolio manager interfaces
- **Initial Python development:** Built the analytical layer in Python (pandas, NumPy, matplotlib, statsmodels); engineering team productionized in Ruby on Rails

---

## Screenshots

*Selected views from the production platform.*

<!-- Screenshots are embedded in this README via the linked repository -->

---

## Stack

**Analytical layer:** Python (pandas, NumPy, matplotlib, statsmodels)  
**Production:** Ruby on Rails  
**Data:** CDP, MSCI, Core Ratings, EPA GHGRP, DOE, NOAA, company-reported  
**Standards:** GHG Protocol, TCFD, SBTi, CDP disclosure framework  

---

## Related Work

- [ERW Scientific MRV & Geochemical Data Analysis](https://github.com/hlangsner1) — BasicDust SPC: PFLOTRAN reactive transport modeling, TOST equivalence testing, solid-phase mass balance analysis for enhanced rock weathering carbon removal verification
- [GCP t-test Statistical Analysis](./GCP1ttest.py) — Example of statistical methodology applied to climate data

---

*Heather Langsner · [heather.langsner@gmail.com](mailto:heather.langsner@gmail.com) · [heatherlangsner.substack.com](https://heatherlangsner.substack.com)*


<img width="728" height="612" alt="Screenshot 2026-01-19 at 9 43 40 PM" src="https://github.com/user-attachments/assets/4ebdad4e-3518-4cf1-a805-9d5e997cbbc5" />
<img width="680" height="540" alt="Screenshot 2026-01-19 at 9 43 55 PM" src="https://github.com/user-attachments/assets/c10aaa7a-bfd6-47c2-9927-64d804577b3b" />
<img width="1036" height="642" alt="Screenshot 2026-01-19 at 9 44 20 PM" src="https://github.com/user-attachments/assets/282dce1a-f1dd-430a-abeb-1ef351b79e32" />
<img width="1033" height="807" alt="Screenshot 2023-10-13 at 11 05 04 AM" src="https://github.com/user-attachments/assets/1ffd87e7-668c-48df-91f9-e954823a3fa2" />
<img width="1036" height="642" alt="Screenshot 2023-10-13 at 11 06 45 AM" src="https://github.com/user-attachments/assets/ca51eeb4-793a-40a6-a645-6ad0e7d67031" />
<img width="1063" height="665" alt="Screenshot 2023-10-13 at 11 06 52 AM" src="https://github.com/user-attachments/assets/01d54032-f482-4887-aee0-cfa54e4738ba" />
<img width="1096" height="829" alt="Screenshot 2023-10-13 at 11 07 01 AM" src="https://github.com/user-attachments/assets/c09e84f6-7177-49f9-8066-2cc29a8d3753" />
<img width="1096" height="799" alt="Screenshot 2023-10-13 at 11 07 09 AM" src="https://github.com/user-attachments/assets/a32cb14d-669e-490f-9531-1c472bb06201" />
<img width="1033" height="807" alt="Screenshot 2023-10-13 at 11 05 04 AM (1)" src="https://github.com/user-attachments/assets/984b5a73-b496-41e9-97fe-079092facbd7" />
