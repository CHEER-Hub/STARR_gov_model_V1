# STARR_gov_model_V1
Government funding model (Version 1), including five different allocation designs.

Purpose:
STARR_gov_model_V1 was developed to evaluate how alternative funding allocation designs influence disaster losses, government spending, and household outcomes over time. It captures the effects of different timing and targeting strategies for distributing mitigation funds, while holding the building inventory constant.

The five allocation designs are:

Fixed efficient – Every year, to census tracts with greatest risk reduction efficiency

Fixed uniform – Every year, to all counties equally

Fixed population-based – Every year, to all counties based on population

Event-triggered efficient – Year after disasters, to census tracts with greatest risk reduction efficiency

Event-triggered damage-based – Year after disasters, to counties based on damage

This manual focuses on the structure, inputs, and outputs of the STARR_gov_model_V1 computational framework. It does not include detailed descriptions of the external datasets used in the model. The corresponding data—along with its own data manual—is provided separately in the designated repository (DesignSafe). Users should refer to that data manual for information on file formats, variable definitions, and data sources.

Overview: File Structure
Framework Summary
<img width="432" alt="image" src="https://github.com/user-attachments/assets/b5c5d86b-5072-4a63-b352-eac510ba7675" />

Figure 1. Framework overview

Figure 1 provides an overview of the STARR framework. It consists of the following major components:
1.	Hazard Model – Provides hurricane scenarios as input.
2.	Loss Model – Computes losses across regions, structure types, resistance levels, and households for each hurricane event.
3.	Government Model – Allocates grants and manages acquisition and retrofit pricing under budget constraints.
4.	Homeowner Model – Uses a discrete choice model (DCM) to simulate homeowner decisions on acquisition, retrofitting, and insurance purchases.
5.	Insurance Model – Optimizes insurer behavior by balancing profits, insolvency risk, reinsurance purchases, and market competition using a Cournot-Nash framework.
The model operates over a multi-year time horizon, simulating the interaction of stakeholders over time.
Note: The CGE model (implemented separately in GAMS) is not included in this version of the STARR code.
 Process
Primary calculation process (example):
<img width="458" alt="image" src="https://github.com/user-attachments/assets/40948b01-3565-47af-92b5-7fe45647c8c9" />

Figure 2: Primary process

Figure 2 outlines the main computational process used in this framework. While simplified (some auxiliary functions are not shown), it captures the key model flow.

The primary execution steps are controlled by the following scripts:
•	main_HPC1_20.py — Used to set up parameters and initial settings.
•	main_s_gini_HPC.py — The main driver script that calls key functions to:
1.	Simulate government acquisition decisions
2.	Determine retrofit actions (including self-funded retrofits if applicable)
3.	Allocate grants
4.	Compute insurance purchase decisions
5.	Estimate equilibrium insurance prices
Each of these steps corresponds to components in the inner and outer games illustrated in Figure 1.

