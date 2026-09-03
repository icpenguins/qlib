# Important details for this project to coding teams
The project has a few end-users, defined in the `End-Users` section who's requirements must be reviewed, analysized, and included as priority 0 for any outcomes. The requirements for this project are divided into 2 parts.

Project Priorty Requirement -1: Subagents must provide a printed acknowledgement that they understand the end-user requirements before they begin working on requests. 

## Part 1: The end-users who validate the financial output of all results
Part 1 defines the end-users of the project. It is priority 0 that these end-users are satified with any updates made. They must be consulted at the after `team-code` finishes their jobs. If the end-users aren't happy, `team-code` must review their feedback and look for methods to improve the work that has been completed.

## Part 2: Implementation Plans and Documentation
Part 2 requires all implementation plans be saved to the `.team-code` folder in the root of the project. Further, all newly created functions must have a markdown file created with the same name that provides a detailed specification of what has been done. Any new work should update any markdown files required to keep a working knowledge of the changes.

## End-Users
### The Profitable Stock Trader
> **Perspective**: *Veteran Discretionary & Quantitative Prop Trader*  
> **Capital at Risk**: Multi-million personal and proprietary capital.  
> **Objective**: Generating consistent alpha, capital preservation, exploiting asymmetric risk/reward setups, and avoiding catastrophic drawdowns.
> **Quote**: "I have reviewed Qlib's architecture. It excels at training regression or ranking algorithms (like LightGBM or ALSTM) on standardized rolling bars. But in real trading, **Qlib operates in a sterile, academic vacuum.** It assumes stationarity across years, ignores the derivatives elephant in the room, has zero awareness of order flow or volume distribution, and naively rebalances portfolios with equal weights at the daily closing price. If you trade real size with raw Qlib today, you will get chopped to pieces by market regimes, front-run on execution, and blown up on earnings gaps."

### Institutional Hedge Fund Manager
> **Perspective**: *Chief Investment Officer (CIO) / Head of Quantitative Research, Multi-Billion Quantitative Hedge Fund*  
> **Mandate**: Deliver double-digit net annualized returns with a Sharpe ratio $> 2.0$, net zero market/factor beta, and zero catastrophic drawdown tolerance.

### Executive Assessment of the Trader's Demands
"The trader's demands reflect genuine frontline intuition. They correctly identify that Qlib in its current state is primarily an **academic tabular ML benchmarking toolkit**, rather than an institutional production platform. The trader's focus on non-stationarity, derivatives flow, and event risk is spot-on.

However, the trader's approach suffers from classic discretionary heuristics: **a lack of cross-sectional factor orthogonalization, unconstrained sizing risks, and naive backtest assumptions.** Below is my formal institutional audit of the trader's demands, followed by the critical gaps the trader completely overlooked."
