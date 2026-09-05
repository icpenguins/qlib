# resolve_factor_attribution Specification

## Purpose
Transforms raw LightGBM booster feature names (Column_i or acronyms) into institutional alpha factor attributions with mathematical formulas and financial/economic explanations.

## Inputs
- eature_names: List[str]: Raw feature names from booster.
- importances: np.ndarray: Feature importance array (gain-based).

## Methodology
1. Queries ields, canonical_names = Alpha158DL.get_feature_config() dynamically from Qlib to obtain canonical names and exact Qlib mathematical expressions.
2. Cross-references factor prefixes (CORD, ROC, CNTP, RSQR, SUMP, RSV, KMID, etc.) against an institutional factor ontology.
3. Formulates behavioral and quantitative explanations (e.g. price-volume accumulation, trend linearity, reversal exhaustion).

## Returns
List[Dict[str, Any]]: Sorted descending by importance gain, containing:
- eature: str
- gain: float
- ormula: str
- description: str
