# LicFlow research artifact

Supplemental materials for *LLM-Based Actionable Representation of License Terms for Compliance Analysis in AI Supply Chains*.

## Recompute the reported results

Install Python 3.11 or later and run these commands from the repository root.

```sh
python -m pip install -r requirements.txt
python reproduce.py
```

The script recomputes metrics from the saved evaluation inputs and outputs. It requires no network connection or API credentials after dependency installation. Its JSON output follows the evaluation sections and includes the assessment results and derived-license selections for the ten branches of the three case studies. Round the results to the precision used in the paper when comparing table values.

| Paper section | Contents |
| --- | --- |
| License modeling quality and cost | `data/modeling/` contains the 60 license identifiers, 60 expert-authored reference entries, 20 checklist-derived OSADL profiles, and 900 generated entries across five configurations and three repetitions. |
| Comparison of entry construction methods | `data/modeling/baseline/` contains 108 whole-entry generation results. `baseline_licenses.json` identifies the 12 licenses used for the paired comparison. |
| Effects on reuse analysis | `data/plans/input/` contains the 69 original plan descriptions. `expert_programs/` contains the reference programs. `expert_entries/` and `generated_entries/` contain the saved results of executing the fixed programs with expert and generated entries, respectively. |
| Natural-language plan interpretation | `data/plans/generated_programs/` contains 207 translations across three repetitions. `program_results/` contains the associated execution results using expert entries. |
| Case studies | `data/cases/` contains the comparison runs for the ten branches discussed in the paper, including the case settings and individual events. The numbered runs use saved Flash entries constructed with reasoning disabled. |

Entry-F1 uses the scoring code under `benchmark/tools/`. The saved list of excluded reference atoms records the effect of the original scorer's check for explicitly named compatibility references in the source text. The OSADL profiles are the evaluation's mappings from the [OSADL License Obligation Checklists](https://www.osadl.org/OSADL-Open-Source-License-Checklists.oss-compliance-checklists.0.html) to the entry schema. The script calculates time and token use from saved measurements and applies the paper's fixed pricing scenario to those token counts. It uses recorded cache hits in the full evaluation and prices input as uncached in the paired comparison.

`data/modeling/reference_links.json` lists the links recorded in the expert entries. Some point to a project or publisher page. Supply the specific license version used by your experiment when constructing a new entry.

## Construct an entry from a supplied license text

The question-based construction implementation is in `scripts/license_modeling_qa/`. The whole-entry generation and review procedure is in `scripts/llm_license_helper.py`. To try a fresh construction, install the additional dependencies and supply a local copy of the source text. Configure the API key through `LICENSE_LLM_API_KEY` for a DeepSeek preset, or through `OPENAI_API_KEY` for an OpenAI model. Optional embedding retrieval uses an OpenAI-compatible embeddings endpoint configured through `OPENAI_API_KEY` and `OPENAI_BASE_URL`.

```sh
python -m pip install -r requirements-modeling.txt
python construct_entry.py MIT /path/to/MIT.txt --model deepseek --output mit.yaml
```

Use `--method legacy` to run whole-entry generation with audit and repair. A fresh construction makes provider calls and may vary with the model snapshot, provider settings, and responses. Use the saved entries for exact metric reproduction. See [construction settings](docs/construction-settings.md) for recorded parameters.

The analysis engine is implemented in `scripts/`. The offline reproduction command reads saved execution results.

## License

Code and original research materials in this repository are licensed under the [Apache License 2.0](LICENSE).

The reference profiles in `data/modeling/osadl/` adapt checklist data from the [Open Source Automation Development Lab (OSADL) eG and contributors](https://www.osadl.org/Access-to-raw-data.oss-compliance-raw-data-access.0.html), licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The profiles map the checklist data to the LicFlow entry schema.
