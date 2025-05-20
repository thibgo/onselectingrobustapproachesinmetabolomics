### Steps to Reproduce the results of "On selecting robust approaches for learning predictive biomarkers in metabolomics datasets"

1. **Prepare the datasets**:
   - Add the datasets to the `data` directory. (Datasets are currently in the compressed file `mtbls-835-datasets.tar.gz`)

2. **Run computation scripts**:
   - Run `python compute_exp_multivariate.py`. This script takes a significant amount of comuting ressources and time, consider running it on a powerful computer. Can be runned in parallel by changing the value of `n_jobs` in `config/config_local.py`.
   - Run `python compute_exp_univariate.py`.

3. **Generate metadata**:
   - Run the notebook `figures_0_generating_metadf.ipynb`

4. **Generate figures**:
   - Run the 4 notebooks called `figures_*.ipynb`.

## Results files

- `figures/datasets_table.csv`: Contains details about each dataset in the collection and summary results of the experiments.
- `big_perf_df_balancedaccuracy.csv`: Contains more detailed results of the experiments.
