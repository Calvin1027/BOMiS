"""Read-only final audit; no implicit precedence between competing result roots."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .contracts import DatasetSchema
from .provenance import sha256_file, dataframe_sha256, write_json_new
from .seeds import semantic_seed
from .adapters._external import tree_sha256
from .final_integrity import validate_cell_integrity
from functools import lru_cache


@lru_cache(maxsize=1)
def current_source_hash():
    return tree_sha256(Path(__file__).resolve().parent)


def read(path):
    return json.loads(Path(path).read_text())


def audit_cell(generation, evaluation, dataset_path, experiment_path):
    # Existing detailed assertions must never be disabled by python -O/PYTHONOPTIMIZE.
    if not __debug__:
        raise RuntimeError("deep audit requires Python assertions enabled")
    experiment, dataset = read(experiment_path), read(dataset_path)
    marker, success = read(generation / 'GENERATED.json'), read(evaluation / 'SUCCESS.json')
    name, model, seed = dataset['name'], marker['model'], dataset['split_seed']
    validate_cell_integrity(
        generation, evaluation, dataset=name, model=model, seed=int(seed),
        experiment_hash=sha256_file(experiment_path),
        models_hash=sha256_file(experiment_path.parent / experiment['models_config']),
        source_hash=current_source_hash(),
    )
    assert success['status'] == 'SUCCESS'
    assert marker['dataset'] == success['dataset'] == name
    assert success['model'] == model and success['split_seed'] == seed
    assert success['generation_marker_sha256'] == sha256_file(generation / 'GENERATED.json')
    assert marker['config_sha256']['experiment'] == success['experiment_sha256'] == sha256_file(experiment_path)
    assert marker['config_sha256']['dataset'] == success['dataset_sha256'] == sha256_file(dataset_path)
    assert marker['fallback_used'] is False and marker['backfill_used'] is False
    entry = dataset['splits']['outer_train']
    assert sha256_file(Path(entry['path'])) == entry['sha256']
    train = pd.read_csv(entry['path'])
    schema = DatasetSchema.from_outer_train(train, target=dataset['target'], positive_label=dataset['positive_label'], numerical_columns=tuple(dataset['numerical_columns']), categorical_columns=tuple(dataset['categorical_columns']))
    assert dataframe_sha256(train) == marker['outer_train_sha256']
    path = generation / 'synthetic_minority.parquet'
    assert sha256_file(path) == marker['synthetic_file_sha256']
    frame = pd.read_parquet(path)
    assert {c: str(frame[c].dtype) for c in frame} == schema.dtypes
    validated = schema.restore_and_validate(frame, expected_n=dataset['M_d'])
    assert dataframe_sha256(frame) == dataframe_sha256(validated) == marker['synthetic_sha256'] == success['synthetic_sha256']
    fit, sample = read(generation / 'fit.json')['fit'], read(generation / 'sample.json')['sample']
    assert fit['n_outer_train'] == len(train)
    assert fit['n_minority_train'] == int(schema.binary_target(train).sum())
    assert fit['model_seed'] == semantic_seed(experiment['model_seed_root'], name, model, seed, 'fit')
    assert sample['sampling_seed'] == semantic_seed(experiment['sampling_seed_root'], name, model, seed, 'sample')
    assert fit['model_seed'] != sample['sampling_seed']
    for artifact in fit['checkpoint_artifacts'] + fit['loss_artifacts']:
        artifact_path = generation / 'model' / artifact
        assert artifact_path.is_file() or (artifact_path.is_dir() and any(artifact_path.rglob('*.pt'))), artifact
    raw = sample['raw_draw_count']
    if raw is not None:
        assert raw >= len(frame)
    if sample['rejection_used']:
        accepted = sample['accepted_minority_count']
        assert len(frame) <= accepted <= raw <= sample['raw_draw_cap']
        assert np.isclose(sample['minority_acceptance_rate'], accepted / raw)
    metrics = read(evaluation / 'metrics.json')
    f, d = metrics['fidelity'], metrics['downstream']
    required = ['mean_numerical_ks', 'mean_categorical_tvd', 'mean_category_coverage', 'categorical_tuple_validity', 'alpha_precision', 'beta_recall']
    for key in required:
        assert np.isfinite(f[key]), key
    for key in ('test_mcc', 'test_pr_auc', 'validation_mcc'):
        assert np.isfinite(d[key]), key
    assert -1 <= d['test_mcc'] <= 1 and 0 <= d['test_pr_auc'] <= 1
    assert d['mcc_threshold_selected_on'] == 'outer_validation'
    assert d['classifier_seed'] == semantic_seed(experiment['classifier_seed_root'], name, model, seed, 'classifier')
    assert d['training_policy'] == experiment['classifier']['training_policy']
    assert d['n_augmented_train'] == 2 * dataset['M_d'] and d['n_observed_minority_used'] == 0
    assert f['alpha_precision_metric_seed'] == semantic_seed(experiment['metric_seed_root'], name, model, seed, 'fidelity')
    marginal = pd.read_csv(evaluation / 'marginal_components.csv', dtype={'column': str})
    assert len(marginal) == len(schema.feature_columns)
    assert set(marginal['column']) == set(schema.feature_columns)
    for kind, key in [('numerical', 'mean_numerical_ks'), ('categorical', 'mean_categorical_tvd')]:
        assert np.isclose(marginal.loc[marginal.kind == kind, 'discrepancy'].mean(), f[key])
    pairwise = pd.read_csv(evaluation / 'pairwise_components.csv')
    assert len(pairwise) == len(schema.feature_columns) * (len(schema.feature_columns) - 1) // 2
    undefined = []
    for kind in ('numerical_numerical', 'numerical_categorical', 'categorical_categorical'):
        subset = pairwise.loc[(pairwise.pair_type == kind) & pairwise.defined, 'similarity']
        assert len(subset) == f['defined_pairs_' + kind]
        value = f['mean_pair_similarity_' + kind]
        if len(subset):
            assert np.isfinite(subset).all() and np.isclose(value, subset.mean())
        else:
            assert value is None
            undefined.append(kind)
    return {'dataset': name, 'model': model, 'split_seed': seed, 'status': 'PASS', 'undefined_pair_types': undefined, 'raw_draw_count_unavailable': raw is None}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', action='append', type=Path, required=True)
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--experiment', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--require-complete', action='store_true')
    p.add_argument('--lineage-policy', type=Path)
    args = p.parse_args()
    experiment = read(args.experiment)
    policy = read(args.lineage_policy) if args.lineage_policy else None
    results = []
    for name in experiment['datasets']:
        for model in experiment['models']:
            for seed in experiment['split']['seeds']:
                relative = Path(name) / model / f'split-{seed}'
                roots = [r for r in args.root if (r / 'generation' / relative / 'GENERATED.json').is_file() and (r / 'evaluation' / relative / 'SUCCESS.json').is_file()]
                if not roots:
                    results.append({'dataset': name, 'model': model, 'split_seed': seed, 'status': 'MISSING'})
                    continue
                if len(roots) != 1:
                    results.append({'dataset': name, 'model': model, 'split_seed': seed,
                                    'status': 'FAIL', 'error': 'ambiguous duplicate completed roots'})
                    continue
                root = roots[-1]
                try:
                    result = audit_cell(root / 'generation' / relative, root / 'evaluation' / relative, args.data_root / name / f'split-{seed}' / 'dataset.json', args.experiment)
                    if policy:
                        generated = read(root / 'generation' / relative / 'GENERATED.json')
                        evaluated = read(root / 'evaluation' / relative / 'SUCCESS.json')
                        expected_config = policy['models'][model]['models_config_sha256']
                        assert generated['config_sha256']['models'] == expected_config
                        assert evaluated['benchmark_source_tree_sha256'] == policy['evaluator_source_tree_sha256']
                except Exception as error:
                    import traceback
                    result = {'dataset': name, 'model': model, 'split_seed': seed, 'status': 'FAIL', 'error': repr(error), 'traceback': traceback.format_exc()}
                results.append(result)
                print(name, model, seed, result['status'], flush=True)
    counts = {status: sum(row['status'] == status for row in results) for status in ('PASS', 'FAIL', 'MISSING')}
    write_json_new(args.output, {'counts': counts, 'cells': results,
        'experiment_sha256': sha256_file(args.experiment),
        'benchmark_source_tree_sha256': current_source_hash(),
        'scope': 'artifact integrity, output contract, seeds, metric presence and component aggregation; no classifier refit'})
    print(counts)
    if counts['FAIL'] or (args.require_complete and counts['MISSING']):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
