"""Build a clean generation view from an explicit per-model lineage policy."""
import argparse
import json
import os
from pathlib import Path
from .provenance import sha256_file, write_json_new


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8'))


def assemble(experiment_path: Path, data_root: Path, policy_path: Path, output_root: Path) -> Path:
    experiment, policy = read(experiment_path), read(policy_path)
    output_root.mkdir(parents=True, exist_ok=False)
    selections = []
    try:
        for dataset in experiment['datasets']:
            for model in experiment['models']:
                lineage = policy['models'][model]
                for seed in experiment['split']['seeds']:
                    relative = Path(dataset) / model / f'split-{seed}'
                    source = Path(lineage['generation_root']) / 'generation' / relative
                    marker_path = source / 'GENERATED.json'
                    marker = read(marker_path)
                    dataset_path = data_root / dataset / f'split-{seed}' / 'dataset.json'
                    dataset_spec = read(dataset_path)
                    checks = {
                        'status': marker.get('status') == 'GENERATED_UNVERIFIED',
                        'dataset': marker.get('dataset') == dataset,
                        'model': marker.get('model') == model,
                        'M_d': int(marker.get('M_d', -1)) == int(dataset_spec['M_d']),
                        'experiment': marker.get('config_sha256', {}).get('experiment') == sha256_file(experiment_path),
                        'dataset_config': marker.get('config_sha256', {}).get('dataset') == sha256_file(dataset_path),
                        'models_config': marker.get('config_sha256', {}).get('models') == lineage['models_config_sha256'],
                        'fallback': marker.get('fallback_used') is False,
                        'backfill': marker.get('backfill_used') is False,
                        'synthetic': (source / 'synthetic_minority.parquet').is_file(),
                    }
                    if not all(checks.values()):
                        raise RuntimeError(f'lineage failure {relative}: {checks}')
                    if sha256_file(source / 'synthetic_minority.parquet') != marker['synthetic_file_sha256']:
                        raise RuntimeError(f'synthetic hash failure {relative}')
                    destination = output_root / 'generation' / relative
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    os.symlink(source.resolve(), destination, target_is_directory=True)
                    selections.append({'dataset': dataset, 'model': model, 'split_seed': seed, 'source': str(source.resolve()), 'generation_marker_sha256': sha256_file(marker_path), 'models_config_sha256': lineage['models_config_sha256']})
    except Exception as error:
        write_json_new(output_root / 'ASSEMBLY_FAILURE.json', {'status': 'FAILED', 'error': repr(error), 'completed_selections': len(selections)})
        raise
    expected_cells = (
        len(experiment['datasets'])
        * len(experiment['models'])
        * len(experiment['split']['seeds'])
    )
    if len(selections) != expected_cells:
        raise RuntimeError(
            f'assembly coverage mismatch: expected={expected_cells}, found={len(selections)}'
        )
    write_json_new(
        output_root / 'ASSEMBLED.json',
        {
            'status': 'PASS',
            'expected_cells': expected_cells,
            'selections': selections,
            'policy_sha256': sha256_file(policy_path),
        },
    )
    return output_root


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--policy', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    args = parser.parse_args()
    print(assemble(args.experiment, args.data_root, args.policy, args.output_root))
