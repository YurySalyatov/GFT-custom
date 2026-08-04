import argparse
import yaml
import sys
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Update 'all' section in pt_data.yaml with given datasets.")
    parser.add_argument('--datasets', type=str, required=True,
                        help='Comma-separated list of dataset names, e.g., "arnetminer,qian,zbmath"')
    parser.add_argument('--config', type=str, default='config/pt_data.yaml',
                        help='Path to YAML config file (default: config/pt_data.yaml)')
    parser.add_argument('--default_weight', type=float, default=5.0,
                        help='Default weight if dataset has no dedicated section (default: 5.0)')
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config file {config_path} not found.", file=sys.stderr)
        sys.exit(1)

    # Загружаем существующий YAML
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f) or {}

    # Парсим список датасетов
    dataset_names = [name.strip() for name in args.datasets.split(',') if name.strip()]
    if not dataset_names:
        print("Warning: No datasets provided. 'all' section will be empty.", file=sys.stderr)

    # Строим новый словарь для all
    new_all = {}
    for name in dataset_names:
        # Ищем секцию с именем датасета (она может содержать одну пару: имя: вес)
        if name in config and isinstance(config[name], dict):
            # Проверяем, что внутри только одна пара с ключом = name
            if name in config[name]:
                weight = config[name][name]
            else:
                # Если структура не соответствует ожидаемой, берём вес по умолчанию
                weight = args.default_weight
        else:
            weight = args.default_weight
        new_all[name] = weight

    # Обновляем секцию all
    config['all'] = new_all

    # Сохраняем обратно
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    print(f"Updated 'all' section in {config_path} with datasets: {dataset_names}")
    print("New 'all' content:")
    for k, v in new_all.items():
        print(f"  {k}: {v}")

if __name__ == '__main__':
    main()