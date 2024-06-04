import argparse
import json

from olmo_eval.tasks.olmes_v0_1.utils import combine_metrics

_parser = argparse.ArgumentParser()
_parser.add_argument(
    "--metrics-file", type=str, nargs="+", required=True, help="Metric files from evaluation"
)
_parser.add_argument(
    "--output-file", type=str, required=False, help="Output file for combined metrics"
)


def main(args: argparse.Namespace):
    metrics = []
    for metrics_file in args.metrics_file:
        with open(metrics_file) as f:
            metrics += json.load(f)["metrics"]

    combined_metrics = combine_metrics(metrics)
    if args.output_file:
        with open(args.output_file, "w") as f:
            json.dump(combined_metrics, f)
    print("** Combined OLMES-v0.1 scores **")
    for d in combined_metrics:
        print(f"{d['task']:13s}: {100 * d['score']:.1f}  ({'MCF' if d['used_mc'] else 'CF'})")
    print("--------------------------")
    avg_score = sum(d["score"] for d in combined_metrics) / len(combined_metrics)
    print(f"{'average':13s}: {100 * avg_score:.1f}")


if __name__ == "__main__":
    main(_parser.parse_args())
