{
    model_path: "EleutherAI/pythia-160m",
    revision: "step140000",
    gpus_needed: 1, // Not used here, but useful for reference
    model_max_length: 2048,
    max_batch_tokens: 20480,
    task: ["arc_challenge", "arc_easy"],
    split: "validation",
    limit: 10,
    num_shots: 0,
    random_subsample_seed: 1234,
    num_recorded_inputs: 3,
    full_output_file: "predictions.jsonl",  // Set to "/results/predictions.jsonl" in beaker jobs
    metrics_file: "metrics.json",  // Set to "/results/metrics.jsonl" in beaker jobs
    gsheet: "OLMo-evals-testing"  // Set to null if no Google Sheet is needed
}
