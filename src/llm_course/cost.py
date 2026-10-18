def estimate_cost_usd(
    input_tokens: int,
    output_tokens: int,
    input_usd_per_million: float,
    output_usd_per_million: float,
) -> float:
    return (
        input_tokens / 1_000_000 * input_usd_per_million
        + output_tokens / 1_000_000 * output_usd_per_million
    )
print(estimate_cost_usd(12_000, 800, 0.0, 0.0))