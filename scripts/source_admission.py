"""High-level admission for source text used to derive feedback and teaching.

Numerical scoring remains defined by native token IDs. Before a coordinator uses
the associated text as an observed attempt, require that it describes those same
actions under the active tokenizer. EOS/special tokens may be omitted in text.
"""


def validate_source_text(engine, generation):
    """Reject contradictory source text without scoring or modifying the engine."""
    if not generation.response_ids or any(
        type(token) is not int or token < 0 for token in generation.response_ids
    ):
        raise ValueError("Source generation must retain nonempty native response IDs")
    if generation.text != engine.tokenizer.decode(
        generation.response_ids, skip_special_tokens=True
    ):
        raise ValueError("Source text differs from tokenizer-decoded native response IDs")
