from experiments.mibd.models.token_audit import audit_token_positions


class TinyTokenizer:
    def convert_ids_to_tokens(self, ids):
        vocab = {
            1: "<user>",
            2: "harmful",
            3: "request",
            4: "</user>",
            5: "<assistant>",
        }
        return [vocab[i] for i in ids]


def test_audit_token_positions_maps_relative_positions_and_markers():
    audit = audit_token_positions(
        input_ids=[1, 2, 3, 4, 5],
        tokenizer=TinyTokenizer(),
        relative_positions=[-5, -2, -1],
        user_end_markers=["</user>"],
        assistant_start_markers=["<assistant>"],
    )

    assert audit.sequence_length == 5
    assert audit.relative_to_absolute[-5] == 0
    assert audit.relative_to_absolute[-2] == 3
    assert audit.relative_to_absolute[-1] == 4
    assert audit.user_end_pos == 3
    assert audit.assistant_start_pos == 4

