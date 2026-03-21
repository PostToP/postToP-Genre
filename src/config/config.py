TABLE = {
    "Pop": 0,
    "Rock": 1,
    "Electronic": 2,
    "Hip Hop": 3,
    "Metal": 4,
    # "Folk": 5,
    "Classical": 5,
    "Jazz": 6,
    # "R&B": 7,
    "Country": 7,
    "Reggae": 8,
    # "Latin": 10,
    "Easy Listening": 9,
    "Blues": 10,
    # "New Age": 12,
    "Traditional Music": 11,
}
TABLE_BACK = {v: k for k, v in TABLE.items()}

NUM_LABELS = len(TABLE)

TRANSFORMER_MODEL_NAME = "MIT/ast-finetuned-audioset-10-10-0.4593"
VERSION = "v1.0.0"


AUDIO_LENGTH = 10
SAMPLE_RATE = 16000
