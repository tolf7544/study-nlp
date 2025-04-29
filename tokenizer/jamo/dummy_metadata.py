class TokenizerMetadata():
    sentence_length: int = 200
    default_special_token
    
    length_nomalizer: {
        truncation: bool,
        padding: bool
    }

    sentence_nomalizer: list(NormalizationMethod)
    
