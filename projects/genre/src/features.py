"""GTZAN feature extraction. Mel-spectrograms + the 58-feature tabular tier.
Heavy imports (librosa) inside functions so this module imports without them."""
def extract_mel(wav_path, **kw):       raise NotImplementedError("TODO librosa.melspectrogram")
def extract_tabular_58(wav_path, **kw): raise NotImplementedError("TODO or load shipped CSV")
