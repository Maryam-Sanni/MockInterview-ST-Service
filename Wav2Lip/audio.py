import numpy as np
# import tensorflow as tf
import soundfile as sf
from scipy import signal
from scipy.io import wavfile
from hparams import hparams as hp

def load_wav(path, sr):
    wav, source_sr = sf.read(path, dtype='float32')
    if wav.ndim > 1:
        wav = np.mean(wav, axis=1)
    if source_sr != sr:
        gcd = np.gcd(source_sr, sr)
        wav = signal.resample_poly(wav, sr // gcd, source_sr // gcd)
    return wav.astype(np.float32)

def save_wav(wav, path, sr):
    wav *= 32767 / max(0.01, np.max(np.abs(wav)))
    #proposed by @dsmiller
    wavfile.write(path, sr, wav.astype(np.int16))

def save_wavenet_wav(wav, path, sr):
    sf.write(path, wav, sr)

def preemphasis(wav, k, preemphasize=True):
    if preemphasize:
        return signal.lfilter([1, -k], [1], wav)
    return wav

def inv_preemphasis(wav, k, inv_preemphasize=True):
    if inv_preemphasize:
        return signal.lfilter([1], [1, -k], wav)
    return wav

def get_hop_size():
    hop_size = hp.hop_size
    if hop_size is None:
        assert hp.frame_shift_ms is not None
        hop_size = int(hp.frame_shift_ms / 1000 * hp.sample_rate)
    return hop_size

def linearspectrogram(wav):
    D = _stft(preemphasis(wav, hp.preemphasis, hp.preemphasize))
    S = _amp_to_db(np.abs(D)) - hp.ref_level_db
    
    if hp.signal_normalization:
        return _normalize(S)
    return S

def melspectrogram(wav):
    D = _stft(preemphasis(wav, hp.preemphasis, hp.preemphasize))
    S = _amp_to_db(_linear_to_mel(np.abs(D))) - hp.ref_level_db
    
    if hp.signal_normalization:
        return _normalize(S)
    return S

def _lws_processor():
    import lws
    return lws.lws(hp.n_fft, get_hop_size(), fftsize=hp.win_size, mode="speech")

def _stft(y):
    if hp.use_lws:
        return _lws_processor(hp).stft(y).T
    else:
        return _librosa_compatible_stft(
            y,
            n_fft=hp.n_fft,
            hop_length=get_hop_size(),
            win_length=hp.win_size,
        )

def _librosa_compatible_stft(y, n_fft, hop_length, win_length):
    y = np.asarray(y, dtype=np.float32)
    fft_window = signal.get_window('hann', win_length, fftbins=True).astype(np.float32)
    fft_window = _pad_center(fft_window, n_fft)

    pad = n_fft // 2
    pad_mode = 'reflect' if y.shape[0] > 1 else 'edge'
    y = np.pad(y, pad, mode=pad_mode)

    if y.shape[0] < n_fft:
        y = np.pad(y, (0, n_fft - y.shape[0]), mode='constant')

    n_frames = 1 + (y.shape[0] - n_fft) // hop_length
    y_frames = np.lib.stride_tricks.as_strided(
        y,
        shape=(n_frames, n_fft),
        strides=(hop_length * y.strides[0], y.strides[0]),
        writeable=False,
    )
    return np.fft.rfft(fft_window * y_frames, axis=1).T

def _pad_center(data, size):
    if data.shape[0] > size:
        raise ValueError('Target size must be at least input size')
    left = (size - data.shape[0]) // 2
    right = size - data.shape[0] - left
    return np.pad(data, (left, right), mode='constant')

##########################################################
#Those are only correct when using lws!!! (This was messing with Wavenet quality for a long time!)
def num_frames(length, fsize, fshift):
    """Compute number of time frames of spectrogram
    """
    pad = (fsize - fshift)
    if length % fshift == 0:
        M = (length + pad * 2 - fsize) // fshift + 1
    else:
        M = (length + pad * 2 - fsize) // fshift + 2
    return M


def pad_lr(x, fsize, fshift):
    """Compute left and right padding
    """
    M = num_frames(len(x), fsize, fshift)
    pad = (fsize - fshift)
    T = len(x) + 2 * pad
    r = (M - 1) * fshift + fsize - T
    return pad, pad + r
##########################################################
#Librosa correct padding
def librosa_pad_lr(x, fsize, fshift):
    return 0, (x.shape[0] // fshift + 1) * fshift - x.shape[0]

# Conversions
_mel_basis = None

def _linear_to_mel(spectogram):
    global _mel_basis
    if _mel_basis is None:
        _mel_basis = _build_mel_basis()
    return np.dot(_mel_basis, spectogram)

def _build_mel_basis():
    assert hp.fmax <= hp.sample_rate // 2
    return _mel_filter_bank(
        sr=hp.sample_rate,
        n_fft=hp.n_fft,
        n_mels=hp.num_mels,
        fmin=hp.fmin,
        fmax=hp.fmax,
    )

def _mel_filter_bank(sr, n_fft, n_mels, fmin, fmax):
    def hz_to_mel(frequencies):
        scalar = np.isscalar(frequencies)
        frequencies = np.asanyarray(frequencies)
        frequencies = np.atleast_1d(frequencies)
        f_sp = 200.0 / 3
        mels = frequencies / f_sp
        min_log_hz = 1000.0
        min_log_mel = min_log_hz / f_sp
        logstep = np.log(6.4) / 27.0
        log_t = frequencies >= min_log_hz
        mels[log_t] = min_log_mel + np.log(frequencies[log_t] / min_log_hz) / logstep
        return mels[0] if scalar else mels

    def mel_to_hz(mels):
        scalar = np.isscalar(mels)
        mels = np.asanyarray(mels)
        mels = np.atleast_1d(mels)
        f_sp = 200.0 / 3
        frequencies = f_sp * mels
        min_log_hz = 1000.0
        min_log_mel = min_log_hz / f_sp
        logstep = np.log(6.4) / 27.0
        log_t = mels >= min_log_mel
        frequencies[log_t] = min_log_hz * np.exp(logstep * (mels[log_t] - min_log_mel))
        return frequencies[0] if scalar else frequencies

    fft_freqs = np.linspace(0, float(sr) / 2, int(1 + n_fft // 2))
    mel_f = mel_to_hz(np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2))

    fdiff = np.diff(mel_f)
    ramps = np.subtract.outer(mel_f, fft_freqs)

    weights = np.zeros((n_mels, int(1 + n_fft // 2)))
    for i in range(n_mels):
        lower = -ramps[i] / fdiff[i]
        upper = ramps[i + 2] / fdiff[i + 1]
        weights[i] = np.maximum(0, np.minimum(lower, upper))

    enorm = 2.0 / (mel_f[2:n_mels + 2] - mel_f[:n_mels])
    weights *= enorm[:, np.newaxis]
    return weights

def _amp_to_db(x):
    min_level = np.exp(hp.min_level_db / 20 * np.log(10))
    return 20 * np.log10(np.maximum(min_level, x))

def _db_to_amp(x):
    return np.power(10.0, (x) * 0.05)

def _normalize(S):
    if hp.allow_clipping_in_normalization:
        if hp.symmetric_mels:
            return np.clip((2 * hp.max_abs_value) * ((S - hp.min_level_db) / (-hp.min_level_db)) - hp.max_abs_value,
                           -hp.max_abs_value, hp.max_abs_value)
        else:
            return np.clip(hp.max_abs_value * ((S - hp.min_level_db) / (-hp.min_level_db)), 0, hp.max_abs_value)
    
    assert S.max() <= 0 and S.min() - hp.min_level_db >= 0
    if hp.symmetric_mels:
        return (2 * hp.max_abs_value) * ((S - hp.min_level_db) / (-hp.min_level_db)) - hp.max_abs_value
    else:
        return hp.max_abs_value * ((S - hp.min_level_db) / (-hp.min_level_db))

def _denormalize(D):
    if hp.allow_clipping_in_normalization:
        if hp.symmetric_mels:
            return (((np.clip(D, -hp.max_abs_value,
                              hp.max_abs_value) + hp.max_abs_value) * -hp.min_level_db / (2 * hp.max_abs_value))
                    + hp.min_level_db)
        else:
            return ((np.clip(D, 0, hp.max_abs_value) * -hp.min_level_db / hp.max_abs_value) + hp.min_level_db)
    
    if hp.symmetric_mels:
        return (((D + hp.max_abs_value) * -hp.min_level_db / (2 * hp.max_abs_value)) + hp.min_level_db)
    else:
        return ((D * -hp.min_level_db / hp.max_abs_value) + hp.min_level_db)
