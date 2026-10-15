"""D-101 (eeg-dir3 request, report-only reference): the SparrKULee insert-phone artefact control recording, aligned to its stimulus.

sub-002 ses-shortstories01 listeningActive run-01 (stimulus audiobook_1_artefact; APEX audiobook_1_insertphone_artefact.xml
"Insertphone artefact conditie"). 64 EEG channels in volts, NO filter, re-reference, scaling or clamp. Alignment exactly as
scripts/ext_prepare.py (D-099): Status-channel trigger onsets vs the stimulus trigger file, count repair as brain_pipe, linear
fit of trigger times (clock rate, start). The 1,024 Hz EEG is resampled to 512 Hz (polyphase, anti-aliasing) on the EEG clock
and then linearly interpolated onto the stimulus-time grid k / 512 s, k = 0 .. floor(300.3 s * 512) - 1. EEG sample k <->
audio sample k * 48000 / 512 (= 93.75 k) of the 48 kHz waveform in the npz (stored unchanged).
Output: data/cache/d101_artifact/sparrkulee_artefact/{sparrkulee_artefact.npz, sparrkulee_artefact.json}

usage: python scripts/d101_sparr_artefact.py
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, os.environ.get('ESE_NUM_THREADS', '8'))
import sys, gzip, shutil, json, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(ROOT, 'scripts'))
from ext_prepare import biosemi_trigger_onsets, pulse_onsets, match_triggers, npz_gz, rs, to_stim_clock

RAW = os.path.join(ROOT, 'data/raw/sparrkulee'); REC = 'sub-002/ses-shortstories01/eeg/sub-002_ses-shortstories01_task-listeningActive_run-01'
OUT = os.path.join(ROOT, 'data/cache/d101_artifact/sparrkulee_artefact'); FS_OUT = 512


def main():
    import mne
    os.makedirs(OUT, exist_ok=True); tmp = '/tmp/d101_sparr_artefact.bdf'
    with gzip.open(os.path.join(RAW, REC + '_eeg.bdf.gz'), 'rb') as a, open(tmp, 'wb') as b: shutil.copyfileobj(a, b, 1 << 24)
    raw = mne.io.read_raw_bdf(tmp, preload=True, verbose=False); fs = raw.info['sfreq']; names = raw.ch_names[:64]
    status = raw.get_data(picks=['Status'])[0]; x = raw.get_data(picks=names); os.remove(tmp)            # volts
    trig, fs_t = npz_gz(os.path.join(RAW, 'stimuli/eeg/t_audiobook_1_artefact.npz.gz'))
    audio, fs_a = npz_gz(os.path.join(RAW, 'stimuli/eeg/audiobook_1_artefact.npz.gz'))
    b_on, note = match_triggers(biosemi_trigger_onsets(status), fs, pulse_onsets(trig), fs_t)
    s_on = pulse_onsets(trig); tb, ts = b_on / fs, s_on / fs_t; A = np.c_[ts, np.ones_like(ts)]; (rate, t0), *_ = np.linalg.lstsq(A, tb, rcond=None)
    resid = tb - A @ np.r_[rate, t0]
    y = rs(x, fs, FS_OUT); dur = len(audio) / fs_a; n = int(np.floor(dur * FS_OUT))
    z = to_stim_clock(y, FS_OUT, t0, rate, n).astype(np.float32)
    np.savez(os.path.join(OUT, 'sparrkulee_artefact.npz'), eeg=z, ch_names=np.array(names), fs=FS_OUT, audio=audio, audio_fs=fs_a)
    info = dict(recording=REC, stimulus='audiobook_1_artefact', apex='stimuli/eeg/audiobook_1_insertphone_artefact.xml ("Insertphone artefact conditie")',
                eeg_unit='V', eeg_fs=FS_OUT, eeg_shape=list(z.shape), native_fs=float(fs), processing='none (no filter, no re-reference, no scaling, no clamp); '
                'polyphase 1024 -> 512 Hz on the EEG clock, then linear interpolation onto the stimulus clock',
                eeg_sample0_audio_sample=0.0, audio_samples_per_eeg_sample=fs_a / FS_OUT, stimulus_duration_s=dur,
                stim_start_in_recording_s=float(t0), clock_rate=float(rate), drift_ppm=float((rate - 1) * 1e6), n_triggers=int(len(b_on)),
                trigger_note=note, trigger_resid_max_ms=float(np.abs(resid).max() * 1000), bad_channels=[], bad_note='SparrKULee ships no channel status marks',
                ch_names=names, montage='biosemi64 (10-20 names of the Biosemi 64 cap)', events_onset_s=5.34)
    json.dump(info, open(os.path.join(OUT, 'sparrkulee_artefact.json'), 'w'), indent=1); print(json.dumps({k: v for k, v in info.items() if k != 'ch_names'}), flush=True)


if __name__ == '__main__':
    main()
