from __future__ import annotations

import os
import random
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

import config
import data_paths
import generate

from simulation import (
    add_background_noise, add_gaussian_noise, build_mos_list,
    choose_mos_event_count, cut_mosquito_source, fit_peak,
    gen_mos_pos, get_source_sampling_mode, load_audio,
    prepare_mos, rms, select_mos_source, select_noise_file,
    stable_seed,
)


SEED = 42
SOURCE = config.INDOOR_SOURCE
SPLIT = "train"
TARGET_TYPE = "species_sex"
ENV = "urban"

sr = int(config.SAMPLING_RATE)
duration = float(config.AUDIO_DURATION)
viz_dir = config.viz_dir

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 10,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.linewidth": 0.8
})

# Plot colors
COLOR_BACKGROUND = "#3B5A78"   # slate blue — background traces
COLOR_RAW_SOURCE = "#7B241C"   # maroon — raw mosquito source
COLOR_CROPPED = "#A93226"      # brick red — cropped event
COLOR_SMOOTHED = "#C0663A"     # burnt sienna — boundary smoothing
COLOR_SNR = "#B7950B"          # gold — SNR-scaled event
COLOR_FULL_TRACK = "#117864"   # dark teal — full mosquito track
COLOR_MIXTURE = "#1F3A5F"      # navy — background+mosquito mixture
COLOR_FINAL = "#4A235A"        # deep plum — final signal

HIGHLIGHT_COLOR = "#922B21"    # deep red accent for event spans / markers


def save_plot(fig, name):
    png = os.path.join(viz_dir, f"{name}.png")
    pdf = os.path.join(viz_dir, f"{name}.pdf")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {png}")
    print(f"saved: {pdf}")


background_type = generate.SOURCE_BACKGROUND[SOURCE]

if TARGET_TYPE not in generate.SOURCE_TARGETS[SOURCE]:
    raise ValueError(f"{SOURCE} does not support {TARGET_TYPE}")

if background_type == "gaussian" and ENV != "urban":
    raise ValueError("Outdoor MIRU Gaussian simulation uses ENV='urban'")

source_files = generate.get_source_files(SOURCE, SPLIT)
noise_list = generate.get_noise_files(SPLIT) if background_type == "real" else None
noise_dir = data_paths.raw_noise_dir(SPLIT) if background_type == "real" else ""

allowed_species = config.ENVIRONMENTS[ENV]["mosquito_species"]
mos_list = build_mos_list(source_files, allowed_species, target_type=TARGET_TYPE)

background_name = "gaussian" if background_type == "gaussian" else ENV
seed = generate.job_seed(SEED, TARGET_TYPE, SOURCE, SPLIT, background_name)

random.seed(seed)
np.random.seed(seed)

mos_seed = stable_seed(seed, SOURCE, SPLIT, ENV, "source")
noise_seed = stable_seed(seed, SOURCE, SPLIT, ENV, "noise")
sampling_mode = get_source_sampling_mode(SPLIT)

source_scheduler = select_mos_source(mos_list, mos_seed, sampling_mode=sampling_mode)

if background_type == "real":
    noise_scheduler = select_noise_file(noise_list, noise_seed)
    background, noise_file, subenv, noise_source_start, noise_use_index = add_background_noise(
        noise_dir, ENV, SPLIT, source_name=SOURCE,
        noise_list=noise_list, noise_scheduler=noise_scheduler
    )
else:
    noise_scheduler = None
    noise_file = None
    noise_source_start = None
    background = add_gaussian_noise(ENV, duration, sr)


# Get the number of mosquito events for this clip
requested_num = generate.get_num_simulations(SOURCE, SPLIT)
job_num = generate.get_job_num(
    SOURCE, SPLIT, TARGET_TYPE, ENV,
    source_files, requested_num
)

n_events = choose_mos_event_count(source_scheduler, job_num)
positions = gen_mos_pos(duration, n_events=n_events)


# Make the first mosquito event separately so we can show each step
scheduler_state = source_scheduler.snapshot()
python_state = random.getstate()
numpy_state = np.random.get_state()

chosen = source_scheduler.pick()
event_start, event_end = positions[0]
event_start_sample = int(round(event_start * sr))
requested_samples = max(1, int(round((event_end - event_start) * sr)))

mos_faded, source_start, source_end, source_looped = cut_mosquito_source(
    chosen["path"], requested_samples, sr
)

raw_mosquito = load_audio(chosen["path"], sr)

if len(raw_mosquito) >= requested_samples:
    source_start_sample = int(round(source_start * sr))
    mos_cut = raw_mosquito[source_start_sample:source_start_sample + requested_samples].copy()
else:
    repeats = int(np.ceil(requested_samples / len(raw_mosquito)))
    mos_cut = np.tile(raw_mosquito, repeats)[:requested_samples]

mos_cut = (mos_cut - np.mean(mos_cut, dtype=np.float64)).astype(np.float32)

local_background = np.asarray(
    background[event_start_sample:event_start_sample + len(mos_faded)],
    dtype=np.float64
)

noise_power = float(np.mean(local_background ** 2))
mosquito_power = float(np.mean(np.asarray(mos_faded, dtype=np.float64) ** 2))

if background_type == "gaussian":
    snr_low, snr_high = sorted(map(float, config.GAUSSIAN_BACKGROUND["snr_range"]))
else:
    snr_low, snr_high = sorted(map(float, config.ENVIRONMENTS[ENV]["snr_range"]))

target_snr = random.uniform(snr_low, snr_high)
snr_scale = np.sqrt(noise_power * (10.0 ** (target_snr / 10.0)) / mosquito_power)
mos_scaled = (mos_faded * snr_scale).astype(np.float32)

real_snr = 10.0 * np.log10(
    np.mean(np.asarray(mos_scaled, dtype=np.float64) ** 2) / noise_power
)


# Reset the random states so prepare_mos uses the same first event
source_scheduler.restore(scheduler_state)
random.setstate(python_state)
np.random.set_state(numpy_state)

mosquito_audio, metadata = prepare_mos(
    ENV, mos_list, background, positions,
    source_name=SOURCE, noise_env=ENV,
    background_type=background_type,
    source_scheduler=source_scheduler
)

mix_before_peak = background + mosquito_audio
final_audio, output_gain = fit_peak(mix_before_peak, return_info=True)

first_meta = metadata["audio_labels"][0]

time_clip = np.arange(len(background)) / sr
time_event = np.arange(len(mos_cut)) / sr
time_source = np.arange(len(raw_mosquito)) / sr

mos_cut_track = np.zeros_like(background)
mos_faded_track = np.zeros_like(background)
mos_scaled_track = np.zeros_like(background)

end = min(event_start_sample + len(mos_cut), len(background))
mos_cut_track[event_start_sample:end] = mos_cut[:end - event_start_sample]

end = min(event_start_sample + len(mos_faded), len(background))
mos_faded_track[event_start_sample:end] = mos_faded[:end - event_start_sample]

end = min(event_start_sample + len(mos_scaled), len(background))
mos_scaled_track[event_start_sample:end] = mos_scaled[:end - event_start_sample]


# Plot 1: waveform for each simulation step
wave_steps = [
    (background, f"(a) Prepared {background_name} background", time_clip, COLOR_BACKGROUND),
    (raw_mosquito, f"(b) Raw mosquito source ({chosen['species']}, {chosen['sex']})", time_source, COLOR_RAW_SOURCE),
    (mos_cut, "(c) Cropped mosquito event", time_event, COLOR_CROPPED),
    (mos_faded, "(d) Boundary smoothing", time_event, COLOR_SMOOTHED),
    (mos_scaled, f"(e) SNR scaling ({target_snr:.2f} dB)", time_event, COLOR_SNR),
    (mosquito_audio, f"(f) Full mosquito track ({n_events} events)", time_clip, COLOR_FULL_TRACK),
    (background, "(g) Background before mixing", time_clip, COLOR_BACKGROUND),
    (mix_before_peak, "(h) Background and mosquito mixture", time_clip, COLOR_MIXTURE),
    (final_audio, f"(i) Final signal after peak fitting (gain = {output_gain:.3f})", time_clip, COLOR_FINAL),
]

fig, ax = plt.subplots(len(wave_steps), 1, figsize=(9.0, 10.8))

for i, (signal, title, t, color) in enumerate(wave_steps):
    ax[i].plot(t, signal, linewidth=0.5, color=color)
    ax[i].set_title(title, pad=4)
    ax[i].set_ylabel("Amplitude")
    ax[i].grid(alpha=0.18, linewidth=0.4)
    ax[i].spines["top"].set_visible(False)
    ax[i].spines["right"].set_visible(False)

    if i == 1 and not source_looped:
        ax[i].axvspan(source_start, source_end, alpha=0.15, color=HIGHLIGHT_COLOR)

    if i >= 5:
        for start, end in positions:
            ax[i].axvspan(start, end, alpha=0.08, color=HIGHLIGHT_COLOR)

    if i < len(wave_steps) - 1:
        ax[i].tick_params(axis="x", labelbottom=False)

ax[-1].set_xlabel("Time (s)")
fig.suptitle("Step-by-step audio simulation pipeline", fontsize=13, y=0.985)
fig.subplots_adjust(top=0.94, bottom=0.06, left=0.10, right=0.97, hspace=0.55)

save_plot(fig, "simulation_pipeline_waveform")


# Plot 2: spectrogram for each simulation step
n_fft = 1024
hop_length = 512

spec_steps = [
    (background, f"(a) Prepared {background_name} background"),
    (mos_cut_track, "(b) Cropped first mosquito event"),
    (mos_faded_track, "(c) First event after boundary smoothing"),
    (mos_scaled_track, f"(d) First event after SNR scaling ({target_snr:.2f} dB)"),
    (mosquito_audio, f"(e) Full mosquito track ({n_events} events)"),
    (mix_before_peak, "(f) Background and mosquito mixture"),
    (final_audio, "(g) Final signal after peak fitting"),
]

magnitudes = [
    np.abs(librosa.stft(signal, n_fft=n_fft, hop_length=hop_length, win_length=n_fft))
    for signal, _ in spec_steps
]

shared_ref = max(max(float(np.max(x)) for x in magnitudes), 1e-12)

spec_db = [
    20.0 * np.log10(np.maximum(x, 1e-12) / shared_ref)
    for x in magnitudes
]

fig, ax = plt.subplots(len(spec_steps), 1, figsize=(9.5, 10.5), sharex=True)
image = None

for i, (a, (_, title), spec) in enumerate(zip(ax, spec_steps, spec_db)):
    image = librosa.display.specshow(
        spec, sr=sr, hop_length=hop_length,
        x_axis="time", y_axis="linear",
        cmap="magma", vmin=-80, vmax=0, ax=a
    )

    if i in {1, 2, 3}:
        a.axvline(event_start, linestyle="--", linewidth=0.6, color=HIGHLIGHT_COLOR)
        a.axvline(event_end, linestyle="--", linewidth=0.6, color=HIGHLIGHT_COLOR)

    if i >= 4:
        for start, end in positions:
            a.axvline(start, linestyle="--", linewidth=0.45, color=HIGHLIGHT_COLOR)
            a.axvline(end, linestyle="--", linewidth=0.45, color=HIGHLIGHT_COLOR)

    a.set_ylim(0, min(3000, sr / 2))
    a.set_ylabel("Frequency (Hz)")
    a.set_title(title, pad=4)

    if i < len(spec_steps) - 1:
        a.set_xlabel("")
        a.tick_params(axis="x", labelbottom=False)

ax[-1].set_xlabel("Time (s)")

fig.suptitle(
    "Step-by-step time-frequency representation of the simulation pipeline",
    fontsize=13,
    y=0.985
)

# Leave space on the right for the colorbar
fig.subplots_adjust(
    top=0.94,
    bottom=0.07,
    left=0.10,
    right=0.84,
    hspace=0.40
)

cax = fig.add_axes([0.87, 0.15, 0.018, 0.68])
cbar = fig.colorbar(image, cax=cax)
cbar.set_label("Magnitude (dB)")

save_plot(fig, "simulation_pipeline_spectrogram")


# Plot 3: simple summary figure
summary = [
    (background, "(a) Background", COLOR_BACKGROUND),
    (mosquito_audio, "(b) Mosquito contribution", COLOR_FULL_TRACK),
    (final_audio, "(c) Final simulated signal", COLOR_FINAL),
]

summary_mag = [
    np.abs(librosa.stft(signal, n_fft=n_fft, hop_length=hop_length, win_length=n_fft))
    for signal, _, _ in summary
]

summary_ref = max(max(float(np.max(x)) for x in summary_mag), 1e-12)
summary_db = [20.0 * np.log10(np.maximum(x, 1e-12) / summary_ref) for x in summary_mag]

fig, ax = plt.subplots(3, 2, figsize=(9.0, 7.0))
image = None

for row, ((signal, title, color), spec) in enumerate(zip(summary, summary_db)):
    ax[row, 0].plot(time_clip, signal, linewidth=0.5, color=color)
    ax[row, 0].set_title(title)
    ax[row, 0].set_ylabel("Amplitude")
    ax[row, 0].grid(alpha=0.18, linewidth=0.4)
    ax[row, 0].spines["top"].set_visible(False)
    ax[row, 0].spines["right"].set_visible(False)

    image = librosa.display.specshow(
        spec, sr=sr, hop_length=hop_length,
        x_axis="time", y_axis="linear",
        cmap="magma", vmin=-80, vmax=0, ax=ax[row, 1]
    )

    ax[row, 1].set_ylim(0, min(3000, sr / 2))
    ax[row, 1].set_ylabel("Frequency (Hz)")

    for start, end in positions:
        ax[row, 0].axvspan(start, end, alpha=0.08, color=HIGHLIGHT_COLOR)
        ax[row, 1].axvline(start, linestyle="--", linewidth=0.4, color=HIGHLIGHT_COLOR)
        ax[row, 1].axvline(end, linestyle="--", linewidth=0.4, color=HIGHLIGHT_COLOR)

    if row < 2:
        ax[row, 0].set_xlabel("")
        ax[row, 1].set_xlabel("")

ax[-1, 0].set_xlabel("Time (s)")
ax[-1, 1].set_xlabel("Time (s)")

fig.suptitle("Representative simulated mosquito recording", fontsize=13, y=0.98)

fig.subplots_adjust(
    top=0.91,
    bottom=0.09,
    left=0.09,
    right=0.84,
    hspace=0.42,
    wspace=0.30
)

cax = fig.add_axes([0.87, 0.18, 0.018, 0.62])
cbar = fig.colorbar(image, cax=cax)
cbar.set_label("Magnitude (dB)")

save_plot(fig, "simulation_pipeline_summary")


print()
print("Visualization summary")
print("---------------------")
print(f"Source             : {SOURCE}")
print(f"Split              : {SPLIT}")
print(f"Target             : {TARGET_TYPE}")
print(f"Environment        : {ENV}")
print(f"Background type    : {background_type}")
print(f"Source files       : {len(source_files)}")
print(f"Sampling mode      : {sampling_mode}")
print(f"Events             : {n_events}")
print(f"First species      : {first_meta['species']}")
print(f"First sex          : {first_meta['sex']}")
print(f"First source       : {os.path.basename(first_meta['mos_path'])}")
print(f"First event        : {first_meta['start_time']:.3f} - {first_meta['end_time']:.3f} s")
print(f"Target SNR         : {first_meta['target_snr']:.3f} dB")
print(f"Measured SNR       : {first_meta['real_snr']:.3f} dB")

if noise_file is not None:
    print(f"Noise file         : {os.path.basename(noise_file)}")
    print(f"Noise crop start   : {noise_source_start:.3f} s")
else:
    print("Noise file         : Gaussian")

print(f"Background RMS     : {20 * np.log10(max(rms(background), 1e-12)):.3f} dBFS")
print(f"Peak before fit    : {np.max(np.abs(mix_before_peak)):.6f}")
print(f"Output gain        : {output_gain:.6f}")
print(f"Final peak         : {np.max(np.abs(final_audio)):.6f}")
print(f"Figures            : {viz_dir}")

