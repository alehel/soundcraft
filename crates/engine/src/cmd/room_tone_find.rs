//! Find Room Tone: pick the quietest clean stretch of the recordings as the room tone.
//!
//! Each audio clip is measured in short blocks. A window of the requested length is a candidate
//! when its level is steady, no block in it much louder or much quieter than the window's average
//! (no click, breath or word, and no edge of speech or silence), and it is not digital silence;
//! the quietest candidate wins.

use super::room_tone::{MIN_TONE_SECONDS, describe};
use super::{CommandSpec, bad, bool_or, f64_or, has_tracks, range_param, tracks_param};
use crate::{Engine, Result, cmd};
use serde_json::{Value, json};
use soundcraft_model::{Clip, ClipContent, ClipId, RoomTone, Session, SourceAudio, TrackId, TrackKind};
use soundcraft_time::{Range, Samples};
use std::collections::VecDeque;

const FIND: &str = "edit.room_tone_find";

/// Block length the recordings are measured in, in seconds.
const BLOCK_SECONDS: f64 = 0.01;
/// A window is clean when every block in it carries between 1/this and this many times the
/// window's mean power (±6 dB): steady noise stays well inside it; a click, breath or word, or a
/// part of the window that is speech or silence, does not.
const CLEAN_RATIO: f64 = 4.0;
/// Mean power below which a window counts as digital silence (-100 dBFS), not room tone.
const SILENCE_POWER: f64 = 1e-10;
/// The longest room tone searched for, in seconds.
const MAX_TONE_SECONDS: f64 = 60.0;

pub fn specs() -> Vec<CommandSpec> {
    vec![cmd!(
        "edit.room_tone_find",
        "Find Room Tone",
        ["Edit", "Room Tone"],
        None,
        "{tracks?, start?, end?, length_ms?: 2000, set?: true} — searches the audio clips on the tracks (default: every audio track) between `start` and `end` (default: the whole session; the edit selection is not used, so finding again after a find searches everything) for the quietest stretch of `length_ms` with no click, breath or word in it and no digital silence, then makes it the room tone and selects it (`set: false` only reports it). Clips that are time-stretched are skipped; clip fades are left out. Returns {track, clip, start, end, rms_db, set, room_tone}",
        has_tracks,
        find
    )]
}

/// The best window found so far.
struct Found {
    track: TrackId,
    clip: ClipId,
    /// Window start on the timeline.
    start: Samples,
    /// Window start in the source.
    offset: Samples,
    /// Mean power of the window, after clip gain.
    power: f64,
    tone: RoomTone,
}

fn find(e: &mut Engine, p: &Value) -> Result<Value> {
    let s = e.session();
    let sr = s.sample_rate;
    let block = sr.samples(BLOCK_SECONDS).max(1);
    let length_ms = f64_or(p, "length_ms", 2000.0);
    if !length_ms.is_finite() {
        return Err(bad(FIND, "`length_ms` must be a number"));
    }
    let seconds = (length_ms / 1000.0).clamp(MIN_TONE_SECONDS, MAX_TONE_SECONDS);
    // Whole blocks, so the window is what was measured.
    let blocks = usize::try_from(sr.samples(seconds) / block).unwrap_or(0).max(1);
    let length = Samples::try_from(blocks).unwrap_or(0).saturating_mul(block);
    let tracks = if p.get("tracks").is_some() || p.get("track").is_some() {
        tracks_param(e, FIND, p)?
    } else {
        s.tracks.iter().filter(|t| t.kind == TrackKind::Audio).map(|t| t.id).collect()
    };
    let range = if ["start", "end", "length"].iter().any(|k| p.get(k).is_some()) { range_param(e, FIND, p)? } else { Range::new(0, s.content_end()) };
    let mut best: Option<Found> = None;
    for t in tracks {
        let Some(track) = s.track(t).filter(|tr| tr.kind == TrackKind::Audio) else { continue };
        for clip in track.clips() {
            if let Some(f) = search_clip(s, t, clip, range, block, blocks)
                && best.as_ref().is_none_or(|b| f.power < b.power)
            {
                best = Some(f);
            }
        }
    }
    let Some(found) = best else {
        let secs = length as f64 / f64::from(sr.hz());
        return Err(bad(
            FIND,
            format!(
                "no stretch of {secs:.1} s without clicks, breaths or speech was found in the audio clips there; try a shorter length or another range"
            ),
        ));
    };
    let end = found.start.saturating_add(length);
    let rms_db = 10.0 * found.power.log10();
    let apply = bool_or(p, "set", true);
    if apply {
        let s = e.session_mut();
        s.room_tone = Some(found.tone);
        s.edit.selected_tracks = vec![found.track];
        s.edit.selection = Range::new(found.start, end);
    }
    let s = e.session();
    Ok(json!({
        "track": found.track.0,
        "clip": found.clip.0,
        "start": found.start,
        "end": end,
        "source_offset": found.offset,
        "rms_db": (rms_db * 10.0).round() / 10.0,
        "set": apply,
        "room_tone": describe(s, found.tone),
    }))
}

/// The quietest clean window of `blocks` blocks in `clip` within `range`, fades left out.
fn search_clip(s: &Session, track: TrackId, clip: &Clip, range: Range, block: Samples, blocks: usize) -> Option<Found> {
    let ClipContent::Audio { source, .. } = clip.content else { return None };
    if (clip.stretch - 1.0).abs() > f64::EPSILON || !clip.gain_db.is_finite() {
        return None;
    }
    let audio = s.pool.get(source)?;
    // The part of the clip inside the range and outside its fades, on the timeline.
    let inner = Range::new(clip.start.saturating_add(clip.fade_in.len.max(0)), clip.end().saturating_sub(clip.fade_out.len.max(0)));
    let span = inner.intersect(&range)?;
    let from = clip.source_offset().saturating_add(span.start - clip.start);
    let gain = 10f64.powf(f64::from(clip.gain_db) / 10.0);
    let powers = block_powers(audio, from, span.len(), block, gain);
    let (first, power) = quietest_clean(&powers, blocks)?;
    let skip = Samples::try_from(first).ok()?.checked_mul(block)?;
    let offset = from.checked_add(skip)?;
    let length = Samples::try_from(blocks).ok()?.checked_mul(block)?;
    let frames = u64::try_from(audio.frames()).ok()?;
    let tone = RoomTone { source, offset, length, gain_db: clip.gain_db }.sanitized(frames)?;
    Some(Found { track, clip: clip.id, start: span.start.checked_add(skip)?, offset, power, tone })
}

/// Mean power (all channels, times `gain`) of each whole block of `len` frames from source frame
/// `from`. A block with a non-finite sample, or past the end of the audio, is infinitely loud, so
/// no window holding it is clean.
fn block_powers(audio: &SourceAudio, from: Samples, len: Samples, block: Samples, gain: f64) -> Vec<f64> {
    let (Ok(from), Ok(len), Ok(block)) = (usize::try_from(from), usize::try_from(len), usize::try_from(block)) else { return Vec::new() };
    let channels = audio.buffer.channels.len();
    if channels == 0 || block == 0 {
        return Vec::new();
    }
    (0..len / block)
        .map(|i| {
            let start = from.saturating_add(i.saturating_mul(block));
            let mut sum = 0.0f64;
            for ch in &audio.buffer.channels {
                let Some(frames) = ch.get(start..start.saturating_add(block)) else { return f64::INFINITY };
                sum += frames.iter().map(|x| f64::from(*x) * f64::from(*x)).sum::<f64>();
            }
            let power = sum / (block * channels) as f64 * gain;
            if power.is_finite() { power } else { f64::INFINITY }
        })
        .collect()
}

/// The first block and mean power of the quietest window of `n` blocks whose blocks all stay within
/// [`CLEAN_RATIO`] of the window's mean, and whose mean is above digital silence.
fn quietest_clean(powers: &[f64], n: usize) -> Option<(usize, f64)> {
    if n == 0 || powers.len() < n {
        return None;
    }
    let mut best: Option<(usize, f64)> = None;
    // The window's finite power, and how many of its blocks are infinitely loud.
    let (mut sum, mut loud) = (0.0f64, 0usize);
    // Indices of the window's blocks with the loudest, and the quietest, at the front.
    let (mut loudest, mut quietest): (VecDeque<usize>, VecDeque<usize>) = (VecDeque::new(), VecDeque::new());
    for (i, &p) in powers.iter().enumerate() {
        account(&mut sum, &mut loud, p, true);
        slide(&mut loudest, powers, i, |q| q <= p);
        slide(&mut quietest, powers, i, |q| q >= p);
        let Some(first) = (i + 1).checked_sub(n) else { continue };
        for q in [&mut loudest, &mut quietest] {
            if q.front().is_some_and(|&j| j < first) {
                q.pop_front();
            }
        }
        let mean = if loud > 0 { f64::INFINITY } else { sum / n as f64 };
        let at = |q: &VecDeque<usize>| q.front().and_then(|&j| powers.get(j)).copied().unwrap_or(f64::INFINITY);
        let steady = at(&loudest) <= mean * CLEAN_RATIO && at(&quietest) * CLEAN_RATIO >= mean;
        if mean.is_finite() && mean > SILENCE_POWER && steady && best.is_none_or(|(_, b)| mean < b) {
            best = Some((first, mean));
        }
        account(&mut sum, &mut loud, powers.get(first).copied().unwrap_or(0.0), false);
    }
    best
}

/// Push block `i` onto a monotonic queue, first dropping the blocks at the back it outranks
/// (`outranked` is true for their power).
fn slide(q: &mut VecDeque<usize>, powers: &[f64], i: usize, outranked: impl Fn(f64) -> bool) {
    while q.back().and_then(|&j| powers.get(j)).is_some_and(|&x| outranked(x)) {
        q.pop_back();
    }
    q.push_back(i);
}

/// Add block power `p` to a window's finite `sum` and count of infinitely `loud` blocks as it
/// enters the window, or take it away as it leaves.
fn account(sum: &mut f64, loud: &mut usize, p: f64, entering: bool) {
    match (p.is_finite(), entering) {
        (true, true) => *sum += p,
        // Rounding can leave the running sum a hair below zero over long, quiet stretches.
        (true, false) => *sum = (*sum - p).max(0.0),
        (false, true) => *loud += 1,
        (false, false) => *loud = loud.saturating_sub(1),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use soundcraft_audio_io::{AudioBuffer, FileFormat};
    use soundcraft_model::{ChannelFormat, SourceId};
    use std::sync::Arc;

    const RATE: Samples = 48_000;

    /// Deterministic white noise at `level` (peak amplitude).
    fn noise(len: usize, level: f32, seed: &mut u32) -> Vec<f32> {
        (0..len)
            .map(|_| {
                *seed ^= *seed << 13;
                *seed ^= *seed >> 17;
                *seed ^= *seed << 5;
                (*seed as f32 / u32::MAX as f32 - 0.5) * 2.0 * level
            })
            .collect()
    }

    /// One mono track with one clip over a take made of `parts` (seconds, noise level; level 0 is
    /// digital silence), and the take's source id.
    fn take(parts: &[(f64, f32)]) -> (Engine, TrackId, ClipId, SourceId) {
        let mut seed = 0x1234_5678;
        let mut samples = Vec::new();
        for (secs, level) in parts {
            let n = (secs * RATE as f64) as usize;
            if *level == 0.0 { samples.extend(std::iter::repeat_n(0.0, n)) } else { samples.extend(noise(n, *level, &mut seed)) }
        }
        let mut e = Engine::default();
        let s = e.session_mut();
        let t = s.add_track(TrackKind::Audio, ChannelFormat::Mono, Some("Narration"));
        let src = SourceId(s.alloc());
        let frames = samples.len();
        s.sources.push(soundcraft_model::Source {
            id: src,
            name: "Take".into(),
            path: "Take.wav".into(),
            channels: 1,
            frames: frames as u64,
            sample_rate: 48_000,
            format: FileFormat::Wav,
            time_reference: 0,
            unsaved: false,
        });
        s.pool.insert(src, Arc::new(SourceAudio::new(AudioBuffer { sample_rate: 48_000, channels: vec![samples] })));
        let id = s.new_clip_id();
        if let Some(pl) = s.track_mut(t).and_then(|tr| tr.playlist_mut()) {
            pl.clips.push(Clip::audio(id, "Take", src, 0, 0, frames as Samples));
        }
        (e, t, id, src)
    }

    fn found_start(r: &Value) -> f64 {
        r["start"].as_i64().unwrap() as f64 / RATE as f64
    }

    #[test]
    fn finds_the_quietest_clean_stretch_and_skips_clicks_and_silence() {
        // Speech, a very quiet stretch with a click in it, speech, a clean quiet stretch, and
        // digital silence: only the clean stretch is room tone.
        let (mut e, t, clip, src) = take(&[(1.0, 0.3), (3.0, 0.001), (1.0, 0.3), (3.0, 0.003), (1.0, 0.3), (3.0, 0.0)]);
        let click = (2.5 * RATE as f64) as usize;
        let s = e.session_mut();
        let mut audio = s.pool.get(src).unwrap().buffer.clone();
        audio.channels[0][click] = 0.8;
        s.pool.insert(src, Arc::new(SourceAudio::new(audio)));
        let r = e.execute(FIND, &json!({})).unwrap();
        assert_eq!((r["track"].as_u64(), r["clip"].as_u64(), r["set"].as_bool()), (Some(t.0), Some(clip.0), Some(true)));
        let start = found_start(&r);
        assert!((5.0..=6.0).contains(&start), "the clean stretch is 5–8 s, found {start} s");
        let tone = e.session().room_tone.unwrap();
        assert_eq!((tone.source, tone.length), (src, 2 * RATE));
        assert_eq!(tone.offset, r["start"].as_i64().unwrap(), "clip and source line up at 0");
        assert_eq!(e.session().edit.selection, Range::new(r["start"].as_i64().unwrap(), r["end"].as_i64().unwrap()), "the find is selected");
        let rms = r["rms_db"].as_f64().unwrap();
        assert!((-61.0..-55.0).contains(&rms), "white noise at 0.003 peak is about -55.2 dBFS RMS, got {rms}");
        e.execute("edit.undo", &json!({})).unwrap();
        assert_eq!(e.session().room_tone, None, "undo forgets the find");
    }

    #[test]
    fn a_click_stretch_is_found_when_the_window_fits_beside_the_click() {
        // 3 s of very quiet noise with a click at 2.5 s: a 1 s window fits on either side of it.
        let (mut e, _, _, src) = take(&[(1.0, 0.3), (3.0, 0.001), (1.0, 0.3), (3.0, 0.003)]);
        let s = e.session_mut();
        let mut audio = s.pool.get(src).unwrap().buffer.clone();
        audio.channels[0][(2.5 * RATE as f64) as usize] = 0.8;
        s.pool.insert(src, Arc::new(SourceAudio::new(audio)));
        let r = e.execute(FIND, &json!({"length_ms": 1000})).unwrap();
        let (start, end) = (found_start(&r), r["end"].as_i64().unwrap() as f64 / RATE as f64);
        assert!(start >= 1.0 && end <= 4.0, "the very quiet stretch is 1–4 s, found {start}–{end} s");
        assert!(end <= 2.5 || start > 2.5, "the window holds the click: {start}–{end} s");
    }

    #[test]
    fn report_only_changes_nothing() {
        let (mut e, ..) = take(&[(1.0, 0.3), (3.0, 0.002)]);
        let before = e.session().edit.selection;
        let r = e.execute(FIND, &json!({"set": false})).unwrap();
        assert_eq!(r["set"], false);
        assert_eq!(e.session().room_tone, None);
        assert_eq!(e.session().edit.selection, before);
    }

    #[test]
    fn nothing_long_enough_is_an_error() {
        let (mut e, ..) = take(&[(1.0, 0.3), (1.0, 0.002), (1.0, 0.3), (3.0, 0.0)]);
        let err = e.execute(FIND, &json!({"length_ms": 2000})).unwrap_err();
        assert!(err.to_string().contains("no stretch of 2.0 s"), "{err}");
        assert_eq!(e.session().room_tone, None);
        assert!(e.execute(FIND, &json!({"length_ms": 500})).is_ok(), "a shorter tone fits");
    }

    #[test]
    fn the_range_limits_the_search_but_the_selection_does_not() {
        let (mut e, t, ..) = take(&[(3.0, 0.001), (1.0, 0.3), (3.0, 0.004)]);
        let r = e.execute(FIND, &json!({"start": 4 * RATE, "end": 7 * RATE})).unwrap();
        assert!(found_start(&r) >= 4.0, "the quieter stretch is outside the range");
        let other = e.session_mut().add_track(TrackKind::Audio, ChannelFormat::Mono, None);
        e.session_mut().edit.selected_tracks = vec![other];
        let r = e.execute(FIND, &json!({})).unwrap();
        let start = found_start(&r);
        assert!(
            start <= 1.0 && r["track"].as_u64() == Some(t.0),
            "a plain find searches every track, the whole session: {start} s on {}",
            r["track"]
        );
        assert!(e.execute(FIND, &json!({"tracks": [other.0]})).is_err(), "the empty track has nothing");
    }

    #[test]
    fn clip_gain_counts_and_fades_are_left_out() {
        let (mut e, t, clip, src) = take(&[(3.0, 0.002)]);
        // A second clip of the same quiet stretch, 12 dB lower, after the first.
        let s = e.session_mut();
        let quieter = s.new_clip_id();
        let mut c = Clip::audio(quieter, "Quieter", src, 0, 4 * RATE, 3 * RATE);
        c.gain_db = -12.0;
        c.fade_in.len = RATE / 2;
        if let Some(pl) = s.track_mut(t).and_then(|tr| tr.playlist_mut()) {
            pl.clips.push(c);
            pl.sort();
        }
        let r = e.execute(FIND, &json!({})).unwrap();
        assert_eq!(r["clip"].as_u64(), Some(quieter.0), "the quieter clip wins, not {clip:?}");
        assert!(found_start(&r) >= 4.5, "the fade-in is left out");
        assert_eq!(e.session().room_tone.map(|t| t.gain_db), Some(-12.0));
    }

    #[test]
    fn hostile_audio_does_not_panic_or_win() {
        let (mut e, _, _, src) = take(&[(3.0, 0.002), (3.0, 0.004)]);
        let s = e.session_mut();
        let mut audio = s.pool.get(src).unwrap().buffer.clone();
        for x in audio.channels[0].iter_mut().take(3 * RATE as usize).step_by(1000) {
            *x = f32::NAN;
        }
        s.pool.insert(src, Arc::new(SourceAudio::new(audio)));
        let r = e.execute(FIND, &json!({"length_ms": 1000})).unwrap();
        assert!(r["start"].as_i64().unwrap() > 143_000, "no window may hold a NaN (the last is at sample 143,000)");
        assert!(e.execute(FIND, &json!({"length_ms": 1e300})).is_err(), "clamped to 60 s, longer than the take");
        assert!(e.execute(FIND, &json!({"length_ms": "long"})).is_ok(), "a bad length falls back to the default, searching past the last find");
        let mut empty = Engine::default();
        empty.session_mut().add_track(TrackKind::Audio, ChannelFormat::Mono, None);
        assert!(empty.execute(FIND, &json!({})).is_err());
    }

    #[test]
    fn the_window_search_matches_a_brute_force_search() {
        let mut seed = 99u32;
        // A steady floor with some spread, sparse loud spikes, and a dropout to near silence.
        let powers: Vec<f64> = noise(400, 1.0, &mut seed)
            .into_iter()
            .enumerate()
            .map(|(i, x)| if (200..230).contains(&i) { 1e-6 } else { 1.0 + 2.0 * f64::from(x * x) + if x > 0.97 { 50.0 } else { 0.0 } })
            .collect();
        for n in [1, 2, 7, 40, 400] {
            let brute = (0..=powers.len() - n)
                .filter_map(|i| {
                    let w = &powers[i..i + n];
                    let mean = w.iter().sum::<f64>() / n as f64;
                    let peak = w.iter().copied().fold(0.0, f64::max);
                    let low = w.iter().copied().fold(f64::INFINITY, f64::min);
                    (mean > SILENCE_POWER && peak <= mean * CLEAN_RATIO && low * CLEAN_RATIO >= mean).then_some((i, mean))
                })
                .fold(None, |b: Option<(usize, f64)>, (i, m)| if b.is_none_or(|(_, bm)| m < bm) { Some((i, m)) } else { b });
            let fast = quietest_clean(&powers, n);
            assert!(n == 400 || brute.is_some(), "the fixture must have clean windows of {n}");
            assert_eq!(fast.map(|f| f.0), brute.map(|b| b.0), "n = {n}");
        }
        assert_eq!(quietest_clean(&powers, 401), None);
        assert_eq!(quietest_clean(&[0.0; 10], 5), None, "digital silence");
        assert_eq!(quietest_clean(&[1.0, f64::INFINITY, 1.0], 1).map(|f| f.0), Some(0));
    }
}
