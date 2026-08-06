# game/affectscore/orchestrator.py
import os
import time
import threading
import wave
import struct
import io

# Module-level ref to the currently running orchestrator.
# Ren'Py save/load creates a fresh _StreamingOrchestratorImpl but daemon
# threads from the old instance keep running.  start() uses this to stop
# the old thread before launching a new one, preventing multiple generation
# loops from competing over the same chunk files.
_ACTIVE_ORCHESTRATOR = None


def _va_to_mood_words_local(valence: float, arousal: float) -> str:
    """Local copy of va_to_mood_words() for orchestrator telemetry logging.

    Kept in sync with server/affectscore_server.py and training/train_lora.py.
    Used only for log_chunk_event() mood_text field -- not for conditioning.
    """
    if valence >= 0.3 and arousal >= 0.3:
        primary = "triumphant joyful"
    elif valence >= 0.3 and arousal < -0.3:
        primary = "serene peaceful"
    elif valence < -0.3 and arousal >= 0.3:
        primary = "tense anxious"
    elif valence < -0.3 and arousal < -0.3:
        primary = "melancholic somber"
    else:
        primary = "neutral ambient"
    if abs(arousal) >= 0.6:
        energy = "energetic"
    elif abs(arousal) <= 0.2:
        energy = "calm"
    else:
        energy = ""
    parts = [primary] + ([energy] if energy else []) + ["music"]
    return " ".join(parts)


class _StreamingOrchestratorImpl:
    """
    Manages the double-buffered playback of generated audio chunks.

    Lifecycle per chunk:
      1. Combined signal snapshot -> encoder -> conditioning embedding
      2. Background thread sends embedding to DiT server
      3. Server returns WAV bytes -> saved to temp file
      4. Orchestrator queues file into the next Ren'Py channel
      5. Crossfade from current channel to next channel

    The latency governor tracks generation times and adjusts
    the max_latency_ms hint sent to the server, which the server
    uses to choose its diffusion step count.
    """

    def __init__(self, audio_dir=None):
        self._active_channel = "afs_primary"
        self._standby_channel = "afs_secondary"
        self._chunk_index = 0
        self._running = False
        self._thread = None
        self._last_gen_time_ms = 0
        self._last_chunk_path = None
        self._audio_dir = audio_dir

        # Latency governor state -- set externally via set_config() before start()
        self._latency_budget_ms = 1800
        self._chunk_duration_s = 4.0
        self._crossfade_ms = 300
        self._style_prompt = "ambient orchestral game soundtrack"
        self._gen_time_history = []

        # Lock protecting _active_channel / _standby_channel shared between
        # the background generation thread (reads) and the Ren'Py main-thread
        # closure _do_swap (writes).
        self._channel_lock = threading.Lock()

        # Scheduler: monotonic-clock timestamp of the most recent channel swap.
        # Written by _do_swap() (main thread) under _swap_lock;
        # read by _generation_loop() (background thread) under _swap_lock.
        # Initialized to time.monotonic() so the first deadline is reasonable.
        self._last_swap_time = time.monotonic()
        self._swap_lock = threading.Lock()

        # Playback mode: "A" (static), "B" (base model), "C" (LoRA). Set via set_config().
        self._condition = "C"
        # Path to static WAVs used by Condition A.
        self._static_dir = None
        # audio2audio guidance strength for inter-scene transitions.
        # Applied once per scene change (not every chunk) so cascade is not a risk.
        self._ref_audio_strength = 0.5
        # How long to ask the server to generate (seconds). Must be >= chunk_duration_s.
        # Longer generation gives the diffusion model more harmonic context, improving
        # musical structure in the first chunk_duration_s seconds that are played back.
        # Set via set_config(); defaults to chunk_duration_s (no trimming).
        self._generation_duration_s = self._chunk_duration_s

        # Scene-change trigger event.
        # The generation loop sleeps here after queueing a looping chunk.
        # affectscore_scene_enter() / trigger_scene_change() sets this event
        # to wake the loop and regenerate with the new V-A signal.
        self._trigger_event = threading.Event()

        # Fixed seed passed to the DiT pipeline's manual_seeds parameter.
        # Locks the initial noise across all chunks so instrumentation and
        # timbre stay consistent while V-A conditioning varies emotion.
        # None = random seed per chunk (maximum variation).
        self._generation_seed = None

        # Set to True after the first chunk is successfully queued for playback.
        # Middleware uses this to gate "loading..." display.
        self._first_chunk_ready = False

        # References to companion objects (set by middleware before start())
        self._signals = None
        self._encoder = None
        self._client = None

    def set_config(self, config_dict):
        """Apply AFFECTSCORE_CONFIG values. Call before start()."""
        self._latency_budget_ms = config_dict.get("latency_budget_ms", 1800)
        self._chunk_duration_s = config_dict.get("chunk_duration_s", 4.0)
        # generation_duration_s: how long to ask the server to generate.
        # Defaults to chunk_duration_s when not set (backward compatible -- no trimming).
        self._generation_duration_s = config_dict.get("generation_duration_s", self._chunk_duration_s)
        self._crossfade_ms = config_dict.get("crossfade_ms", 300)
        self._style_prompt = config_dict.get("style_prompt", "ambient orchestral game soundtrack")
        if "audio_dir" in config_dict:
            self._audio_dir = config_dict["audio_dir"]
        if "condition" in config_dict:
            self._condition = config_dict["condition"]
        if "static_dir" in config_dict:
            self._static_dir = config_dict["static_dir"]
        if "ref_audio_strength" in config_dict:
            self._ref_audio_strength = config_dict["ref_audio_strength"]
        if "generation_seed" in config_dict:
            self._generation_seed = config_dict["generation_seed"]

    def set_style_prompt(self, prompt):
        """Update the style prompt for subsequent generation requests.

        Thread-safe: the generation loop reads self._style_prompt each
        iteration, so the new value takes effect on the next chunk boundary.
        """
        self._style_prompt = prompt

    def trigger_scene_change(self):
        """Signal the generation loop to regenerate with the current V-A signal.

        The loop continues playing the current chunk while the new one generates,
        then crossfades. audio2audio continuation is applied once (not on every
        chunk) to avoid cascade spectral decay across multiple scenes.
        """
        self._trigger_event.set()

    @property
    def is_ready(self):
        """True once the first looping chunk is playing."""
        return self._first_chunk_ready

    def set_companions(self, signals, encoder, client):
        """Inject companion objects. Call before start()."""
        self._signals = signals
        self._encoder = encoder
        self._client = client

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop('_thread', None)         # threading.Thread is not picklable
        state.pop('_channel_lock', None)   # threading.Lock is not picklable
        state.pop('_swap_lock', None)      # threading.Lock is not picklable
        state.pop('_trigger_event', None)  # threading.Event is not picklable
        state['_running'] = False          # never restore a "running" flag
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._thread = None
        self._running = False
        self._channel_lock = threading.Lock()
        self._swap_lock = threading.Lock()
        self._trigger_event = threading.Event()
        self._last_swap_time = time.monotonic()
        # Ensure these survive round-trip for attributes added after a save was taken
        if not hasattr(self, '_condition'):
            self._condition = "C"
        if not hasattr(self, '_static_dir'):
            self._static_dir = None
        if not hasattr(self, '_ref_audio_strength'):
            self._ref_audio_strength = 0.5
        if not hasattr(self, '_generation_duration_s'):
            self._generation_duration_s = self._chunk_duration_s
        if not hasattr(self, '_generation_seed'):
            self._generation_seed = None
        if not hasattr(self, '_first_chunk_ready'):
            self._first_chunk_ready = False

    def start(self):
        """Begin the generation loop in a background thread."""
        global _ACTIVE_ORCHESTRATOR
        # Kill any lingering thread from a previous save/load cycle.
        if _ACTIVE_ORCHESTRATOR is not None and _ACTIVE_ORCHESTRATOR is not self:
            _ACTIVE_ORCHESTRATOR._running = False
        _ACTIVE_ORCHESTRATOR = self

        if self._running:
            return
        if self._audio_dir is None:
            raise RuntimeError("audio_dir must be set before starting")
        self._running = True
        self._thread = threading.Thread(
            target=self._generation_loop, daemon=True
        )
        self._thread.start()
        print("[AffectScore] Orchestrator started.")

    def stop(self):
        """Stop the generation loop and fade out audio.

        Called from the Ren'Py main thread (script context), so we stop
        channels directly -- using invoke_in_main_thread would queue the
        stop for the next frame, which fires after `return` already sends
        the player to the main menu, leaving audio playing on custom channels.
        """
        self._running = False
        fadetime = self._crossfade_ms / 1000.0
        try:
            import renpy
            renpy.audio.music.stop(channel=self._active_channel, fadeout=fadetime)
            renpy.audio.music.stop(channel=self._standby_channel, fadeout=fadetime)
        except Exception as e:
            print(f"[AffectScore] Stop: could not fade channels: {e}")
        print("[AffectScore] Orchestrator stopped.")

    def _generation_loop(self):
        """Scene-triggered loop: generate chunk, queue as looping clip, wait for trigger, repeat."""
        chunk_dur = self._chunk_duration_s
        gen_dur = self._generation_duration_s
        crossfade_s = self._crossfade_ms / 1000.0

        _condition_a_playing = None

        while self._running:
            if self._condition == "A":
                arousal = (
                    self._signals.get_combined_signal()[1]
                    if self._signals is not None else 0.0
                )
                src_name = "tense_energetic.wav" if arousal >= 0.5 else "calm_ambient.wav"

                if src_name != _condition_a_playing:
                    _condition_a_playing = src_name
                    src_path = os.path.join(self._static_dir or "", src_name)
                    try:
                        import renpy as _renpy
                        rel = os.path.relpath(src_path, _renpy.config.gamedir).replace('\\', '/')
                        _invoke = _renpy.exports.invoke_in_main_thread

                        def _do_play_loop(path=rel, fade=crossfade_s):
                            if not self._running:
                                return
                            try:
                                _renpy.audio.music.play(
                                    path, channel="afs_primary",
                                    loop=True, fadein=fade,
                                )
                            except Exception as e:
                                print(f"[AffectScore] Condition A play error: {e}")

                        _invoke(_do_play_loop)

                        if self._signals is not None:
                            self._signals.log_chunk_event(
                                chunk_index=self._chunk_index,
                                condition="A",
                                mood_text="static",
                                generation_time_ms=0.0,
                            )
                        self._chunk_index += 1
                    except Exception as e:
                        print(f"[AffectScore] Condition A: {e}")

                # Poll every 0.5 s so _running is checked between waits.
                self._trigger_event.wait(timeout=0.5)
                self._trigger_event.clear()
                continue

            if self._signals is not None:
                signal = self._signals.get_combined_signal()
            else:
                signal = [0.0] * 6

            if self._encoder is not None:
                embedding = self._encoder.encode(signal)
            else:
                embedding = [0.0] * 512

            budget = self._get_adjusted_budget()

            # audio2audio disabled: V-A conditioning provides sufficient musical continuity.
            if self._client is not None:
                wav_bytes, gen_time_ms = self._client.generate_chunk(
                    affect_embedding=embedding,
                    style_prompt=self._style_prompt,
                    chunk_duration_s=gen_dur,
                    max_latency_ms=budget,
                    valence=signal[0],
                    arousal=signal[1],
                    ref_audio_path=None,
                    ref_audio_strength=0.5,
                    seed=self._generation_seed,
                )
            else:
                wav_bytes, gen_time_ms = None, None

            if gen_time_ms is not None:
                self._last_gen_time_ms = gen_time_ms
                self._gen_time_history.append(gen_time_ms)
                if len(self._gen_time_history) > 20:
                    self._gen_time_history.pop(0)

            if wav_bytes is not None:
                if gen_dur > chunk_dur:
                    wav_bytes = self._trim_wav_bytes(wav_bytes, chunk_dur)

                # Bake loop crossfade: blend the clip's tail into its head so
                # Ren'Py's loop repeat sounds seamless instead of cutting.
                wav_bytes = self._bake_loop_crossfade(wav_bytes, chunk_dur)

                chunk_path = os.path.join(
                    self._audio_dir,
                    f"chunk_{self._chunk_index:06d}.wav",
                )
                with open(chunk_path, "wb") as f:
                    f.write(wav_bytes)

                self._last_chunk_path = chunk_path

                # For all chunks after the first: wait for the current loop to
                # reach its natural end before crossfading to the new one.
                # The first chunk plays immediately so there is no silence on start.
                if self._first_chunk_ready:
                    self._wait_for_loop_boundary(chunk_dur, crossfade_s)

                self._queue_playback(chunk_path, crossfade_s, loop=True)

                self._first_chunk_ready = True

                if self._signals is not None:
                    _mood = _va_to_mood_words_local(signal[0], signal[1])
                    self._signals.log_chunk_event(
                        chunk_index=self._chunk_index,
                        condition=self._condition,
                        mood_text=_mood,
                        generation_time_ms=gen_time_ms or 0.0,
                    )

                self._chunk_index += 1
            else:
                self._handle_generation_failure(crossfade_s)

            # Poll in 0.5 s intervals so _running is always checked.
            while self._running:
                if self._trigger_event.wait(timeout=0.5):
                    self._trigger_event.clear()
                    break

    def _wait_for_loop_boundary(self, chunk_dur, crossfade_s=0.3):
        """Sleep until crossfade_s before the next loop boundary.

        Starting the crossfade crossfade_s early means the new track reaches
        full volume exactly when the old loop would have repeated -- the
        transition happens *across* the seam rather than after it.
        """
        with self._swap_lock:
            swap_time = self._last_swap_time
        elapsed = time.monotonic() - swap_time
        if elapsed < 0:
            return
        time_in_loop = elapsed % chunk_dur
        time_to_boundary = chunk_dur - time_in_loop
        wait_s = time_to_boundary - crossfade_s
        if wait_s < 0.05:
            return
        deadline = time.monotonic() + wait_s
        while self._running and time.monotonic() < deadline:
            time.sleep(0.05)

    def _bake_loop_crossfade(self, wav_bytes, chunk_dur):
        """Blend the clip's tail into its head so the loop repeat is seamless.

        The crossfade window is min(2.0 s, 15% of chunk_dur). For a 15-second
        loop that's 2.0 s. Within that window the ending samples ramp from
        100% original -> 0%, mixed with the beginning samples ramping 0% -> 100%.
        When Ren'Py repeats the file, the loop boundary is already smoothed.

        Requires PCM_16 WAV (what the server writes with subtype='PCM_16').
        Falls back silently to unmodified bytes on any parse error.
        """
        import array as _array
        try:
            xfade_s = min(2.0, chunk_dur * 0.15)
            buf_in = io.BytesIO(wav_bytes)
            with wave.open(buf_in, 'rb') as wf:
                sr = wf.getframerate()
                n_ch = wf.getnchannels()
                sw = wf.getsampwidth()
                n_frames = wf.getnframes()
                raw = wf.readframes(n_frames)

            if sw != 2:
                return wav_bytes  # float32 WAV -- unsupported; check server subtype

            xfade_frames = min(int(xfade_s * sr), n_frames // 4)
            samples = _array.array('h', raw)   # signed int16, native endian

            for i in range(xfade_frames):
                t = i / xfade_frames
                tail_base = (n_frames - xfade_frames + i) * n_ch
                head_base = i * n_ch
                for ch in range(n_ch):
                    blended = int(
                        samples[tail_base + ch] * (1.0 - t) +
                        samples[head_base + ch] * t
                    )
                    samples[tail_base + ch] = max(-32768, min(32767, blended))

            buf_out = io.BytesIO()
            with wave.open(buf_out, 'wb') as wf:
                wf.setnchannels(n_ch)
                wf.setsampwidth(sw)
                wf.setframerate(sr)
                wf.writeframes(samples.tobytes())
            return buf_out.getvalue()
        except Exception as e:
            print(f"[AffectScore] Loop crossfade bake failed: {e}")
            return wav_bytes

    def _trim_wav_bytes(self, wav_bytes, target_duration_s):
        """Return WAV bytes trimmed to first target_duration_s seconds."""
        try:
            buf_in = io.BytesIO(wav_bytes)
            with wave.open(buf_in, 'rb') as wf:
                sr = wf.getframerate()
                n_channels = wf.getnchannels()
                sampwidth = wf.getsampwidth()
                n_target = int(target_duration_s * sr)
                frames = wf.readframes(n_target)

            buf_out = io.BytesIO()
            with wave.open(buf_out, 'wb') as wf:
                wf.setnchannels(n_channels)
                wf.setsampwidth(sampwidth)
                wf.setframerate(sr)
                wf.writeframes(frames)
            return buf_out.getvalue()
        except Exception as e:
            print(f"[AffectScore] WAV trim failed: {e}")
            return wav_bytes

    def _queue_playback(self, filepath, crossfade_s, loop=True):
        """Queue a generated WAV into the standby channel, then swap with crossfade.

        loop=True: the file loops indefinitely until the next scene change
        triggers a new generation and crossfade.
        """
        try:
            import renpy
            import os as _os
            _music = renpy.audio.music
            _invoke = renpy.exports.invoke_in_main_thread
            # Use forward slashes -- Ren'Py VFS requires them on all platforms.
            relative_path = _os.path.relpath(filepath, renpy.config.gamedir).replace('\\', '/')

            def _do_swap():
                if not self._running:
                    return  # stop() already called; discard queued swap
                try:
                    with self._channel_lock:
                        active_ch = self._active_channel
                        standby_ch = self._standby_channel
                    _music.stop(
                        channel=active_ch,
                        fadeout=crossfade_s,
                    )
                    _music.play(
                        relative_path,
                        channel=standby_ch,
                        loop=loop,
                        fadein=crossfade_s,
                    )
                    with self._channel_lock:
                        self._active_channel, self._standby_channel = (
                            self._standby_channel,
                            self._active_channel,
                        )
                    with self._swap_lock:
                        self._last_swap_time = time.monotonic()
                except Exception as e:
                    print(f"[AffectScore] _do_swap error: {e}")

            _invoke(_do_swap)
        except Exception as e:
            print(f"[AffectScore] _queue_playback failed: {e}")

    def _handle_generation_failure(self, crossfade_s):
        """Loop the previous chunk to avoid silence when the DiT server fails."""
        if self._last_chunk_path and os.path.exists(self._last_chunk_path):
            last_path = self._last_chunk_path

            try:
                import renpy
                import os as _os
                _music = renpy.audio.music
                _invoke = renpy.exports.invoke_in_main_thread
                relative = _os.path.relpath(last_path, renpy.config.gamedir).replace('\\', '/')

                with self._channel_lock:
                    active_ch = self._active_channel

                def _do_loop():
                    _music.play(
                        relative,
                        channel=active_ch,
                        loop=True,
                        fadein=crossfade_s,
                    )

                _invoke(_do_loop)
            except Exception as e:
                print(f"[AffectScore] Fallback loop failed: {e}")

            print("[AffectScore] Fallback: looping previous chunk.")
        else:
            print("[AffectScore] Fallback: no previous chunk available.")

    def _get_adjusted_budget(self):
        """Latency governor: adjust the budget hint based on recent generation times.

        Tightens the budget when approaching the limit; increases it when there
        is significant headroom, allowing the server to use more diffusion steps.
        """
        base = self._latency_budget_ms
        if len(self._gen_time_history) < 3:
            return base

        import statistics
        avg = statistics.mean(self._gen_time_history[-5:])
        ratio = avg / base

        if ratio > 0.9:
            return int(base * 0.8)
        elif ratio < 0.5:
            return int(base * 1.2)
        else:
            return base
