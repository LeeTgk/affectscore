# game/affectscore/signals.py
import os
import time
import json
import threading

MAX_HISTORY = 100  # Rolling window for z-score normalization (samples)


class _SignalCollectorImpl:
    """
    Collects the two-layer signal that conditions music generation.

    LAYER 1 -- Designer intent (authored, deterministic):
      - scene_valence:   Emotional valence of current scene [-1, +1]
      - scene_arousal:   Emotional arousal of current scene [0, 1]
      - arc_position:    Normalized progress through emotional arc [0, 1]
      These are set by the narrative designer in the game script
      and form the PRIMARY emotional axis of the conditioning.

    LAYER 2 -- Player engagement (captured in real time):
      - choice_latency_ms:  Time from menu display to player selection
      - dwell_deviation_s:  Dwell time minus estimated reading time
      - interaction_rate:   Clicks/taps per second in current window
      These MODULATE the designer's intent -- affecting intensity,
      pacing, and texture, not the fundamental emotional direction.

    Player engagement features are z-score normalized using running
    session statistics to account for individual differences.
    """

    def __init__(self):
        self.reset_session()

    def reset_session(self):
        """Call at game start or when resetting player statistics."""
        self._scene_enter_time = time.time()
        self._menu_display_time = None
        self._click_timestamps = []
        self._choice_latency_ms = 0.0

        # Layer 1: Designer intent (set per scene via script calls)
        self._scene_valence = 0.0
        self._scene_arousal = 0.5
        self._arc_position = 0.0
        self._estimated_read_time_s = 5.0  # Default; overridden per scene

        # Running statistics for z-score normalization (Layer 2 only)
        self._engagement_history = {
            "choice_latency_ms": [],
            "dwell_deviation_s": [],
            "interaction_rate": [],
        }

        # Full signal log for post-hoc analysis
        self._log = []

        # Chunk-level event log (chunk_index, condition, mood_text,
        # generation_time_ms). Populated by orchestrator via log_chunk_event().
        self._chunk_log = []

    def set_scene_emotion(self, valence, arousal):
        """Set the designer's intended emotion for the current scene.

        Args:
            valence: Float in [-1.0, 1.0]. Negative = tense/sad, positive = warm/joyful.
            arousal: Float in [0.0, 1.0]. Low = calm/contemplative, high = intense/exciting.
        """
        self._scene_valence = max(-1.0, min(1.0, valence))
        self._scene_arousal = max(0.0, min(1.0, arousal))

    def set_arc_position(self, position):
        """Set the narrative arc position.

        Args:
            position: Float in [0.0, 1.0]. 0.0 = arc onset, 0.5 = rising action, 1.0 = climax.
        """
        self._arc_position = max(0.0, min(1.0, position))

    def on_scene_enter(self, estimated_read_time_s=5.0):
        """Call when a new scene/label begins.

        Args:
            estimated_read_time_s: Approximate time to read the scene's text at average
                speed. Used to compute dwell deviation (actual - expected).
                Estimate as ~(word_count / 200) * 60.
        """
        self._scene_enter_time = time.time()
        self._click_timestamps = []
        self._estimated_read_time_s = estimated_read_time_s

    def on_menu_display(self):
        """Call immediately before showing a choice menu."""
        self._menu_display_time = time.time()

    def on_choice_made(self):
        """Call when the player selects a menu option."""
        if self._menu_display_time is not None:
            self._choice_latency_ms = (
                (time.time() - self._menu_display_time) * 1000.0
            )
            self._menu_display_time = None

    def on_click(self):
        """Call on any player click/tap/keypress during a scene."""
        self._click_timestamps.append(time.time())

    def get_combined_signal(self):
        """Return the full two-layer signal as a list of 6 floats.

        Order: [valence, arousal, arc_position,
                choice_latency_norm, dwell_deviation_norm, interaction_rate_norm]

        Layer 1 values are passed through on their authored scales.
        Layer 2 values are z-score normalized against session history.
        """
        now = time.time()
        raw_dwell = now - self._scene_enter_time
        dwell_deviation = raw_dwell - self._estimated_read_time_s

        # Interaction rate: clicks in last 5 seconds
        window = 5.0
        recent = [t for t in self._click_timestamps if now - t <= window]
        rate = len(recent) / window if window > 0 else 0.0

        raw_engagement = {
            "choice_latency_ms": self._choice_latency_ms,
            "dwell_deviation_s": dwell_deviation,
            "interaction_rate": rate,
        }

        normalized_engagement = {}
        for k, v in raw_engagement.items():
            hist = self._engagement_history[k]
            hist.append(v)
            if len(hist) > MAX_HISTORY:
                hist.pop(0)
            if len(hist) >= 5:
                import statistics
                mu = statistics.mean(hist)
                sd = statistics.stdev(hist)
                normalized_engagement[k] = (
                    (v - mu) / sd if sd > 0.01 else 0.0
                )
            else:
                normalized_engagement[k] = v  # Raw fallback until history builds

        combined_list = [
            self._scene_valence,
            self._scene_arousal,
            self._arc_position,
            normalized_engagement["choice_latency_ms"],
            normalized_engagement["dwell_deviation_s"],
            normalized_engagement["interaction_rate"],
        ]

        self._log.append({
            "timestamp": now,
            "designer_intent": {
                "scene_valence": self._scene_valence,
                "scene_arousal": self._scene_arousal,
                "arc_position": self._arc_position,
            },
            "player_engagement_raw": raw_engagement.copy(),
            "player_engagement_norm": normalized_engagement.copy(),
            "combined": combined_list[:],
        })

        return combined_list

    def log_chunk_event(self, chunk_index, condition, mood_text, generation_time_ms):
        """Record a chunk-generation event for post-hoc analysis.

        Called by the orchestrator after each chunk is generated and queued.

        Args:
            chunk_index:         Integer index of the generated chunk.
            condition:           Evaluation condition string: "A", "B", or "C".
            mood_text:           Mood descriptor string sent to the server,
                                 or "static" for Condition A.
            generation_time_ms:  Client-side wall time for this chunk in ms
                                 (HTTP + generation + WAV encode + file write).
                                 Pass 0 for Condition A.
        """
        self._chunk_log.append({
            "chunk_index": chunk_index,
            "condition": condition,
            "mood_text": mood_text,
            "generation_time_ms": generation_time_ms,
            "timestamp": time.time(),
        })

    def export_log(self, filepath):
        """Export signal log and chunk event log to JSON for analysis."""
        combined = {
            "signal_log": self._log,
            "chunk_log": self._chunk_log,
        }
        with open(filepath, "w") as f:
            json.dump(combined, f, indent=2)
