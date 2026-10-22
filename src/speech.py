"""
==========================================================
SIGN LANGUAGE RECOGNITION SYSTEM
Speech Manager
Dynamic Event-Based Speech
==========================================================
"""

import threading
import queue
import pyttsx3

from config import SPEECH_RATE, SPEECH_VOLUME


class SpeechManager:

    # ======================================================
    # INITIALIZE
    # ======================================================

    def __init__(self):

        print("Loading Speech Module...")

        # --------------------------------------------------
        # Speech Queue
        # --------------------------------------------------

        self.speech_queue = queue.Queue()

        # --------------------------------------------------
        # State
        # --------------------------------------------------

        self.running = True

        self.current_speech = None

        self.last_spoken_gesture = None

        # --------------------------------------------------
        # Lock
        # --------------------------------------------------

        self.lock = threading.Lock()

        # --------------------------------------------------
        # Worker Thread
        # --------------------------------------------------

        self.speech_thread = threading.Thread(
            target=self._speech_worker,
            name="SpeechWorker",
            daemon=True
        )

        self.speech_thread.start()

        print("✓ Speech Module Ready")

    # ======================================================
    # CREATE TTS ENGINE
    # ======================================================

    def _create_engine(self):

        try:

            engine = pyttsx3.init()

            engine.setProperty(
                "rate",
                SPEECH_RATE
            )

            engine.setProperty(
                "volume",
                SPEECH_VOLUME
            )

            return engine

        except Exception as error:

            print(
                "TTS Engine Error:",
                error
            )

            return None

    # ======================================================
    # SPEECH WORKER
    # ======================================================

    def _speech_worker(self):

        print(
            "Speech Worker Started"
        )

        while self.running:

            try:

                item = self.speech_queue.get(
                    timeout=0.2
                )

            except queue.Empty:

                continue

            # --------------------------------------------------
            # Stop signal
            # --------------------------------------------------

            if item is None:

                self.speech_queue.task_done()

                break

            text = str(
                item
            ).strip()

            if text == "":

                self.speech_queue.task_done()

                continue

            # --------------------------------------------------
            # Current speech
            # --------------------------------------------------

            with self.lock:

                self.current_speech = text

            engine = None

            try:

                print(
                    f"Speaking: {text}"
                )

                # --------------------------------------------------
                # Create fresh TTS engine for this event
                # --------------------------------------------------

                engine = self._create_engine()

                if engine is None:

                    print(
                        f"Unable to speak: {text}"
                    )

                    continue

                # --------------------------------------------------
                # Speak actual predicted gesture
                # --------------------------------------------------

                engine.say(
                    text
                )

                engine.runAndWait()

                # --------------------------------------------------
                # Successfully completed
                # --------------------------------------------------

                with self.lock:

                    self.last_spoken_gesture = text

                print(
                    f"✓ Spoken: {text}"
                )

            except Exception as error:

                print(
                    f"Speech Error [{text}]:",
                    error
                )

            finally:

                if engine is not None:

                    try:

                        engine.stop()

                    except Exception:
                        pass

                with self.lock:

                    self.current_speech = None

                self.speech_queue.task_done()

        print(
            "Speech Worker Stopped"
        )

    # ======================================================
    # SPEAK CONFIRMED GESTURE
    # ======================================================

    def speak_event(
        self,
        gesture
    ):

        """
        Receives the actual confirmed gesture
        from Predictor/UI.

        No gesture is hard-coded here.

        Example:

            user shows HELLO
            -> gesture = "HELLO"

            user shows YES
            -> gesture = "YES"

            user shows NO
            -> gesture = "NO"

        Whatever confirmed gesture is received
        will be spoken.
        """

        if not self.running:

            return False

        if gesture is None:

            return False

        gesture = str(
            gesture
        ).strip()

        if gesture == "":

            return False

        normalized = gesture.upper()

        # ==================================================
        # DUPLICATE CHECK
        # ==================================================

        with self.lock:

            # --------------------------------------------------
            # Already speaking same gesture
            # --------------------------------------------------

            if (
                self.current_speech is not None
                and
                normalized
                ==
                str(
                    self.current_speech
                ).upper()
            ):

                return False

            # --------------------------------------------------
            # Already spoken same gesture
            #
            # A different confirmed gesture must come
            # before this gesture can be spoken again.
            # --------------------------------------------------

            if (
                self.last_spoken_gesture is not None
                and
                normalized
                ==
                str(
                    self.last_spoken_gesture
                ).upper()
            ):

                return False

        # ==================================================
        # CHECK PENDING QUEUE
        # ==================================================

        try:

            pending_items = list(
                self.speech_queue.queue
            )

            for item in pending_items:

                if item is None:

                    continue

                if (
                    normalized
                    ==
                    str(
                        item
                    ).strip().upper()
                ):

                    return False

        except Exception:
            pass

        # ==================================================
        # ADD ACTUAL GESTURE TO QUEUE
        # ==================================================

        try:

            self.speech_queue.put(
                gesture
            )

            print(
                f"Speech Queued: {gesture}"
            )

            return True

        except Exception as error:

            print(
                "Speech Queue Error:",
                error
            )

            return False

    # ======================================================
    # GENERAL SPEAK
    # ======================================================

    def speak(
        self,
        text
    ):

        """
        General dynamic speech method.

        The text supplied by the caller is spoken.
        No predefined sign is used.
        """

        if text is None:

            return False

        text = str(
            text
        ).strip()

        if text == "":

            return False

        if not self.running:

            return False

        try:

            self.speech_queue.put(
                text
            )

            print(
                f"Speech Queued: {text}"
            )

            return True

        except Exception as error:

            print(
                "Speech Queue Error:",
                error
            )

            return False

    # ======================================================
    # IS SPEAKING
    # ======================================================

    def is_speaking(self):

        with self.lock:

            return (
                self.current_speech
                is not None
            )

    # ======================================================
    # CURRENT SPEECH
    # ======================================================

    def get_current_speech(self):

        with self.lock:

            return self.current_speech

    # ======================================================
    # LAST SPOKEN
    # ======================================================

    def get_last_spoken(self):

        with self.lock:

            return self.last_spoken_gesture

    # ======================================================
    # PENDING COUNT
    # ======================================================

    def get_pending_count(self):

        try:

            return self.speech_queue.qsize()

        except Exception:

            return 0

    # ======================================================
    # CLEAR QUEUE
    # ======================================================

    def clear_queue(self):

        while True:

            try:

                self.speech_queue.get_nowait()

                self.speech_queue.task_done()

            except queue.Empty:

                break

    # ======================================================
    # RESET
    # ======================================================

    def reset(self):

        self.clear_queue()

        with self.lock:

            self.current_speech = None

            self.last_spoken_gesture = None

        print(
            "Speech Manager Reset"
        )

    # ======================================================
    # STOP
    # ======================================================

    def stop(self):

        if not self.running:

            return

        print(
            "Stopping Speech Manager..."
        )

        self.running = False

        self.clear_queue()

        try:

            self.speech_queue.put_nowait(
                None
            )

        except Exception:
            pass

    # ======================================================
    # CLOSE
    # ======================================================

    def close(self):

        self.stop()

        if (
            self.speech_thread is not None
            and
            self.speech_thread.is_alive()
        ):

            self.speech_thread.join(
                timeout=3.0
            )

        with self.lock:

            self.current_speech = None

        print(
            "✓ Speech Module Closed"
        )