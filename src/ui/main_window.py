"""
==========================================================
SIGN LANGUAGE RECOGNITION SYSTEM
Main Window
==========================================================
"""

import customtkinter as ctk

from config import *


class MainWindow(ctk.CTk):

    # ======================================================
    # Initialize Window
    # ======================================================

    def __init__(self):

        super().__init__()

        # ==================================================
        # Window Configuration
        # ==================================================

        self.title(WINDOW_TITLE)

        self.geometry(
            f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}"
        )

        self.minsize(
            WINDOW_WIDTH,
            WINDOW_HEIGHT
        )

        self.resizable(
            True,
            True
        )

        # ==================================================
        # Background
        # ==================================================

        self.configure(
            fg_color=BACKGROUND
        )

        # ==================================================
        # Close Event
        # ==================================================

        self.protocol(
            "WM_DELETE_WINDOW",
            self.on_close
        )

        # ==================================================
        # Main Window Grid
        # ==================================================

        # ------------------------------------------
        # Header
        # ------------------------------------------

        self.grid_rowconfigure(
            0,
            weight=0,
            minsize=HEADER_HEIGHT
        )

        # ------------------------------------------
        # Main Content
        # ------------------------------------------

        self.grid_rowconfigure(
            1,
            weight=1
        )

        # ------------------------------------------
        # Control Panel
        # ------------------------------------------

        self.grid_rowconfigure(
            2,
            weight=0,
            minsize=CONTROL_PANEL_HEIGHT
        )

        # ------------------------------------------
        # Footer
        # ------------------------------------------

        self.grid_rowconfigure(
            3,
            weight=0,
            minsize=FOOTER_HEIGHT
        )

        # ------------------------------------------
        # Camera Column
        # ------------------------------------------

        self.grid_columnconfigure(
            0,
            weight=3,
            uniform="main"
        )

        # ------------------------------------------
        # Information Column
        # ------------------------------------------

        self.grid_columnconfigure(
            1,
            weight=1,
            uniform="main"
        )

    # ======================================================
    # Close Window
    # ======================================================

    def on_close(self):

        self.destroy()

    # ======================================================
    # Run Application
    # ======================================================

    def run(self):

        self.mainloop()


# ==========================================================
# Standalone Test
# ==========================================================

if __name__ == "__main__":

    app = MainWindow()

    app.run()