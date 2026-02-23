import csv
import os
import sys
import webbrowser
from hashlib import md5
from io import BytesIO

# Third-party libraries for PDF handling, plotting, and numerical operations
import fitz  # PyMuPDF
import matplotlib.patches as patches
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.path import Path
from numpy import asarray

# GUI Libraries (PySide6) for the application interface
from PySide6 import QtCore, QtGui
from PySide6.QtGui import QImage, QPixmap, QIcon
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QListWidget, QListWidgetItem, QMainWindow, QFileDialog,
    QToolButton, QMessageBox, QCheckBox
)

# FontTools libraries for parsing font data (CFF format)
from fontTools.agl import UV2AGL
from fontTools.cffLib import CFFFontSet
from fontTools.pens.basePen import BasePen


# Function to extract raw font data from a specific page in a PDF
# It looks for a font with a specific name in CFF format and returns its binary buffer
def extract_cff_fonts(pdf_path, page, font_name):
    # Open the PDF file using PyMuPDF
    with fitz.open(pdf_path) as doc:
        # Load the specific page object
        page_obj = doc.load_page(page)
        # Get a list of all fonts referenced on this page
        fonts = page_obj.get_fonts(full=True)

        # Iterate through the found fonts to find the one matching font_name
        for font in fonts:
            # Extract font metadata and the binary content (buffer)
            name, ext, _, buffer = doc.extract_font(font[0])

            # We only care about CFF (Compact Font Format) files that match our target name
            if ext and ext.lower() == "cff" and name == font_name:
                return buffer

        # If the loop finishes without returning, the font was not found
        raise ValueError(f"Font '{font_name}' not found or not in CFF format.")


# Function to scan the entire PDF document structure
# It builds a dictionary mapping page numbers to lists of CFF fonts found on them
def extract_pdf_data(pdf_path):
    font_map = {}  # Dictionary to store results: { page_number: [font_names] }

    with fitz.open(pdf_path) as doc:
        # Loop through every page in the document
        for page in doc:
            cff_names = []
            # Get all fonts on the current page
            fonts = page.get_fonts(full=True)

            for font in fonts:
                name, ext, _, buffer = doc.extract_font(font[0])
                # Filter only CFF fonts
                if ext and ext.lower() == "cff":
                    cff_names.append(name)

            # If we found any CFF fonts on this page, add them to the map
            if cff_names:
                font_map[page.number] = cff_names

    return font_map


# Class representing a custom drawing pen compatible with FontTools
# This pen translates font glyph commands (move, line, curve) into Matplotlib Path codes
class MatplotlibPen(BasePen):
    def __init__(self, glyphset):
        super().__init__(glyphset)
        self.vertices = []  # List of (x, y) coordinates
        self.codes = []  # List of Matplotlib path commands (MOVETO, LINETO, etc.)

    # Handles the 'move to' command (starting a new contour)
    def _moveTo(self, p):
        self.vertices.append(p)
        self.codes.append(Path.MOVETO)

    # Handles the 'line to' command (straight line)
    def _lineTo(self, p):
        self.vertices.append(p)
        self.codes.append(Path.LINETO)

    # Handles cubic Bezier curves
    # p1, p2 are control points, p3 is the end point
    def _curveToOne(self, p1, p2, p3):
        self.vertices.extend([p1, p2, p3])
        # Matplotlib requires CURVE4 repeated 3 times for cubic beziers
        self.codes.extend([Path.CURVE4, Path.CURVE4, Path.CURVE4])

    # Handles closing the shape (connecting last point to start point)
    def _closePath(self):
        # We only close the path if there are vertices present
        if self.vertices:
            self.vertices.append(self.vertices[0])
            self.codes.append(Path.CLOSEPOLY)


# Class representing a pen that generates a string signature of a glyph
# This is used for identification/hashing. It records the sequence of commands
# but ignores specific coordinates if needed (though here we include them)
class SignaturePen(BasePen):
    def __init__(self, glyphset):
        super().__init__(glyphset)
        self.signature = []

    def _moveTo(self, p):
        self.signature.append(f"M{p}")

    def _lineTo(self, p):
        self.signature.append(f"L{p}")

    def _curveToOne(self, p1, p2, p3):
        self.signature.append(f"C{p1}{p2}{p3}")

    def _closePath(self):
        self.signature.append("Z")

    # Returns the complete string representation of the shape
    def get_signature(self):
        return "".join(self.signature)


# Custom widget that integrates Matplotlib figure into the PySide6 GUI
# Used to render the glyph visualization
class GlyphCanvas(FigureCanvas):
    def __init__(self, font):
        self.font = font
        # Create a square figure
        self.fig = Figure(figsize=(4, 4))
        super().__init__(self.fig)
        self.ax = self.fig.add_subplot()

    # Main method to draw a specific glyph onto the canvas
    def draw_glyph(self, glyphset, glyph_name, notdef_max_y, notdef_min_y):
        ax = self.ax
        ax.clear()  # Clear previous drawing
        ax.axis('off')  # Hide X/Y axis ticks and labels

        # Handle case where font or glyph is missing
        if not glyphset or glyph_name not in glyphset:
            ax.text(0.5, 0.5, "No glyph", ha='center', va='center', fontsize=48,
                    color='dimgray', weight='bold', style='italic')
            self.draw()
            return

        glyph = glyphset[glyph_name]
        pen = MatplotlibPen(glyphset)
        glyph.draw(pen)

        if not pen.vertices:
            ax.text(0.5, 0.5, "Empty glyph\n(likely space)", ha='center', va='center',
                    fontsize=30, color='dimgray', weight='bold', style='italic')
            self.draw()
            return

        xs, _ = zip(*pen.vertices)
        min_x, max_x = min(xs), max(xs)
        width = max_x - min_x

        # Calculate font metrics to scale the glyph properly to the canvas
        # We try to get the FontBBox from the font object, otherwise use defaults
        ascent = getattr(self.font, 'FontBBox', [0, 0, 0, 1000])[3]
        descent = getattr(self.font, 'FontBBox', [0, -200, 0, 0])[1]
        font_height = ascent - descent

        scale = 0.8 / font_height
        bottom_margin = 0.05 * scale

        if notdef_min_y is not None and notdef_max_y is not None:
            min_y = (notdef_min_y - descent) * scale + bottom_margin
            max_y = (notdef_max_y - descent) * scale + bottom_margin
            ax.axhline(y=min_y, color='blue', linestyle=':', linewidth=1.5)
            ax.axhline(y=max_y, color='blue', linestyle=':', linewidth=1.5)
        else:
            # Fallback red line if .notdef metrics are missing
            ax.axhline(y=bottom_margin, color='red', linestyle=':', linewidth=1.5)

        vertices = []
        for x, y in pen.vertices:
            x_transformed = (x - min_x - width / 2) * scale
            y_transformed = (y - descent) * scale + bottom_margin
            vertices.append((x_transformed, y_transformed))

        path = Path(vertices, pen.codes)
        patch = patches.PathPatch(path, facecolor='black', lw=1)
        ax.add_patch(patch)

        ax.set_xlim(-0.5, 0.5)
        ax.set_aspect('equal')  # Ensure aspect ratio is preserved (no stretching)
        self.draw()  # Trigger render


# Main Application Window Class
class FontWidget(QMainWindow):
    ICON_SIZE = 64
    CSV_PATH = "glyph_mappings.csv"  # Database file path

    KNOWN_LIGATURES = {
        "IJ": "0132",
        "ij": "0133",
        "OE": "0152",
        "oe": "0153",
        "ff": "fb00",
        "fi": "fb01",
        "fl": "fb02",
        "ffi": "fb03",
        "ffl": "fb04",
        "ft": "fb05",
        "st": "fb06",
        "AE": "00c6",
        "ae": "00e6",
    }

    def __init__(self):
        super().__init__()
        # Initialize internal state variables
        self.pdf_path = None
        self.current_page = None
        self.current_font_name = None
        self.current_font = None
        self.current_glyph_set = None
        self.current_font_glyph_names = []
        self.current_index = 0

        # Dictionaries for data storage
        self.user_glyph_to_char = {}  # Stores current session mappings
        self.font_cache = {}  # Caches extracted font data to avoid re-parsing
        self.db_cache = {}
        self.known_glyph_hashes = set()  # Stores hashes already in the CSV database

        # Setup GUI components
        self._setup_menus()
        self._setup_ui()
        self.clear_ui_state()

        # Window configuration
        self.setMinimumSize(1200, 800)
        self._update_window_title()
        self.statusBar().showMessage("Select PDF to repair")

        self.auto_save_timer = QtCore.QTimer(self)
        self.auto_save_timer.timeout.connect(self.auto_save_interval_triggered)

    def closeEvent(self, event):
        if not self.unsaved_changes:
            event.accept()
            return

        box = QMessageBox(self)
        box.setIcon(QMessageBox.Warning)
        box.setWindowTitle("Unsaved changes")
        box.setText("You have unsaved glyph mappings.")
        box.setInformativeText("Do you want to save before closing?")
        save_btn = box.addButton("Save", QMessageBox.AcceptRole)
        discard_btn = box.addButton("Discard", QMessageBox.DestructiveRole)
        cancel_btn = box.addButton("Cancel", QMessageBox.RejectRole)
        box.setDefaultButton(save_btn)
        box.exec()

        clicked = box.clickedButton()

        if clicked == discard_btn:
            event.accept()
            return

        if clicked == save_btn:
            self.save_to_db()
            event.accept()
            return

        # Cancel OR user closed the dialog with the "X"
        event.ignore()

    def _update_window_title(self):
        app_name = "GlyphRepair"

        pdf_name = os.path.basename(self.pdf_path) if self.pdf_path else "select file to repair"

        self.setWindowTitle(app_name + " - " + pdf_name)

    # Creates the top menu bar (File, Pages, Fonts)
    def _setup_menus(self):
        menubar = self.menuBar()

        # PDF File Menu
        pdf_menu = menubar.addMenu("PDF")

        open_action = pdf_menu.addAction("Open PDF")
        open_action.setIcon(QIcon.fromTheme("folder-open"))
        open_action.triggered.connect(self.open_pdf)

        self.export_action = pdf_menu.addAction("Save PDF")
        self.export_action.setIcon(QIcon.fromTheme("document-save"))
        self.export_action.setEnabled(False)
        self.export_action.triggered.connect(self.save_pdf)

        exit_action = pdf_menu.addAction("Exit")
        exit_action.setIcon(QIcon.fromTheme("window-close"))
        exit_action.triggered.connect(self.close)

        # Create placeholders for dynamic menus (Pages and Fonts)
        self.pages_menu = menubar.addMenu("Pages")
        self.fonts_menu = menubar.addMenu("Fonts")
        self._menu_placeholder(self.pages_menu)
        self._menu_placeholder(self.fonts_menu)

        # --- Settings Menu ---
        settings_menu = menubar.addMenu("Settings")

        self.action_page_mode = settings_menu.addAction("Page Mode Navigation")
        self.action_page_mode.setCheckable(True)
        # Instantly update UI when toggled
        self.action_page_mode.toggled.connect(self.update_navigation_labels)

        self.action_auto_jump_glyph = settings_menu.addAction("Auto-jump to Next Glyph")
        self.action_auto_jump_glyph.setCheckable(True)
        self.action_auto_jump_glyph.setChecked(True)

        self.action_auto_jump_font = settings_menu.addAction("Auto-jump Font at 100%")
        self.action_auto_jump_font.setCheckable(True)
        self.action_auto_jump_font.setChecked(True)

        self.action_auto_save_100 = settings_menu.addAction("Auto-save Database at 100% Font")
        self.action_auto_save_100.setCheckable(True)
        self.action_auto_save_100.setChecked(True)

        self.action_auto_save_timer = settings_menu.addAction("Auto-save every 5 mins")
        self.action_auto_save_timer.setCheckable(True)
        self.action_auto_save_timer.toggled.connect(self.toggle_auto_save_timer)

    def toggle_auto_save_timer(self, checked):
        if checked:
            self.auto_save_timer.start(5 * 60 * 1000)  # 5 minut v milisekundách
            self.statusBar().showMessage("Auto-save timer enabled (5 mins)", 3000)
        else:
            self.auto_save_timer.stop()
            self.statusBar().showMessage("Auto-save timer disabled", 3000)

    def auto_save_interval_triggered(self):
        if self.unsaved_changes:
            self.save_to_db()
            self.statusBar().showMessage("Auto-saved (5 min interval)", 3000)

    # Helper to add a disabled item when a menu is empty
    def _menu_placeholder(self, menu):
        placeholder = menu.addAction("No file loaded")
        placeholder.setIcon(QIcon.fromTheme("sync-error"))
        placeholder.setEnabled(False)

    # initializes all widgets and layouts
    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)

        # Create main widgets
        self.glyph_list = QListWidget()  # Left sidebar list
        self.canvas = GlyphCanvas(None)  # Center plotting area
        self.label = QLabel("Select glyph")  # Info text
        self.user_input = QLineEdit()  # Input for character char
        self.btn_special = QPushButton("Special Chars")  # Link to unicode table
        self.btn_next_unmapped = QPushButton("Next unmapped")
        self.btn_glyph = QPushButton("Save Glyph")
        self.btn_font = QPushButton("Save mappings")
        self.btn_prev_font = QToolButton()
        self.btn_next_font = QToolButton()
        self.lbl_font = QLabel("No font loaded")

        # NEW: Page navigation and stats widgets
        self.btn_prev_page = QToolButton()  # Prev page
        self.btn_next_page = QToolButton()  # Next page
        self.lbl_page = QLabel("Page: -")  # Page info text
        self.lbl_font_stats = QLabel("Font - z -")  # Font stats text
        self.nav_page_widget = QWidget()  # Container for page nav

        # Configure Glyph List Appearance
        self.glyph_list.setIconSize(QtCore.QSize(self.ICON_SIZE, self.ICON_SIZE))
        self.glyph_list.setSpacing(0)
        font = self.glyph_list.font()
        font.setPointSize(20)
        font.setBold(True)
        self.glyph_list.setFont(font)
        self.glyph_list.itemClicked.connect(self.on_glyph_clicked)

        # Configure Info Label
        self.label.setStyleSheet("font-weight: bold; font-size: 32px; color: white;")
        self.label.setAlignment(QtCore.Qt.AlignCenter)

        # Configure Input Field
        self.user_input.setPlaceholderText("Enter char")
        self.user_input.setMaxLength(3)
        self.user_input.returnPressed.connect(self.save_glyph)

        # Connect Button Signals
        self.btn_special.clicked.connect(self.open_special)
        self.btn_glyph.clicked.connect(self.save_glyph)
        self.btn_font.clicked.connect(self.submit_ToUnicode)
        self.btn_prev_font.clicked.connect(self.go_to_prev_font)
        self.btn_next_font.clicked.connect(self.go_to_next_font)
        self.btn_next_unmapped.clicked.connect(self.jump_to_next_unmapped)

        # NEW: Connect page buttons
        self.btn_prev_page.clicked.connect(self.go_to_prev_page)
        self.btn_next_page.clicked.connect(self.go_to_next_page)

        # Configure Navigation Buttons
        self.btn_prev_font.setArrowType(QtCore.Qt.LeftArrow)
        self.btn_next_font.setArrowType(QtCore.Qt.RightArrow)
        self.btn_prev_font.setFixedSize(40, 40)
        self.btn_next_font.setFixedSize(40, 40)
        self.btn_prev_font.setToolTip("Previous font")
        self.btn_next_font.setToolTip("Next font")

        # NEW: Configure Page Navigation Buttons
        self.btn_prev_page.setArrowType(QtCore.Qt.LeftArrow)
        self.btn_next_page.setArrowType(QtCore.Qt.RightArrow)

        # Configure Font Name Label
        lbl_font_style = self.lbl_font.font()
        lbl_font_style.setPointSize(14)
        lbl_font_style.setBold(True)
        self.lbl_font.setFont(lbl_font_style)
        self.lbl_font.setAlignment(QtCore.Qt.AlignCenter)
        self.lbl_font.setMinimumWidth(180)

        # NEW: Configure Page Info Label
        lbl_page_style = self.lbl_page.font()
        lbl_page_style.setPointSize(12)
        lbl_page_style.setBold(True)
        self.lbl_page.setFont(lbl_page_style)
        self.lbl_page.setAlignment(QtCore.Qt.AlignCenter)

        # NEW: Configure Font Stats Label
        self.lbl_font_stats.setAlignment(QtCore.Qt.AlignCenter)
        self.lbl_font_stats.setStyleSheet("color: #aaaaaa; font-size: 12px;")

        # Layout Construction

        # NEW: Page Navigation Row
        nav_page_layout = QHBoxLayout(self.nav_page_widget)
        nav_page_layout.setContentsMargins(0, 0, 0, 0)
        nav_page_layout.addWidget(self.btn_prev_page)
        nav_page_layout.addWidget(self.lbl_page)
        nav_page_layout.addWidget(self.btn_next_page)
        self.nav_page_widget.setVisible(False)  # Hidden by default

        # NEW: Font labels layout (vertical)
        labels_layout = QVBoxLayout()
        labels_layout.addWidget(self.lbl_font)
        labels_layout.addWidget(self.lbl_font_stats)
        labels_layout.setSpacing(2)

        # Navigation Row
        nav = QHBoxLayout()
        nav.addWidget(self.btn_prev_font)
        nav.addLayout(labels_layout)  # NEW: using vertical layout instead of just lbl_font
        nav.addWidget(self.btn_next_font)
        nav.setSpacing(6)

        # Input Row
        inputs = QHBoxLayout()
        inputs.addWidget(self.user_input)
        inputs.addWidget(self.btn_next_unmapped)
        inputs.addWidget(self.btn_special)
        inputs.addWidget(self.btn_glyph)
        inputs.addWidget(self.btn_font)

        # Right Column (Canvas + Controls)
        right = QVBoxLayout()
        right.addWidget(self.nav_page_widget)
        right.addLayout(nav)
        right.addWidget(self.canvas)
        right.addWidget(self.label)
        right.addLayout(inputs)

        # Main Layout (List + Right Column)
        main = QHBoxLayout(central)
        main.addWidget(self.glyph_list, 2)  # Ratio 2:5
        main.addLayout(right, 5)

    # Resets the UI elements when no font is loaded
    def clear_ui_state(self):
        self.glyph_list.clear()
        self.label.setText("No font loaded")
        self.canvas.draw_glyph(None, None, None, None)
        self.user_input.clear()
        self.user_input.setEnabled(False)
        self.btn_glyph.setEnabled(False)
        self.btn_font.setEnabled(False)
        # Reset navigation labels
        self.lbl_font.setText("No font loaded")
        self.lbl_page.setText("Page: -")
        self.lbl_font_stats.setText("Font - z -")
        self.nav_page_widget.setVisible(False)
        self.unsaved_changes = False
        self._update_window_title()

    # Font Navigation Logic
    def go_to_prev_font(self):
        self._navigate_font(-1)

    def go_to_next_font(self):
        self._navigate_font(1)

    # Finds current font index in the menu list and jumps to prev/next
    def _navigate_font(self, step):
        if not self.pdf_path or not hasattr(self, 'menu_structure'):
            return

        if self.action_page_mode.isChecked():
            # PAGE MODE: Točíme se jen mezi fonty na aktuální stránce
            fonts_on_page = self.menu_structure.get(self.current_page, [])
            if not fonts_on_page: return

            try:
                idx = fonts_on_page.index(self.current_font_name)
            except ValueError:
                idx = 0

            next_idx = (idx + step) % len(fonts_on_page)
            self.load_font(self.current_page, fonts_on_page[next_idx])

        else:
            # STANDARD MODE: Točíme se mezi všemi unikátními fonty
            seq = self._get_standard_mode_sequence()
            if not seq: return

            idx = 0
            for i, (p, f) in enumerate(seq):
                if f == self.current_font_name:
                    idx = i
                    break

            next_idx = (idx + step) % len(seq)
            next_page, next_font = seq[next_idx]
            self.load_font(next_page, next_font)

    # Page Navigation Logic
    def go_to_prev_page(self):
        self._navigate_page(-1)

    def go_to_next_page(self):
        self._navigate_page(1)

    # Core logic for moving between pages
    def _navigate_page(self, step):
        if not self.pdf_path or not hasattr(self, 'menu_structure') or not self.menu_structure:
            return
        available_pages = sorted(self.menu_structure.keys())
        if not available_pages:
            return
        if self.current_page is None:
            next_page = available_pages[0]
        else:
            try:
                current_idx = available_pages.index(self.current_page)
                next_idx = (current_idx + step) % len(available_pages)
                next_page = available_pages[next_idx]
            except ValueError:
                next_page = available_pages[0]
        fonts_on_page = self.menu_structure[next_page]
        if fonts_on_page:
            self.load_font(next_page, fonts_on_page[0])

    def update_navigation_labels(self):
        if not self.pdf_path or not self.current_font_name or self.current_page is None:
            return

        is_page_mode = self.action_page_mode.isChecked()
        self.nav_page_widget.setVisible(is_page_mode)

        occurrences = []
        for p_num, fonts in self.menu_structure.items():
            if self.current_font_name in fonts:
                occurrences.append(str(p_num + 1))

        pages_str = ", ".join(occurrences)
        self.lbl_font.setText(f"{self.current_font_name}\n(Pages: {pages_str})")

        if is_page_mode:
            fonts_on_page = self.menu_structure.get(self.current_page, [])
            total = len(fonts_on_page)
            try:
                current_idx = fonts_on_page.index(self.current_font_name) + 1
            except ValueError:
                current_idx = 0

            self.lbl_font_stats.setText(f"Font {current_idx} z {total} (na této stránce)")

            all_pages = sorted(self.menu_structure.keys())
            page_idx = all_pages.index(self.current_page) + 1
            total_pages = len(all_pages)
            self.lbl_page.setText(f"Page {self.current_page + 1} ({page_idx}/{total_pages})")

        else:
            unique_fonts = self._get_standard_mode_sequence()
            total = len(unique_fonts)
            current_idx = 0
            for i, (p, f) in enumerate(unique_fonts):
                if f == self.current_font_name:
                    current_idx = i + 1
                    break

            self.lbl_font_stats.setText(f"Font {current_idx} z {total} (celkově v PDF)")

    # Moves selection to the next glyph in the list
    def show_next(self):
        if self.current_font_glyph_names:
            self.current_index = (self.current_index + 1) % len(self.current_font_glyph_names)
            self.show_glyph()

    # Callback when user clicks a glyph in the list widget
    def on_glyph_clicked(self, item):
        name = item.data(QtCore.Qt.UserRole)
        if name in self.current_font_glyph_names:
            self.current_index = self.current_font_glyph_names.index(name)
            self.show_glyph()

    # Core Logic: Saves the mapping for a single glyph
    def save_glyph(self):
        text_input = self.user_input.text().strip()
        glyph_name = self.current_font_glyph_names[self.current_index]

        unicode_hex = ""
        agn = ""

        if not text_input:
            text_input = " "
            unicode_hex = "0020"
            agn = "space"

        elif len(text_input) == 1:
            unicode_hex = format(ord(text_input), '04x')
            agn = UV2AGL.get(ord(text_input), "")

        else:
            if text_input in self.KNOWN_LIGATURES:
                unicode_hex = self.KNOWN_LIGATURES[text_input]
                agn = UV2AGL.get(int(unicode_hex, 16), text_input)
            else:
                QMessageBox.warning(
                    self,
                    "Unknown Ligature",
                    f"Combination '{text_input}' is not a known ligature.\n\n"
                )
                return

        g_hash = self.get_glyph_hash(glyph_name)

        # Store in local dictionary
        self.user_glyph_to_char[glyph_name] = {
            "glyph_hash": g_hash,
            "unicode_hex": unicode_hex,
            "AGN": agn
        }

        self.unsaved_changes = True

        # Update UI List Item
        item = self.glyph_list.item(self.current_index)
        display = "[space]" if text_input == " " else text_input
        item.setText(f" → {display}")
        item.setForeground(QtGui.QColor("#228B22"))  # Set to green

        self.user_input.clear()
        mapped_count = sum(1 for g in self.current_font_glyph_names if g in self.user_glyph_to_char)
        total_count = len(self.current_font_glyph_names)
        is_100_percent = (mapped_count == total_count)

        if is_100_percent:
            if self.action_auto_save_100.isChecked():
                self.save_to_db()
                self.statusBar().showMessage("Font 100% completed - Auto-saved", 4000)

            if self.action_auto_jump_font.isChecked():
                self.jump_to_next_unmapped()
                return

        if self.action_auto_jump_glyph.isChecked():
            self.jump_to_next_unmapped()
        else:
            self.show_next()

    def _get_page_mode_sequence(self):
        sequence = []
        if hasattr(self, 'menu_structure') and self.menu_structure:
            for p in sorted(self.menu_structure.keys()):
                for f in self.menu_structure[p]:
                    sequence.append((p, f))
        return sequence

    def _get_standard_mode_sequence(self):
        sequence = []
        if hasattr(self, 'fonts_menu'):
            for action in self.fonts_menu.actions():
                if action.isEnabled() and isinstance(action.data(), tuple):
                    sequence.append(action.data())
        return sequence

    def jump_to_next_unmapped(self):
        if not self.pdf_path or not hasattr(self, 'menu_structure'):
            return

        if self.current_font_glyph_names:
            for i in range(self.current_index + 1, len(self.current_font_glyph_names)):
                if self.current_font_glyph_names[i] not in self.user_glyph_to_char:
                    self.current_index = i
                    self.show_glyph()
                    return

        if self.action_page_mode.isChecked():
            seq = self._get_page_mode_sequence()
        else:
            seq = self._get_standard_mode_sequence()

        if not seq: return

        cur_idx = -1
        current_pair = (self.current_page, self.current_font_name)

        if not self.action_page_mode.isChecked():
            for i, (p, f) in enumerate(seq):
                if f == self.current_font_name:
                    cur_idx = i
                    break
        else:
            for i, item in enumerate(seq):
                if item == current_pair:
                    cur_idx = i
                    break

        if cur_idx != -1:
            ordered_seq = seq[cur_idx + 1:] + seq[:cur_idx + 1]
        else:
            ordered_seq = seq

        for p, fname in ordered_seq:
            if p == self.current_page and fname == self.current_font_name:
                continue

            info = self.font_cache.get((p, fname), {})
            mapped = info.get('mapped_count', 0)
            total = info.get('glyph_count', 0)

            if mapped < total:
                self.load_font(p, fname)
                for i, gname in enumerate(self.current_font_glyph_names):
                    if gname not in self.user_glyph_to_char:
                        self.current_index = i
                        self.show_glyph()
                        return

        if self.current_font_glyph_names:
            for i in range(0, self.current_index):
                if self.current_font_glyph_names[i] not in self.user_glyph_to_char:
                    self.current_index = i
                    self.show_glyph()
                    return

        QMessageBox.information(self, "Hotovo", "Výborně! Nenašel jsem žádné další neopravené glyfy.")

    # Opens a web helper for finding symbols
    def open_special(self):
        webbrowser.open_new_tab("https://www.vertex42.com/ExcelTips/unicode-symbols.html")

    # Calculates MD5 hash of the glyph shape
    # This allows us to recognize the same glyph shape even if the font name changes
    def get_glyph_hash(self, glyph_name):
        if not hasattr(self, 'current_glyph_set') or glyph_name not in self.current_glyph_set:
            return None

        try:
            glyph = self.current_glyph_set[glyph_name]
            pen = SignaturePen(self.current_glyph_set)
            glyph.draw(pen)  # Trace the shape into the pen

            shape_signature = pen.get_signature()
            if not shape_signature:
                shape_signature = "EMPTY_SPACE"

            # Return MD5 hash string
            return md5(shape_signature.encode('utf-8')).hexdigest()

        except Exception as e:
            return None

    # Saves all mappings to the database file
    def submit_ToUnicode(self):
        self.save_to_db()
        # Calculate stats for status bar
        total = len(self.current_font_glyph_names)
        mapped = sum(1 for g in self.current_font_glyph_names if g in self.user_glyph_to_char)
        self.statusBar().showMessage(f"Saved: {mapped}/{total} glyphs", 3000)
        self.go_to_next_font()

    # Loads a specific font from the PDF into memory and UI
    def load_font(self, page, font_name):
        self.current_page = page
        self.current_font_name = font_name
        self.lbl_font.setText(font_name)
        self._update_window_title()

        # Check cache first to avoid slow PDF extraction
        cache = self.font_cache.get((page, font_name))
        if not cache:
            self.statusBar().showMessage(f"Cache empty", 5000)

        try:
            # Get binary data from cache or extract if missing
            font_data = cache.get('data') or extract_cff_fonts(self.pdf_path, page, font_name)
            self.reload_font(font_data)

            # Load existing mappings from database
            self.user_glyph_to_char = {}
            self.load_mappings_for_current_font()

            # Update UI
            self.populate_glyph_list()
            self.show_glyph()

            # Enable controls
            self.user_input.setEnabled(True)
            self.btn_glyph.setEnabled(True)
            self.btn_font.setEnabled(True)
            self.statusBar().showMessage(f"Loaded: {font_name} (Page {page + 1})", 5000)

            # Update dynamic navigation labels
            self.update_navigation_labels()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error while loading font:\n{e}")

    # Decompiles raw binary CFF data into FontTools objects
    def reload_font(self, font_data):
        font = CFFFontSet()
        font.decompile(BytesIO(font_data), None)
        topDict = font.topDictIndex[0]
        glyphSet = topDict.CharStrings
        glyph_names = list(glyphSet.keys())

        # Determine baseline from .notdef glyph if possible
        # .notdef usually represents the "unknown character" box and gives good vertical metrics
        notdef_baseline = notdef_topline = None
        if '.notdef' in glyphSet:
            pen = MatplotlibPen(glyphSet)
            glyphSet['.notdef'].draw(pen)
            if pen.vertices:
                _, ys = zip(*pen.vertices)
                notdef_baseline = min(ys)
                notdef_topline = max(ys)

        # Filter out .notdef from the list shown to user
        glyph_names = [name for name in glyph_names if name != '.notdef']

        # Update state
        self.current_font = topDict
        self.current_glyph_set = glyphSet
        self.current_font_glyph_names = glyph_names
        self.canvas.font = topDict
        self.notdef_baseline = notdef_baseline
        self.notdef_topline = notdef_topline
        self.current_index = 0

    # Helper: Creates a colored circle icon for menus
    def create_status_icon(self, color):
        size = 12
        pix = QPixmap(size, size)
        pix.fill(QtCore.Qt.transparent)
        p = QtGui.QPainter(pix)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        p.setBrush(QtGui.QBrush(QtGui.QColor(color)))
        p.drawEllipse(2, 2, size - 4, size - 4)
        p.end()
        return QIcon(pix)

    # Generates a thumbnail image of a glyph for the list widget
    def generate_icon(self, glyph_name, size=(64, 64)):
        # Create a small Matplotlib figure
        fig = Figure(figsize=(1.0, 1.0), dpi=200)
        ax = fig.add_subplot()
        ax.axis('off')

        glyph = self.current_glyph_set[glyph_name]
        pen = MatplotlibPen(self.current_glyph_set)
        glyph.draw(pen)

        # If glyph has data, draw it
        if pen.vertices:
            xs, ys = zip(*pen.vertices)
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)
            width = max_x - min_x
            height = max_y - min_y
            # Scale to fit comfortably in the icon square
            scale = min(0.8 / max(width, height, 1), 0.8)

            vertices = []
            for x, y in pen.vertices:
                x_transformed = (x - min_x - width / 2) * scale
                y_transformed = (y - min_y - height / 2) * scale
                vertices.append((x_transformed, y_transformed))

            path = Path(vertices, pen.codes)
            patch = patches.PathPatch(path, facecolor='black', lw=0.5)
            ax.add_patch(patch)
            ax.set_xlim(-1.0, 1.0)
            ax.set_ylim(-1.0, 1.0)
            ax.set_aspect('equal')

        # Convert Figure to QPixmap
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        buf = canvas.buffer_rgba()
        arr = asarray(buf)
        img = QImage(arr.data, arr.shape[1], arr.shape[0], QImage.Format_RGBA8888)
        pix = QPixmap.fromImage(img)
        return pix.scaled(*size, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)

    # Fills the QListWidget with glyph thumbnails
    def populate_glyph_list(self):
        w = self.glyph_list
        w.clear()

        for name in self.current_font_glyph_names:
            pix = self.generate_icon(name, size=(self.ICON_SIZE, self.ICON_SIZE))
            item = QListWidgetItem(QIcon(pix), "")
            item.setData(QtCore.Qt.UserRole, name)
            item.setSizeHint(QtCore.QSize(0, self.ICON_SIZE + 4))

            # If already mapped in database, show result
            if name in self.user_glyph_to_char:
                ch = chr(int(self.user_glyph_to_char[name]["unicode_hex"], 16))
                disp = "[space]" if ch == " " else ch
                item.setText(f" → {disp}")
                item.setForeground(QtGui.QColor("#228B22"))
            else:
                item.setText(f" {name}")
                item.setForeground(QtGui.QColor("#888888"))

            w.addItem(item)

    # Updates the main canvas area with the selected glyph
    def show_glyph(self):
        name = self.current_font_glyph_names[self.current_index]
        self.canvas.draw_glyph(self.current_glyph_set, name, self.notdef_topline, self.notdef_baseline)

        # Retrieve mapping info if available
        mapping = self.user_glyph_to_char.get(name, {})
        uhex = mapping.get("unicode_hex", "None")
        agn = mapping.get("AGN", "None")
        ch = chr(int(uhex, 16)) if uhex != "None" else "None"

        if ch == " ": ch = "[space]"

        # Update Information Label using HTML formatting
        html = f"""
                <table width="100%" cellspacing="8">
                    <tr>
                        <td width="50%" align="right"><b>Glyph Name:</b></td>
                        <td width="50%">{name}</td>
                    </tr>
                    <tr>
                        <td width="50%" align="right"><b>Character:</b></td>
                        <td width="50%">{ch}</td>
                    </tr>
                    <tr>
                        <td width="50%" align="right"><b>Unicode:</b></td>
                        <td width="50%">{uhex}</td>
                    </tr>
                    <tr>
                        <td width="50%" align="right"><b>Adobe Glyph List:</b></td>
                        <td width="50%">{agn}</td>
                    </tr>
                </table>
                """
        self.label.setText(html)

        # Ensure the item is selected and visible in list
        item = self.glyph_list.item(self.current_index)
        if item:
            self.glyph_list.setCurrentItem(item)
            self.glyph_list.scrollToItem(item, QListWidget.EnsureVisible)

    # Builds the "Pages" menu structure
    def build_pages_menu(self, menu_data):
        menu = self.pages_menu
        menu.clear()
        if not menu_data:
            menu.addAction("No CFF font").setEnabled(False)
            return

        for page_num in sorted(menu_data.keys()):
            font_names = menu_data[page_num]
            if not font_names: continue

            # Calculate completion percentage for the page
            page_mapped = 0
            page_total = 0
            for name in font_names:
                info = self.font_cache.get((page_num, name), {})
                page_total += info.get('glyph_count', 0)
                page_mapped += info.get('mapped_count', 0)

            status, color = self._get_status_text_color(page_mapped, page_total)
            icon = self.create_status_icon(color)
            page_menu = menu.addMenu(icon, f"Page {page_num + 1} [{status}]")

            # Add individual fonts to page submenu
            for name in font_names:
                info = self.font_cache.get((page_num, name), {})
                mapped = info.get('mapped_count', 0)
                total = info.get('glyph_count', 0)

                status, color = self._get_status_text_color(mapped, total)
                action = page_menu.addAction(self.create_status_icon(color), f"{name} [{status}]")
                action.setData((page_num, name))
                # V build_pages_menu (zapne Page Mode)
                action.triggered.connect(lambda checked, p=page_num, f=name: (
                    self.action_page_mode.setChecked(True),
                    self.load_font(p, f)
                ))

    # Builds the "Fonts" menu
    def build_fonts_menu(self, menu_data):
        menu = self.fonts_menu
        menu.clear()

        unique = {}
        # Aggregate stats for fonts with same name
        for page_num, names in menu_data.items():
            for name in names:
                info = self.font_cache.get((page_num, name), {})
                total = info.get('glyph_count', 0)
                if total == 0: continue

                mapped = info.get('mapped_count', 0)
                if name not in unique:
                    unique[name] = {
                        'total': total,
                        'mapped': mapped,
                        'page': page_num,  # keep first page as default target for click
                        'pages': set(),  # all occurrences
                        'count': 0
                    }
                unique[name]['count'] += 1
                unique[name]['pages'].add(page_num)

        if not unique:
            menu.addAction("No valid CFF fonts").setEnabled(False)
            return

        for name, data in unique.items():
            mapped = data['mapped']
            total = data['total']
            status, color = self._get_status_text_color(mapped, total)

            pages_sorted = sorted(data.get('pages', []))
            pages_text = ", ".join(str(p + 1) for p in pages_sorted)
            pages_suffix = f"(Pages {pages_text})" if pages_text else "(Pages —)"

            action = menu.addAction(
                self.create_status_icon(color),
                f"{name} {pages_suffix} [{status}]"
            )
            action.setData((data['page'], name))
            action.setToolTip(
                f"Mapped: {mapped}/{total} glyphs | Occurrences: {data['count']} | Pages: {pages_text}"
            )
            # V build_fonts_menu (vypne Page Mode)
            action.triggered.connect(lambda checked, p=data['page'], f=name: (
                self.action_page_mode.setChecked(False),
                self.load_font(p, f)
            ))

    # Helper to determine status text and color based on completion percentage
    def _get_status_text_color(self, mapped, total):
        if total == 0:
            return "—", "#888888"
        perc = (mapped / total) * 100
        if perc >= 100:
            return "100%", "#228B22"
        elif perc > 0:
            return f"{int(perc)}%", "#FF8C00"
        else:
            return "—", "#888888"

    # Opens PDF file, scans structure, calculates hashes, and populates menus
    def open_pdf(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select PDF file to repair", "", "PDF Files (*.pdf)")
        if not file_path:
            return

        self.pdf_path = file_path
        self._update_window_title()
        self.statusBar().showMessage("Analyzing PDF and calculating statistics...", 0)
        QApplication.processEvents()

        try:
            self.load_db_cache()
            self.menu_structure = extract_pdf_data(file_path)
            self.font_cache.clear()

            with fitz.open(file_path) as doc:
                first_page = first_name = None

                # Iterate through all pages
                for page_num in range(len(doc)):
                    page = doc.load_page(page_num)

                    # Analyze fonts on each page
                    for font in page.get_fonts(full=True):
                        try:
                            name, ext, _, buffer = doc.extract_font(font[0])
                            # Only process CFF fonts
                            if ext and ext.lower() == "cff":
                                tmp_font = CFFFontSet()
                                tmp_font.decompile(BytesIO(buffer), None)
                                glyph_set = tmp_font.topDictIndex[0].CharStrings
                                total_glyphs = len(glyph_set)

                                # Calculate hashes for all glyphs in this font instance
                                current_font_hashes = []
                                for gname in glyph_set.keys():
                                    try:
                                        glyph = glyph_set[gname]
                                        pen = SignaturePen(glyph_set)
                                        glyph.draw(pen)
                                        sig = pen.get_signature() or "EMPTY_SPACE"
                                        ghash = md5(sig.encode('utf-8')).hexdigest()
                                        current_font_hashes.append(ghash)
                                    except:
                                        pass

                                # Count how many hashes are already in our DB
                                mapped_count = sum(1 for h in current_font_hashes if h in self.known_glyph_hashes)

                                # Cache the data
                                self.font_cache[(page_num, name)] = {
                                    'glyph_count': total_glyphs,
                                    'mapped_count': mapped_count,
                                    'glyph_hashes': current_font_hashes,
                                    'data': buffer
                                }

                                if first_page is None:
                                    first_page, first_name = page_num, name
                                self.export_action.setEnabled(True)
                        except Exception as e:
                            print(f"Error parsing font {name}: {e}")
                            self.font_cache[(page_num, name)] = {
                                'glyph_count': 0, 'mapped_count': 0, 'glyph_hashes': []
                            }

            # Build menus with the gathered data
            self.build_pages_menu(self.menu_structure)
            self.build_fonts_menu(self.menu_structure)

            if first_page is not None:
                self.load_font(first_page, first_name)
            else:
                self.clear_ui_state()
                self.statusBar().showMessage("No CFF fonts found", 3000)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error while loading PDF:\n{e}")
            self.clear_ui_state()

    # Placeholder for future save functionality
    def save_pdf(self):
        QMessageBox.information(self, "Save PDF", "Feature coming soon (TBD)")
        return

    # Refreshes menu statistics after a DB update
    def update_statistics(self):
        self.load_db_cache()
        for key, info in self.font_cache.items():
            hashes = info.get('glyph_hashes', [])
            if not hashes: continue
            new_mapped_count = sum(1 for h in hashes if h in self.known_glyph_hashes)
            info['mapped_count'] = new_mapped_count

        if hasattr(self, 'menu_structure'):
            self.build_pages_menu(self.menu_structure)
            self.build_fonts_menu(self.menu_structure)

    # Loads known hashes from CSV into a Set for fast lookup
    def load_db_cache(self):
        self.known_glyph_hashes = set()
        path = self.CSV_PATH
        if os.path.exists(path):
            try:
                with open(path, 'r', encoding='utf-8', newline='') as f:
                    reader = csv.DictReader(f, delimiter='|', quotechar='"')
                    if "glyph_hash" in reader.fieldnames:
                        for row in reader:
                            self.known_glyph_hashes.add(row["glyph_hash"])
            except Exception as e:
                print(f"DB Cache Error: {e}")



    # Saves current session work to the CSV file
    def save_to_db(self):
        path = self.CSV_PATH
        fieldnames = ["glyph_hash", "font_name", "GlyphName", "unicode_hex", "AGN"]

        existing_data = {}
        # Read existing data first to preserve it
        if os.path.exists(path):
            try:
                with open(path, 'r', encoding='utf-8', newline='') as f:
                    reader = csv.DictReader(f, delimiter='|', quotechar='"')
                    if "glyph_hash" in reader.fieldnames:
                        for row in reader:
                            existing_data[row["glyph_hash"]] = row
            except Exception:
                pass

        count_new = 0
        current_font_name = self.current_font_name or "unknown"

        # Update existing data with new mappings
        for gname, data in self.user_glyph_to_char.items():
            g_hash = data.get("glyph_hash") or self.get_glyph_hash(gname)

            if g_hash:
                existing_data[g_hash] = {
                    "glyph_hash": g_hash,
                    "font_name": current_font_name,
                    "GlyphName": gname,
                    "unicode_hex": data["unicode_hex"],
                    "AGN": data["AGN"]
                }
                count_new += 1

        try:
            # Write back to file
            with open(path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter='|', quotechar='"',
                                        quoting=csv.QUOTE_MINIMAL)
                writer.writeheader()
                for row in existing_data.values():
                    writer.writerow(row)

            # Refresh application state
            self.db_cache.clear()
            self.load_db_cache()
            self.update_statistics()
            self.statusBar().showMessage(f"Saved. Total DB size: {len(existing_data)}", 3000)

            self.unsaved_changes = False

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save:\n{e}")

    # Checks the database for any glyphs in the current font that we already know
    def load_mappings_for_current_font(self):
        self.user_glyph_to_char = {}
        db_map = {}

        # Load DB into memory
        if os.path.exists(self.CSV_PATH):
            try:
                with open(self.CSV_PATH, 'r', encoding='utf-8', newline='') as f:
                    reader = csv.DictReader(f, delimiter='|', quotechar='"')
                    if "glyph_hash" in reader.fieldnames:
                        for row in reader:
                            db_map[row["glyph_hash"]] = row
            except Exception:
                pass

        # Check each glyph in current font against DB
        for name in self.current_font_glyph_names:
            g_hash = self.get_glyph_hash(name)
            if g_hash and g_hash in db_map:
                row = db_map[g_hash]
                self.user_glyph_to_char[name] = {
                    "glyph_hash": g_hash,
                    "unicode_hex": row["unicode_hex"],
                    "AGN": row["AGN"]
                }

        # Special handling for .notdef (default missing character)
        if '.notdef' in self.current_glyph_set:
            nhash = self.get_glyph_hash('.notdef')
            if nhash in db_map:
                row = db_map[nhash]
                self.user_glyph_to_char['.notdef'] = {
                    "glyph_hash": nhash,
                    "unicode_hex": row["unicode_hex"],
                    "AGN": row["AGN"]
                }
            elif '.notdef' not in self.user_glyph_to_char:
                self.user_glyph_to_char['.notdef'] = {
                    "glyph_hash": nhash,
                    "unicode_hex": "FFFD",
                    "AGN": "notdef"
                }

if __name__ == "__main__":
    app = QApplication()

    app.setStyle("Fusion")

    dark_palette = QtGui.QPalette()
    dark_palette.setColor(QtGui.QPalette.Window, QtGui.QColor("#1e1e1e"))
    dark_palette.setColor(QtGui.QPalette.WindowText, QtGui.QColor("#f0f0f0"))
    dark_palette.setColor(QtGui.QPalette.Base, QtGui.QColor("#121212"))
    dark_palette.setColor(QtGui.QPalette.AlternateBase, QtGui.QColor("#1a1a1a"))
    dark_palette.setColor(QtGui.QPalette.ToolTipBase, QtGui.QColor("#f0f0f0"))
    dark_palette.setColor(QtGui.QPalette.ToolTipText, QtGui.QColor("#121212"))
    dark_palette.setColor(QtGui.QPalette.Text, QtGui.QColor("#f0f0f0"))
    dark_palette.setColor(QtGui.QPalette.Button, QtGui.QColor("#2a2a2a"))
    dark_palette.setColor(QtGui.QPalette.ButtonText, QtGui.QColor("#f0f0f0"))
    dark_palette.setColor(QtGui.QPalette.BrightText, QtGui.QColor("#ff0000"))
    dark_palette.setColor(QtGui.QPalette.Link, QtGui.QColor("#3d7eff"))
    dark_palette.setColor(QtGui.QPalette.Highlight, QtGui.QColor("#3d7eff"))
    dark_palette.setColor(QtGui.QPalette.HighlightedText, QtGui.QColor("#ffffff"))

    app.setPalette(dark_palette)

    app.setStyleSheet("""
        QToolTip { color: #f0f0f0; background-color: #2a2a2a; border: 1px solid #444; }
        QMenuBar::item:selected { background: #3d7eff; }
    """)

    window = FontWidget()
    window.show()
    sys.exit(app.exec())