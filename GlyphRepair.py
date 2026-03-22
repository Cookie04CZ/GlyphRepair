import csv
import os
import sys
import webbrowser
import difflib
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

import qtawesome as qta

# GUI Libraries (PySide6) for the application interface
from PySide6 import QtCore, QtGui
from PySide6.QtGui import QImage, QPixmap, QIcon
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QListWidget, QListWidgetItem, QMainWindow, QFileDialog,
    QToolButton, QMessageBox, QGroupBox, QSizePolicy, QDialog, QDialogButtonBox,
    QCheckBox, QTreeWidget, QTreeWidgetItem, QHeaderView, QComboBox
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

# Dialog window for application settings
# It allows the user to configure navigation, auto-jump, and saving preferences
class SettingsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setMinimumWidth(450)

        self.main_layout = QVBoxLayout(self)
        self.main_layout.setSpacing(15)

        # Create standard checkboxes for each setting without default text
        # (Text will be handled by the custom row layout)
        self.chk_page_mode = QCheckBox()
        self.chk_auto_highlight = QCheckBox()
        self.chk_auto_jump_glyph = QCheckBox()
        self.chk_auto_jump_font = QCheckBox()
        self.chk_auto_save_100 = QCheckBox()
        self.chk_auto_save_timer = QCheckBox()

        # Load current values from the parent (FontWidget)
        if parent:
            self.chk_page_mode.setChecked(parent.setting_page_mode)
            self.chk_auto_highlight.setChecked(parent.setting_auto_highlight)
            self.chk_auto_jump_glyph.setChecked(parent.setting_auto_jump_glyph)
            self.chk_auto_jump_font.setChecked(parent.setting_auto_jump_font)
            self.chk_auto_save_100.setChecked(parent.setting_auto_save_100)
            self.chk_auto_save_timer.setChecked(parent.setting_auto_save_timer)

        # Add widgets to layout with detailed descriptions
        self._add_setting_row(
            "Page Mode Navigation",
            "Restrict font navigation to the current page only.",
            self.chk_page_mode
        )
        self._add_setting_row(
            "Auto-highlight Suggestions",
            "Automatically select the first suggestion. Use Left/Right arrows to choose.",
            self.chk_auto_highlight
        )
        self._add_setting_row(
            "Auto-jump to Next Glyph",
            "Automatically select the next unmapped glyph after saving.",
            self.chk_auto_jump_glyph
        )
        self._add_setting_row(
            "Auto-jump Font at 100%",
            "Move to the next font automatically when all glyphs are mapped.",
            self.chk_auto_jump_font
        )
        self._add_setting_row(
            "Auto-save database at 100%",
            "Automatically save your progress to the CSV file when a font is fully mapped.",
            self.chk_auto_save_100
        )
        self._add_setting_row(
            "Auto-save every 5 mins",
            "Periodically save your progress in the background to prevent data loss.",
            self.chk_auto_save_timer
        )

        self.main_layout.addStretch()

        # Standard OK and Cancel buttons
        self.button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        self.main_layout.addWidget(self.button_box)

    # Helper method to create a visually appealing row for each setting
    # It stacks the title and description vertically, and places the checkbox on the right
    def _add_setting_row(self, title, description, checkbox_widget):
        row_layout = QHBoxLayout()

        text_layout = QVBoxLayout()
        text_layout.setSpacing(2)

        lbl_title = QLabel(title)
        lbl_title.setStyleSheet("font-weight: bold; font-size: 14px;")

        lbl_desc = QLabel(description)
        lbl_desc.setStyleSheet("color: #aaaaaa; font-size: 12px;")
        lbl_desc.setWordWrap(True)

        text_layout.addWidget(lbl_title)
        text_layout.addWidget(lbl_desc)

        row_layout.addLayout(text_layout)
        row_layout.addSpacing(20)

        # Align the checkbox to the right side of the row
        row_layout.addWidget(checkbox_widget, alignment=QtCore.Qt.AlignVCenter | QtCore.Qt.AlignRight)

        self.main_layout.addLayout(row_layout)


# Dialog window for selecting a specific page from the loaded PDF
# Uses QTreeWidget with status icons (dots) for visual feedback
class PageSelectionDialog(QDialog):
    def __init__(self, menu_data, font_cache, current_page, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Page")
        self.setMinimumSize(400, 450)
        layout = QVBoxLayout(self)

        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Search page (e.g., '12')...")
        self.search_input.setStyleSheet("padding: 5px; font-size: 14px;")
        self.search_input.setClearButtonEnabled(True)
        self.search_input.textChanged.connect(self.apply_filters)
        layout.addWidget(self.search_input)

        self.tree = QTreeWidget()
        self.tree.setColumnCount(2)
        self.tree.setHeaderHidden(True)
        self.tree.setRootIsDecorated(False)
        self.tree.setAlternatingRowColors(True)

        self.tree.header().setSectionResizeMode(0, QHeaderView.Stretch)
        self.tree.header().setSectionResizeMode(1, QHeaderView.ResizeToContents)

        item_to_scroll = None

        for page_num in sorted(menu_data.keys()):
            font_names = menu_data[page_num]
            if not font_names: continue

            page_mapped = 0
            page_total = 0
            for name in font_names:
                info = font_cache.get((page_num, name), {})
                page_total += info.get('glyph_count', 0)
                page_mapped += info.get('mapped_count', 0)

            # Get status text and color simultaneously
            status_text, color_code = self._get_status_info(page_mapped, page_total)

            item = QTreeWidgetItem([f"Page {page_num + 1}", status_text])
            item.setData(0, QtCore.Qt.UserRole, page_num)
            item.setTextAlignment(1, QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)

            # Add status icon
            item.setIcon(0, self._create_status_icon(color_code))

            if page_num == current_page:
                font = item.font(0)
                font.setBold(True)
                item.setFont(0, font)
                item.setFont(1, font)
                item_to_scroll = item

            self.tree.addTopLevelItem(item)

        self.tree.itemDoubleClicked.connect(self.accept)
        layout.addWidget(self.tree)

        self.button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        layout.addWidget(self.button_box)

        self.search_input.setFocus()
        if item_to_scroll:
            self.tree.setCurrentItem(item_to_scroll)
            self.tree.scrollToItem(item_to_scroll, QTreeWidget.PositionAtCenter)

    # Helper method to create a colored dot icon
    def _create_status_icon(self, color_str):
        size = 14
        pix = QPixmap(size, size)
        pix.fill(QtCore.Qt.transparent)
        p = QtGui.QPainter(pix)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        p.setBrush(QtGui.QBrush(QtGui.QColor(color_str)))
        p.setPen(QtCore.Qt.NoPen)
        p.drawEllipse(2, 2, size - 4, size - 4)
        p.end()
        return QIcon(pix)

    # Improved logic for status text and color
    def _get_status_info(self, mapped, total):
        if total == 0: return "—", "#888888"
        perc = (mapped / total) * 100
        if perc >= 100:
            return "100%", "#228B22"  # Green
        elif perc > 0:
            return f"{int(perc)}%", "#FF8C00"  # Orange
        return "0%", "#888888"  # Gray

    def apply_filters(self):
        search_text = self.search_input.text().lower()
        first_visible_item = None
        for i in range(self.tree.topLevelItemCount()):
            item = self.tree.topLevelItem(i)
            matches = search_text in item.text(0).lower()
            item.setHidden(not matches)
            if matches and first_visible_item is None:
                first_visible_item = item
        if first_visible_item:
            self.tree.setCurrentItem(first_visible_item)

    def get_selected_page(self):
        item = self.tree.currentItem()
        return item.data(0, QtCore.Qt.UserRole) if item else None


class FontSelectionDialog(QDialog):
    def __init__(self, menu_data, font_cache, current_font_name, current_page, parent=None):
        super().__init__(parent)
        self.current_page = current_page
        self.setWindowTitle("Select Font")
        self.setMinimumSize(650, 500)

        main_layout = QVBoxLayout(self)
        content_layout = QHBoxLayout()

        # --- LEFT PANEL ---
        left_layout = QVBoxLayout()
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Search font name...")
        self.search_input.setStyleSheet("padding: 5px; font-size: 14px;")
        self.search_input.setClearButtonEnabled(True)
        self.search_input.textChanged.connect(self.apply_filters)
        left_layout.addWidget(self.search_input)

        self.list_widget = QListWidget()
        self.list_widget.setAlternatingRowColors(True)
        self.list_widget.itemSelectionChanged.connect(self.update_details_panel)
        left_layout.addWidget(self.list_widget)

        # --- RIGHT PANEL ---
        right_widget = QWidget()
        right_widget.setFixedWidth(260)
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)

        filters_group = QGroupBox("Filters")
        f_layout = QVBoxLayout(filters_group)
        self.chk_hide_100 = QCheckBox("Hide 100% mapped")
        self.chk_hide_100.stateChanged.connect(self.apply_filters)

        page_combo_layout = QVBoxLayout()
        page_combo_layout.addWidget(QLabel("Page filter:"))
        self.combo_page = QComboBox()
        self.combo_page.setStyleSheet("QComboBox { combobox-popup: 0; }")
        self.combo_page.setMaxVisibleItems(10)
        self.combo_page.view().setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOn)
        self.combo_page.addItem("All Pages", None)
        for p in sorted(menu_data.keys()):
            self.combo_page.addItem(f"Page {p + 1}", p)
        if self.current_page is not None:
            idx = self.combo_page.findData(self.current_page)
            if idx >= 0: self.combo_page.setCurrentIndex(idx)
        self.combo_page.currentIndexChanged.connect(self.apply_filters)
        page_combo_layout.addWidget(self.combo_page)

        f_layout.addWidget(self.chk_hide_100)
        f_layout.addLayout(page_combo_layout)

        details_group = QGroupBox("Font Details")
        d_layout = QVBoxLayout(details_group)
        self.lbl_det_name = QLabel("<b>Name:</b> -")
        self.lbl_det_name.setWordWrap(True)
        self.lbl_det_status = QLabel("<b>Mapped:</b> -")
        self.lbl_det_pages = QLabel("<b>Occurs on Pages:</b> -")
        self.lbl_det_pages.setWordWrap(True)
        d_layout.addWidget(self.lbl_det_name)
        d_layout.addWidget(self.lbl_det_status)
        d_layout.addWidget(self.lbl_det_pages)
        d_layout.addStretch()

        right_layout.addWidget(filters_group)
        right_layout.addWidget(details_group)
        right_layout.addStretch()

        content_layout.addLayout(left_layout, 1)
        content_layout.addWidget(right_widget)

        self.button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)

        main_layout.addLayout(content_layout)
        main_layout.addWidget(self.button_box)

        # --- POPULATE DATA ---
        unique = {}
        for page_num, names in menu_data.items():
            for name in names:
                info = font_cache.get((page_num, name), {})
                total = info.get('glyph_count', 0)
                if total == 0: continue
                mapped = info.get('mapped_count', 0)
                if name not in unique:
                    unique[name] = {'total': total, 'mapped': mapped, 'page': page_num, 'pages': set()}
                unique[name]['pages'].add(page_num)

        item_to_scroll = None
        for name, data in unique.items():
            item = QListWidgetItem(name)

            # Generate status icon for the list item
            status_text, color_code = self._get_status_info(data['mapped'], data['total'])
            item.setIcon(self._create_status_icon(color_code))

            # Prioritize the current page if the font is available there,
            # otherwise just use the first page it occurs on (from the data dict).
            target_p = self.current_page if self.current_page in data['pages'] else data['page']

            item_data = {
                'target_page': target_p,
                'name': name,
                'all_pages': sorted(data['pages']),
                'status': status_text,
                'mapped': data['mapped'],
                'total': data['total']
            }
            item.setData(QtCore.Qt.UserRole, item_data)

            if name == current_font_name:
                font = item.font()
                font.setBold(True)
                item.setFont(font)
                item_to_scroll = item

            self.list_widget.addItem(item)

        self.list_widget.itemDoubleClicked.connect(self.accept)
        self.search_input.returnPressed.connect(self.accept)
        self.search_input.setFocus()
        self.apply_filters()

        if item_to_scroll and not item_to_scroll.isHidden():
            self.list_widget.setCurrentItem(item_to_scroll)
            self.list_widget.scrollToItem(item_to_scroll, QListWidget.PositionAtCenter)

    # --- HELPER METHODS FOR ICONS AND STATUS ---
    def _create_status_icon(self, color_str):
        size = 14
        pix = QPixmap(size, size)
        pix.fill(QtCore.Qt.transparent)
        p = QtGui.QPainter(pix)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        p.setBrush(QtGui.QBrush(QtGui.QColor(color_str)))
        p.setPen(QtCore.Qt.NoPen)
        p.drawEllipse(2, 2, size - 4, size - 4)
        p.end()
        return QIcon(pix)

    def _get_status_info(self, mapped, total):
        if total == 0: return "—", "#888888"
        perc = (mapped / total) * 100
        if perc >= 100:
            return "100%", "#228B22"
        elif perc > 0:
            return f"{int(perc)}%", "#FF8C00"
        return "0%", "#888888"

    def update_details_panel(self):
        item = self.list_widget.currentItem()
        if not item: return
        data = item.data(QtCore.Qt.UserRole)
        pages_str = ", ".join(str(p + 1) for p in data['all_pages'])
        color = "#f0f0f0"
        if data['mapped'] == data['total'] and data['total'] > 0:
            color = "#228B22"
        elif data['mapped'] > 0:
            color = "#FF8C00"

        self.lbl_det_name.setText(f"<b>Name:</b> {data['name']}")
        self.lbl_det_status.setText(
            f"<b>Mapped:</b> <span style='color:{color}'>{data['status']}</span> ({data['mapped']}/{data['total']})")
        self.lbl_det_pages.setText(f"<b>Occurs on Pages:</b> {pages_str}")

    def apply_filters(self):
        search_text = self.search_input.text().lower()
        hide_100 = self.chk_hide_100.isChecked()
        selected_page = self.combo_page.currentData()
        first_visible = None
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            data = item.data(QtCore.Qt.UserRole)
            is_visible = (search_text in item.text().lower()) and \
                         (not (hide_100 and data['mapped'] == data['total'])) and \
                         (selected_page is None or selected_page in data['all_pages'])
            item.setHidden(not is_visible)
            if is_visible and first_visible is None: first_visible = item
        if first_visible and (not self.list_widget.currentItem() or self.list_widget.currentItem().isHidden()):
            self.list_widget.setCurrentItem(first_visible)

    def get_selected_font(self):
        item = self.list_widget.currentItem()
        return item.data(QtCore.Qt.UserRole)['target_page'], item.data(QtCore.Qt.UserRole)['name'] if item else None

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
        # Initialize QSettings for persistent configuration
        self.settings_db = QSettings("GlyphRepairApp")
        
        def _get_bool(key, default):
            val = self.settings_db.value(key, default)
            if isinstance(val, str):
                return val.lower() == 'true'
            return bool(val)

        # Load settings from system or set default values
        self.setting_page_mode = _get_bool("page_mode", False)
        self.setting_auto_highlight = _get_bool("auto_highlight", True)
        self.setting_auto_jump_glyph = _get_bool("auto_jump_glyph", True)
        self.setting_auto_jump_font = _get_bool("auto_jump_font", True)
        self.setting_auto_save_100 = _get_bool("auto_save_100", True)
        self.setting_auto_save_timer = _get_bool("auto_save_timer", False)

        self.current_suggestion_idx = -1
        self.active_suggestions_count = 0
        
        self.auto_save_timer = QtCore.QTimer(self)
        self.auto_save_timer.timeout.connect(self.auto_save_interval_triggered)

        if self.setting_auto_save_timer:
            self.toggle_auto_save_timer(True)

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
        box.addButton("Cancel", QMessageBox.RejectRole)
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

        event.ignore()

    def _update_window_title(self):
        app_name = "GlyphRepair"

        pdf_name = os.path.basename(self.pdf_path) if self.pdf_path else "select file to repair"

        self.setWindowTitle(app_name + " - " + pdf_name)

    # Creates the top menu bar (File, Pages, Fonts)
    def _setup_menus(self):
        toolbar = self.addToolBar("MainToolbar")
        toolbar.setMovable(False)

        toolbar.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)

        open_action = toolbar.addAction("Open PDF")
        open_icon = qta.icon('fa5s.folder-open', color='white')
        open_action.setIcon(open_icon)
        open_action.triggered.connect(self.open_pdf)

        self.export_action = toolbar.addAction("Save PDF")
        export_icon = qta.icon('fa5s.save', color='white')
        self.export_action.setIcon(export_icon)
        self.export_action.setEnabled(False)
        self.export_action.triggered.connect(self.save_pdf)

        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        toolbar.addWidget(spacer)

        self.lbl_toolbar_info = QLabel("Font - of -")
        self.lbl_toolbar_info.setStyleSheet("color: #aaaaaa; font-size: 13px; margin-right: 15px;")
        toolbar.addWidget(self.lbl_toolbar_info)

        settings_action = toolbar.addAction("Settings")
        settings_icon = qta.icon('fa5s.cog', color='white')
        settings_action.setIcon(settings_icon)
        settings_action.triggered.connect(self.open_settings)

    def toggle_auto_save_timer(self, checked):
        if checked:
            self.auto_save_timer.start(5 * 60 * 1000)
            self.statusBar().showMessage("Auto-save timer enabled", 3000)
        else:
            self.auto_save_timer.stop()
            self.statusBar().showMessage("Auto-save timer disabled", 3000)

    def auto_save_interval_triggered(self):
        if self.unsaved_changes:
            self.save_to_db()
            self.statusBar().showMessage("Auto-save successful", 3000)

    # Initializes all widgets and layouts
    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        # Create left sidebar list
        self.glyph_list = QListWidget()
        self.glyph_list.setIconSize(QtCore.QSize(self.ICON_SIZE, self.ICON_SIZE))
        self.glyph_list.setSpacing(0)
        font = self.glyph_list.font()
        font.setFamily("Consolas")
        font.setPointSize(20)
        font.setBold(True)
        self.glyph_list.setFont(font)
        self.glyph_list.itemClicked.connect(self.on_glyph_clicked)
        nav_group = QGroupBox("Navigation")
        nav_main_layout = QVBoxLayout(nav_group)
        nav_main_layout.setSpacing(10)

        # Page navigation widget
        self.nav_page_widget = QWidget()
        nav_page_layout = QHBoxLayout(self.nav_page_widget)
        nav_page_layout.setContentsMargins(0, 0, 0, 0)
        nav_page_layout.setSpacing(5)

        self.btn_prev_page = QToolButton()
        self.btn_next_page = QToolButton()
        self.btn_select_page = QPushButton("Page: -")

        # Hard lock for page arrows (35x35)
        self.btn_prev_page.setFixedSize(35, 35)
        self.btn_next_page.setFixedSize(35, 35)
        self.btn_prev_page.setArrowType(QtCore.Qt.LeftArrow)
        self.btn_next_page.setArrowType(QtCore.Qt.RightArrow)

        # Hard lock height and set shared font for main buttons
        self.btn_select_page.setFixedHeight(35)
        shared_btn_font = self.btn_select_page.font()
        shared_btn_font.setPointSize(13)
        shared_btn_font.setBold(True)
        self.btn_select_page.setFont(shared_btn_font)

        self.btn_prev_page.clicked.connect(self.go_to_prev_page)
        self.btn_next_page.clicked.connect(self.go_to_next_page)
        self.btn_select_page.clicked.connect(self.open_page_dialog)  # Connect to dialog

        nav_page_layout.addWidget(self.btn_prev_page)
        nav_page_layout.addWidget(self.btn_select_page, 1)  # Added stretch factor
        nav_page_layout.addWidget(self.btn_next_page)

        # Font navigation buttons
        nav_font_row = QHBoxLayout()
        nav_font_row.setSpacing(5)

        self.btn_prev_font = QToolButton()
        self.btn_next_font = QToolButton()
        self.btn_select_font = QPushButton("No font loaded")

        # Hard lock for font arrows (35x35)
        self.btn_prev_font.setFixedSize(35, 35)
        self.btn_next_font.setFixedSize(35, 35)
        self.btn_prev_font.setArrowType(QtCore.Qt.LeftArrow)
        self.btn_next_font.setArrowType(QtCore.Qt.RightArrow)

        # Hard lock height and reuse the exact same font to ensure pixel-perfect match
        self.btn_select_font.setFixedHeight(35)
        self.btn_select_font.setFont(shared_btn_font)

        self.btn_prev_font.clicked.connect(self.go_to_prev_font)
        self.btn_next_font.clicked.connect(self.go_to_next_font)
        self.btn_select_font.clicked.connect(self.open_font_dialog)

        nav_font_row.addWidget(self.btn_prev_font)
        nav_font_row.addWidget(self.btn_select_font, 1)
        nav_font_row.addWidget(self.btn_next_font)

        # Assemble navigation block
        nav_main_layout.addWidget(self.nav_page_widget)
        nav_main_layout.addLayout(nav_font_row)

        self.nav_page_widget.setVisible(False)

        preview_group = QGroupBox("Glyph Preview")
        preview_layout = QVBoxLayout(preview_group)

        self.canvas = GlyphCanvas(None)
        self.label = QLabel("Select glyph")
        self.label.setStyleSheet("font-weight: bold; font-size: 24px; color: white;")
        self.label.setAlignment(QtCore.Qt.AlignCenter)

        preview_layout.addWidget(self.canvas)
        preview_layout.addWidget(self.label)

        mapping_group = QGroupBox("Mapping Tools")
        mapping_layout = QHBoxLayout(mapping_group)

        left_panel = QVBoxLayout()
        left_panel.setContentsMargins(0, 0, 0, 0)

        self.suggestions_layout = QHBoxLayout()
        self.suggestions_layout.setAlignment(QtCore.Qt.AlignLeft)
        self.suggestions_layout.setContentsMargins(0, 0, 0, 0)
        self.suggestions_layout.setSpacing(6)

        self.suggestion_buttons = []
        for _ in range(6):
            btn = QPushButton("")
            btn.setFixedSize(40, 40)
            font_sug = btn.font()
            font_sug.setPointSize(16)
            font_sug.setBold(True)
            btn.setFont(font_sug)
            btn.setStyleSheet("font-family: 'Consolas', monospace; border: 1px solid #555; border-radius: 4px;")
            btn.setEnabled(False)
            btn.setVisible(False)

            btn.suggestion_char = ""
            btn.clicked.connect(lambda checked=False, b=btn: self.apply_suggestion(b.suggestion_char))

            self.suggestions_layout.addWidget(btn)
            self.suggestion_buttons.append(btn)

        self.suggestions_layout.addStretch()  # Push suggestions to the left

        self.user_input = QLineEdit()
        self.user_input.setPlaceholderText("Enter character")
        self.user_input.setMaxLength(3)
        self.user_input.returnPressed.connect(self.save_glyph)
        self.user_input.setEnabled(False)
        self.user_input.setStyleSheet("font-size: 16px; padding: 5px;")
        self.user_input.setMinimumHeight(35)
        self.user_input.installEventFilter(self)
        self.user_input.textChanged.connect(self.on_user_input_changed)

        left_panel.addLayout(self.suggestions_layout)
        left_panel.addWidget(self.user_input)

        right_panel = QVBoxLayout()
        right_panel.setContentsMargins(0, 0, 0, 0)

        self.btn_special = QPushButton("Special Characters")
        self.btn_special.setStyleSheet("font-weight: bold; padding: 5px; min-height: 30px;")
        self.btn_special.clicked.connect(self.open_special)
        self.btn_special.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        bottom_right_layout = QHBoxLayout()
        bottom_right_layout.setContentsMargins(0, 0, 0, 0)

        self.btn_glyph = QPushButton("Save Glyph")
        self.btn_glyph.setStyleSheet("font-weight: bold; padding: 5px; min-height: 30px;")
        self.btn_glyph.setEnabled(False)
        self.btn_glyph.clicked.connect(self.save_glyph)

        self.btn_next_unmapped = QPushButton("Next Unmapped")
        self.btn_next_unmapped.setStyleSheet("font-weight: bold; padding: 5px; min-height: 30px;")
        self.btn_next_unmapped.setEnabled(False)
        self.btn_next_unmapped.clicked.connect(self.jump_to_next_unmapped)

        self.btn_font = QPushButton("Save all to DB")
        self.btn_font.setStyleSheet("font-weight: bold; padding: 5px; min-height: 30px;")
        self.btn_font.setEnabled(False)
        self.btn_font.clicked.connect(self.submit_ToUnicode)

        bottom_right_layout.addWidget(self.btn_glyph)
        bottom_right_layout.addWidget(self.btn_special)
        bottom_right_layout.addWidget(self.btn_font)

        right_panel.addWidget(self.btn_next_unmapped)
        right_panel.addLayout(bottom_right_layout)


        mapping_layout.addLayout(left_panel, 1)
        mapping_layout.addLayout(right_panel, 0)

        right_layout = QVBoxLayout()
        right_layout.addWidget(nav_group, 0)
        right_layout.addWidget(preview_group, 1)
        right_layout.addWidget(mapping_group, 0)

        main_layout = QHBoxLayout(central)

        # Lock max width of the left list so it doesn't take too much space
        self.glyph_list.setMaximumWidth(320)

        main_layout.addWidget(self.glyph_list, 0)
        main_layout.addLayout(right_layout, 4)

    # Opens the settings dialog, applies changes, and saves them persistently
    def open_settings(self):
        dialog = SettingsDialog(self)
        if dialog.exec():
            # Check if critical settings were changed
            page_mode_changed = self.setting_page_mode != dialog.chk_page_mode.isChecked()
            timer_changed = self.setting_auto_save_timer != dialog.chk_auto_save_timer.isChecked()

            # Update state variables
            self.setting_page_mode = dialog.chk_page_mode.isChecked()
            self.setting_auto_highlight = dialog.chk_auto_highlight.isChecked()
            self.setting_auto_jump_glyph = dialog.chk_auto_jump_glyph.isChecked()
            self.setting_auto_jump_font = dialog.chk_auto_jump_font.isChecked()
            self.setting_auto_save_100 = dialog.chk_auto_save_100.isChecked()
            self.setting_auto_save_timer = dialog.chk_auto_save_timer.isChecked()

            # Persist the new settings to the system
            self.settings_db.setValue("page_mode", self.setting_page_mode)
            self.settings_db.setValue("auto_highlight", self.setting_auto_highlight)
            self.settings_db.setValue("auto_jump_glyph", self.setting_auto_jump_glyph)
            self.settings_db.setValue("auto_jump_font", self.setting_auto_jump_font)
            self.settings_db.setValue("auto_save_100", self.setting_auto_save_100)
            self.settings_db.setValue("auto_save_timer", self.setting_auto_save_timer)

            # Apply runtime changes
            if page_mode_changed:
                self.update_navigation_labels()
            if timer_changed:
                self.toggle_auto_save_timer(self.setting_auto_save_timer)

    # Opens the dialog to select a specific page from the PDF
    def open_page_dialog(self):
        if not hasattr(self, 'menu_structure') or not self.menu_structure:
            return

        dialog = PageSelectionDialog(self.menu_structure, self.font_cache, self.current_page, self)
        if dialog.exec():
            selected_page = dialog.get_selected_page()
            if selected_page is not None:
                fonts = self.menu_structure.get(selected_page, [])
                if fonts:
                    self.set_page_mode(True)
                    self.load_font(selected_page, fonts[0])

    # Opens the dialog to select a specific font from the entire PDF
    def open_font_dialog(self):
        if not hasattr(self, 'menu_structure') or not self.menu_structure:
            return

        dialog = FontSelectionDialog(
            self.menu_structure,
            self.font_cache,
            self.current_font_name,
            self.current_page,
            self
        )

        if dialog.exec():
            selected_data = dialog.get_selected_font()
            if selected_data:
                page, font_name = selected_data
                self.load_font(page, font_name)

    # Helper method to change page mode dynamically from the UI
    def set_page_mode(self, mode):
        self.setting_page_mode = mode
        self.update_navigation_labels()

    # Resets the UI elements when no font is loaded
    def clear_ui_state(self):
        self.glyph_list.clear()
        self.label.setText("No font loaded")
        self.canvas.draw_glyph(None, None, None, None)
        self.user_input.clear()
        self.user_input.setEnabled(False)
        self.btn_glyph.setEnabled(False)
        self.btn_font.setEnabled(False)
        if hasattr(self, 'suggestion_buttons'):
            for btn in self.suggestion_buttons:
                btn.setEnabled(False)

        # Reset navigation labels
        self.btn_select_font.setText("No font loaded")
        self.btn_select_page.setText("Page: -")
        self.lbl_toolbar_info.setText("Font - of -")
        self.nav_page_widget.setVisible(False)
        self.unsaved_changes = False
        self._update_window_title()

        if hasattr(self, 'suggestion_buttons'):
            for btn in self.suggestion_buttons:
                btn.setText("")
                btn.setEnabled(False)
                btn.setVisible(False)

    # Font Navigation Logic
    def go_to_prev_font(self):
        self._navigate_font(-1)

    def go_to_next_font(self):
        self._navigate_font(1)

    # Finds current font index in the menu list and jumps to prev/next
    def _navigate_font(self, step):
        if not self.pdf_path or not hasattr(self, 'menu_structure'):
            return

        if self.setting_page_mode:
            fonts_on_page = self.menu_structure.get(self.current_page, [])
            if not fonts_on_page: return

            try:
                idx = fonts_on_page.index(self.current_font_name)
            except ValueError:
                idx = 0

            next_idx = (idx + step) % len(fonts_on_page)
            self.load_font(self.current_page, fonts_on_page[next_idx])

        else:
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
            
            # Prefer loading the same font on the new page if it exists there
            target_font = fonts_on_page[0]
            if self.current_font_name in fonts_on_page:
                target_font = self.current_font_name
                
            self.load_font(next_page, target_font)

    def update_navigation_labels(self):
        if not self.pdf_path or not self.current_font_name or self.current_page is None:
            return

        self.nav_page_widget.setVisible(self.setting_page_mode)

        self.btn_select_font.setText(self.current_font_name)


        if self.setting_page_mode:
            fonts_on_page = self.menu_structure.get(self.current_page, [])
            total = len(fonts_on_page)
            try:
                current_idx = fonts_on_page.index(self.current_font_name) + 1
            except ValueError:
                current_idx = 0

            self.lbl_toolbar_info.setText(f"Font {current_idx} of {total} (Current Page)")

            all_pages = sorted(self.menu_structure.keys())
            page_idx = all_pages.index(self.current_page) + 1
            total_pages = len(all_pages)
            self.btn_select_page.setText(f"Page {self.current_page + 1} ({page_idx}/{total_pages})")

        else:
            unique_fonts = self._get_standard_mode_sequence()
            total = len(unique_fonts)
            current_idx = 0
            for i, (p, f) in enumerate(unique_fonts):
                if f == self.current_font_name:
                    current_idx = i + 1
                    break

            self.lbl_toolbar_info.setText(f"Font {current_idx} of {total} (Global)")

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
            if self.setting_auto_save_100:
                self.save_to_db()
                self.statusBar().showMessage("Font 100% completed - Auto-saved", 4000)

            if self.setting_auto_jump_font:
                self.jump_to_next_unmapped()

            return

        if self.setting_auto_jump_glyph:
            self.jump_to_next_unmapped()
        else:
            pass

    # Returns an ordered list of all (page, font_name) pairs
    def _get_page_mode_sequence(self):
        sequence = []
        if hasattr(self, 'menu_structure') and self.menu_structure:
            for p in sorted(self.menu_structure.keys()):
                for f in self.menu_structure[p]:
                    sequence.append((p, f))
        return sequence

    # Returns a list of unique fonts for global mode mapping
    def _get_standard_mode_sequence(self):
        sequence = []
        if hasattr(self, 'menu_structure') and self.menu_structure:
            unique = set()
            for p, fonts in sorted(self.menu_structure.items()):
                for f in fonts:
                    if f not in unique:
                        unique.add(f)
                        sequence.append((p, f))
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

        if self.setting_page_mode:
            seq = self._get_page_mode_sequence()
        else:
            seq = self._get_standard_mode_sequence()

        if not seq: return

        cur_idx = -1
        current_pair = (self.current_page, self.current_font_name)

        if not self.setting_page_mode:
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
                if self.unsaved_changes:
                    self.save_to_db()

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

        QMessageBox.information(self, "Finished", "Great! No more unmapped glyphs found.")

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
            self.btn_next_unmapped.setEnabled(True)

            if hasattr(self, 'suggestion_buttons'):
                for btn in self.suggestion_buttons:
                    btn.setEnabled(True)

            self.statusBar().showMessage(f"Loaded: {font_name} (Page {page + 1})", 5000)

            # Update dynamic navigation labels
            self.update_navigation_labels()

            self.user_input.setFocus()

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
                        <td width="50%" style="font-family: Consolas, monospace; font-weight: bold; color: white;">{name}</td>
                    </tr>
                    <tr>
                        <td width="50%" align="right"><b>Character:</b></td>
                        <td width="50%" style="font-family: Consolas, monospace; font-weight: bold; color: white;">{ch}</td>
                    </tr>
                    <tr>
                        <td width="50%" align="right"><b>Unicode:</b></td>
                        <td width="50%" style="font-family: Consolas, monospace; font-weight: bold; color: white;">{uhex}</td>
                    </tr>
                    <tr>
                        <td width="50%" align="right"><b>Adobe Glyph List:</b></td>
                        <td width="50%" style="font-family: Consolas, monospace; font-weight: bold; color: white;">{agn}</td>
                    </tr>
                </table>
                """
        self.label.setText(html)

        # Ensure the item is selected and visible in list
        item = self.glyph_list.item(self.current_index)
        if item:
            self.glyph_list.setCurrentItem(item)
            self.glyph_list.scrollToItem(item, QListWidget.EnsureVisible)

        self.update_suggestions_ui(name, self.current_font_name)

        self.user_input.setFocus()

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
                                        
                                        # Detect completely empty glyphs and assign a special hash
                                        if not pen.signature:
                                            ghash = md5("EMPTY_SPACE".encode('utf-8')).hexdigest()
                                        else:
                                            sig = pen.get_signature()
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

    # Loads known hashes from CSV into a Set for fast lookup
    def load_db_cache(self):
        self.known_glyph_hashes = set()
        
        # Always inject the special space hash
        space_hash = md5("EMPTY_SPACE".encode('utf-8')).hexdigest()
        self.known_glyph_hashes.add(space_hash)
        
        self.db_records = []
        path = self.CSV_PATH
        if os.path.exists(path):
            try:
                with open(path, 'r', encoding='utf-8', newline='') as f:
                    reader = csv.DictReader(f, delimiter='|', quotechar='"')
                    if "glyph_hash" in reader.fieldnames:
                        for row in reader:
                            self.known_glyph_hashes.add(row["glyph_hash"])
                            self.db_records.append(row)
            except Exception as e:
                print(f"DB Cache Error: {e}")

        # Generates suggestions based on GlyphName and fuzzy matching of font_name
    def get_suggestions(self, glyph_name, font_name):
        if not hasattr(self, 'db_records') or not self.db_records or not glyph_name or not font_name:
            return []

        # Strip PDF subset prefix (e.g., "GKCLND+Arial" -> "Arial")
        current_clean_font = font_name.split('+', 1)[-1] if '+' in font_name else font_name

        matches = []
        for row in self.db_records:
            if row.get("GlyphName") == glyph_name:
                db_font = row.get("font_name", "")
                db_clean_font = db_font.split('+', 1)[-1] if '+' in db_font else db_font

                # Calculate string similarity ratio (0.0 to 1.0)
                similarity = difflib.SequenceMatcher(None, current_clean_font, db_clean_font).ratio()
                matches.append((similarity, row.get("unicode_hex")))

        # Sort matches by highest similarity first
        matches.sort(key=lambda x: x[0], reverse=True)

        suggestions = []
        for _, hex_val in matches:
            try:
                char = chr(int(hex_val, 16))
                # Add unique characters until we have 6 (for our 6 buttons)
                if char not in suggestions:
                    suggestions.append(char)
                if len(suggestions) >= 6:
                    break
            except (ValueError, TypeError):
                pass

        return suggestions

    # Refreshes the suggestion buttons above the text input
    def update_suggestions_ui(self, glyph_name, font_name):
        suggestions = self.get_suggestions(glyph_name, font_name)
        self.active_suggestions_count = len(suggestions)

        for i, btn in enumerate(self.suggestion_buttons):
            if i < len(suggestions):
                char = suggestions[i]
                btn.setText(char)
                btn.suggestion_char = char  # Update the stored character
                btn.setEnabled(True)
                btn.setVisible(True)  # Show button if we have a suggestion
            else:
                btn.setText("")
                btn.suggestion_char = ""
                btn.setEnabled(False)
                btn.setVisible(False)  # Hide unused button

        # Auto-highlight logic
        if self.setting_auto_highlight and self.active_suggestions_count > 0:
            self.set_suggestion_highlight(0)
        else:
            self.set_suggestion_highlight(-1)

    # Visually highlights a specific suggestion button with a blue border
    def set_suggestion_highlight(self, index):
        self.current_suggestion_idx = index
        for i, btn in enumerate(self.suggestion_buttons):
            if i == index and btn.isVisible():
                # Highlighted style - transparent background, prominent blue border
                btn.setStyleSheet(
                    "font-family: 'Consolas', monospace; border: 2px solid #3d7eff; background-color: transparent; border-radius: 4px; color: white;")
            else:
                # Default style - dark gray border
                btn.setStyleSheet(
                    "font-family: 'Consolas', monospace; border: 1px solid #555; background-color: transparent; border-radius: 4px; color: #f0f0f0;")

    # Removes highlight if the user starts typing manually
    def on_user_input_changed(self, text):
        if text and self.current_suggestion_idx != -1:
            self.set_suggestion_highlight(-1)
        elif not text and self.setting_auto_highlight and self.active_suggestions_count > 0:
            # Re-highlight the first button if the user deletes their text
            self.set_suggestion_highlight(0)

        # Catches keyboard events in the user_input field for suggestion navigation
    def eventFilter(self, obj, event):
        if obj == self.user_input and event.type() == QtCore.QEvent.KeyPress:

            # Allow Up/Down arrow keys to navigate the glyph list directly
            if event.key() == QtCore.Qt.Key_Up:
                if self.current_font_glyph_names and self.current_index > 0:
                    self.current_index -= 1
                    self.show_glyph()
                return True
                
            elif event.key() == QtCore.Qt.Key_Down:
                if self.current_font_glyph_names and self.current_index < len(self.current_font_glyph_names) - 1:
                    self.current_index += 1
                    self.show_glyph()
                return True

            if not self.setting_auto_highlight:
                return super().eventFilter(obj, event)

            # Left arrow
            if event.key() == QtCore.Qt.Key_Left:
                # Allow normal left movement if there is text in the box
                if self.user_input.text():
                    return False

                if self.active_suggestions_count > 0:
                    new_idx = max(0, self.current_suggestion_idx - 1)
                    if self.current_suggestion_idx == -1: new_idx = 0
                    self.set_suggestion_highlight(new_idx)
                    return True  # Block the event

            # Right arrow
            elif event.key() == QtCore.Qt.Key_Right:
                # Allow normal right movement if there is text in the box
                if self.user_input.text():
                    return False

                if self.active_suggestions_count > 0:
                    new_idx = min(self.active_suggestions_count - 1, self.current_suggestion_idx + 1)
                    if self.current_suggestion_idx == -1: new_idx = 0
                    self.set_suggestion_highlight(new_idx)
                    return True  # Block the event

            # Enter or Return key
            elif event.key() in (QtCore.Qt.Key_Return, QtCore.Qt.Key_Enter):
                # Apply highlighted suggestion ONLY if the text box is completely empty
                # AND a valid suggestion is currently highlighted
                if not self.user_input.text().strip() and self.current_suggestion_idx >= 0:
                    try:
                        btn = self.suggestion_buttons[self.current_suggestion_idx]
                        if btn.isVisible():
                            self.apply_suggestion(btn.suggestion_char)
                            return True  # Block the event, we handled it
                    except IndexError:
                        pass  # Failsafe in case active_suggestions_count is out of sync

                # If text box is NOT empty, let the normal returnPressed signal handle it
                return False

                # Pass any unhandled events to the base class
        return super().eventFilter(obj, event)

    # Automatically fills input and triggers the save mechanism
    def apply_suggestion(self, char):
        self.user_input.setText(char)
        self.save_glyph()

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

        # Add intrinsic empty space hash mapping
        space_hash = md5("EMPTY_SPACE".encode('utf-8')).hexdigest()
        db_map[space_hash] = {
            "unicode_hex": "0020",
            "AGN": "space"
        }

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
    dark_palette.setColor(QtGui.QPalette.Highlight, QtGui.QColor("#3d7eff"))
    dark_palette.setColor(QtGui.QPalette.HighlightedText, QtGui.QColor("#ffffff"))
    dark_palette.setColor(QtGui.QPalette.PlaceholderText, QtGui.QColor("#898989"))

    app.setPalette(dark_palette)

    app.setStyleSheet("""
        QToolTip { color: #f0f0f0; background-color: #2a2a2a; border: 1px solid #444; }
        QMenuBar::item:selected { background: #3d7eff; }
        
        QToolBar { 
            border: none;
            border-bottom: 1px solid #333; /* Decentní tmavá linka vespod */
            background: #1e1e1e;
            padding: 3px;
        }
        
        QToolBar QToolButton {
            border: none;
            border-radius: 4px;
            padding: 4px;
            margin: 2px;
        }
        QToolBar QToolButton:hover {
            background-color: #3d3d3d; /* Jemné zvýraznění při najetí */
        }
        QToolBar QToolButton:pressed {
            background-color: #3d7eff; /* Modrá při kliknutí */
        }
    """)

    window = FontWidget()
    window.show()
    sys.exit(app.exec())