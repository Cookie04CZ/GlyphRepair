import csv
import os
import re
import sys
import webbrowser
import difflib
import argparse
from hashlib import md5
from io import BytesIO

import fitz  # PyMuPDF

import qtawesome as qta

# GUI Libraries (PySide6) for the application interface
from PySide6 import QtCore, QtGui
from PySide6.QtGui import QPixmap, QIcon, QRegularExpressionValidator, QShortcut, QKeySequence, QPainterPath
from PySide6.QtCore import QSettings, QRegularExpression
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QListWidget, QListWidgetItem, QMainWindow, QFileDialog,
    QToolButton, QMessageBox, QGroupBox, QSizePolicy, QDialog, QDialogButtonBox,
    QCheckBox, QTreeWidget, QTreeWidgetItem, QHeaderView, QComboBox, QProgressBar, QAbstractItemView, QTextEdit
)

# FontTools libraries for parsing font data (CFF format)
from fontTools.agl import UV2AGL, AGL2UV
from fontTools.cffLib import CFFFontSet
from fontTools.pens.basePen import BasePen

# Dictionary combining standard AGL with custom project-specific glyph names
EXTENDED_AGL = AGL2UV.copy()
EXTENDED_AGL.update({
    "nonbreakingspace": 0x00A0,
    "Ohm": 0x2126,
    "Omegagreek": 0x2126,
    "fi" : 0xfb01,
    # Add any other missing glyphs you want to auto-map here
})


# Function to extract raw font data from a specific page in a PDF
# It looks for a font with a specific name in CFF format (without ToUnicode) and returns its binary buffer
def extract_cff_fonts(pdf_path, page, font_name):
    # Open the PDF file using PyMuPDF
    with fitz.open(pdf_path) as doc:
        # Load the specific page object
        page_obj = doc.load_page(page)
        # Get a list of all fonts referenced on this page
        fonts = page_obj.get_fonts(full=True)

        # Iterate through the found fonts to find the one matching font_name
        for font in fonts:
            xref = font[0]
            # Extract font metadata and the binary content (buffer)
            name, ext, _, buffer = doc.extract_font(xref)

            # We only care about CFF files that match our target name and lack ToUnicode
            if ext and ext.lower() == "cff" and name == font_name:
                if not has_tounicode(doc, xref):
                    return buffer

        # If the loop finishes without returning, the font was not found or has ToUnicode
        raise ValueError(f"Font '{font_name}' not found, not in CFF format, or already has a ToUnicode map.")


# Function to check if a font dictionary contains a /ToUnicode stream
# PyMuPDF returns a tuple (type, value). If the key is missing, type is 'null'.
def has_tounicode(doc, xref):
    key_type, _ = doc.xref_get_key(xref, "ToUnicode")
    return key_type != 'null'

# Class representing a pen that generates a string signature of a glyph
# This is used for identification/hashing.
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


class QtPen(BasePen):
    def __init__(self, glyphset):
        super().__init__(glyphset)
        self.path = QPainterPath()

    def _moveTo(self, p):
        self.path.moveTo(p[0], p[1])

    def _lineTo(self, p):
        self.path.lineTo(p[0], p[1])

    def _curveToOne(self, p1, p2, p3):
        self.path.cubicTo(p1[0], p1[1], p2[0], p2[1], p3[0], p3[1])

    def _closePath(self):
        self.path.closeSubpath()


class GlyphQtWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.path = None
        self.baseline = None
        self.topline = None
        self.msg = None
        self.font_bbox = [0, -200, 0, 1000]
        self.setMinimumSize(400, 400)

    def draw_glyph(self, glyphset, glyph_name, notdef_max_y, notdef_min_y):
        self.msg = None

        if not glyphset or glyph_name not in glyphset:
            self.path = None
            self.msg = "No glyph"
            self.update()
            return

        glyph = glyphset[glyph_name]
        pen = QtPen(glyphset)
        glyph.draw(pen)

        self.path = pen.path

        if self.path.isEmpty():
            self.msg = "Empty glyph\n(likely space)"

        self.baseline = notdef_min_y
        self.topline = notdef_max_y
        self.update()

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        painter.fillRect(self.rect(), QtCore.Qt.white)

        if self.msg:
            painter.setPen(QtGui.QColor("dimgray"))

            font = painter.font()
            font.setBold(True)
            font.setItalic(True)
            font.setPointSize(48 if "No glyph" in self.msg else 30)
            painter.setFont(font)

            painter.drawText(self.rect(), QtCore.Qt.AlignCenter, self.msg)
            return

        if self.path is None:
            return

        rect = self.rect()
        view_unit = min(rect.width(), rect.height())

        if self.baseline is not None and self.topline is not None:
            ref_min, ref_max = self.baseline, self.topline
        else:
            ref_min, ref_max = self.font_bbox[1], self.font_bbox[3]

        ref_height = max(ref_max - ref_min, 1)
        ref_midpoint = (ref_max + ref_min) / 2
        scale_factor = (view_unit * 0.45) / ref_height

        br = self.path.boundingRect()
        glyph_center_x = br.left() + br.width() / 2.0

        painter.save()
        painter.translate(rect.width() / 2.0, rect.height() / 2.0)
        painter.scale(scale_factor, -scale_factor)
        painter.translate(-glyph_center_x, -ref_midpoint)

        line_pen = QtGui.QPen()
        line_pen.setCosmetic(True)
        line_pen.setWidth(1)

        if self.baseline is not None:
            line_pen.setColor(QtGui.QColor("blue"))
            line_pen.setStyle(QtCore.Qt.DotLine)
            painter.setPen(line_pen)
            painter.drawLine(QtCore.QLineF(-10000, self.baseline, 10000, self.baseline))
            painter.drawLine(QtCore.QLineF(-10000, self.topline, 10000, self.topline))
        else:
            line_pen.setColor(QtGui.QColor("red"))
            line_pen.setStyle(QtCore.Qt.SolidLine)
            painter.setPen(line_pen)
            painter.drawLine(QtCore.QLineF(-10000, 0, 10000, 0))

        painter.setBrush(QtGui.QBrush(QtGui.QColor("black")))
        painter.setPen(QtCore.Qt.NoPen)
        painter.drawPath(self.path)
        painter.restore()

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
        self.chk_auto_save_on_switch = QCheckBox()
        self.chk_auto_save_timer = QCheckBox()
        self.chk_show_hex_input = QCheckBox()

        # Load current values from the parent (FontWidget)
        if parent:
            self.chk_page_mode.setChecked(parent.setting_page_mode)
            self.chk_auto_highlight.setChecked(parent.setting_auto_highlight)
            self.chk_auto_jump_glyph.setChecked(parent.setting_auto_jump_glyph)
            self.chk_auto_jump_font.setChecked(parent.setting_auto_jump_font)
            self.chk_auto_save_100.setChecked(parent.setting_auto_save_100)
            self.chk_auto_save_on_switch.setChecked(parent.setting_auto_save_on_switch)
            self.chk_auto_save_timer.setChecked(parent.setting_auto_save_timer)
            self.chk_show_hex_input.setChecked(parent.setting_show_hex_input)

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
            "Automatically save your progress to the PSV file when a font is fully mapped.",
            self.chk_auto_save_100
        )
        self._add_setting_row(
            "Auto-save on Switch",
            "Automatically save your progress when switching to a different font or page.",
            self.chk_auto_save_on_switch
        )
        self._add_setting_row(
            "Auto-save every 5 mins",
            "Periodically save your progress in the background to prevent data loss.",
            self.chk_auto_save_timer
        )
        self._add_setting_row(
            "Show Unicode Hex Input",
            "Display the secondary input field for direct Unicode hex code entry.",
            self.chk_show_hex_input
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
            page_agl = 0
            for name in font_names:
                info = font_cache.get((page_num, name), {})
                page_total += info.get('glyph_count', 0)
                page_mapped += info.get('mapped_count', 0)
                page_agl += info.get('agl_count', 0)

            # Get status text and color simultaneously
            status_text, color_code = parent._get_status_info(page_mapped, page_total, page_agl) if parent else ("—", "#888888")

            item = QTreeWidgetItem([f"Page {page_num + 1}", status_text])
            item.setData(0, QtCore.Qt.UserRole, page_num)
            item.setTextAlignment(1, QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)

            # Add status icon
            if parent:
                item.setIcon(0, parent._create_status_icon(color_code))

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
        self.lbl_det_agl = QLabel("<b>AGL Glyphs:</b> -")
        self.lbl_det_unmapped = QLabel("<b>Unmapped:</b> -")
        self.lbl_det_pages = QLabel("<b>Occurs on Pages:</b> -")
        self.lbl_det_pages.setWordWrap(True)
        d_layout.addWidget(self.lbl_det_name)
        d_layout.addWidget(self.lbl_det_status)
        d_layout.addWidget(self.lbl_det_agl)
        d_layout.addWidget(self.lbl_det_unmapped)
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
                agl_c = info.get('agl_count', 0)
                if name not in unique:
                    unique[name] = {'total': total, 'mapped': mapped, 'agl': agl_c, 'page': page_num, 'pages': set()}
                unique[name]['pages'].add(page_num)

        item_to_scroll = None
        for name, data in unique.items():
            item = QListWidgetItem(name)

            # Generate status icon for the list item
            status_text, color_code = self._get_status_info(data['mapped'], data['total'], data['agl'])
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
                'total': data['total'],
                'agl': data['agl']
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
        p.setPen(QtCore.Qt.NoPen)

        if color_str.upper() == "#00CED1":
            p.setBrush(QtGui.QBrush(QtGui.QColor("#3d7eff")))
            p.drawPie(2, 2, size - 4, size - 4, 90 * 16, 180 * 16)

            p.setBrush(QtGui.QBrush(QtGui.QColor("#228B22")))
            p.drawPie(2, 2, size - 4, size - 4, 270 * 16, 180 * 16)
        else:
            p.setBrush(QtGui.QBrush(QtGui.QColor(color_str)))
            p.drawEllipse(2, 2, size - 4, size - 4)

        p.end()
        return QIcon(pix)

    def _get_status_info(self, mapped, total, agl_count=0):
        if total == 0: return "—", "#888888"
        perc = (mapped / total) * 100

        if perc >= 100:
            if agl_count > 0:
                return "100%", "#00CED1"  # Teal/Cyan for 100% complete containing AGL
            else:
                return "100%", "#228B22"  # Solid Green for 100% strictly manual
        elif perc > 0 or agl_count > 0:
            if agl_count > 0:
                return f"{int(perc)}%", "#3d7eff"  # Blue for in-progress containing AGL
            else:
                return f"{int(perc)}%", "#FF8C00"  # Orange for in-progress purely manual

        return "0%", "#888888"

    def update_details_panel(self):
        item = self.list_widget.currentItem()
        if not item: return
        data = item.data(QtCore.Qt.UserRole)
        pages_str = ", ".join(str(p + 1) for p in data['all_pages'])

        mapped = data['mapped']
        total = data['total']
        agl = data.get('agl', 0)
        unmapped = total - mapped

        # Match color logic with the status dots
        color = "#f0f0f0"
        if mapped == total and total > 0:
            color = "#00CED1" if agl > 0 else "#228B22"
        elif mapped > 0 or agl > 0:
            color = "#3d7eff" if agl > 0 else "#FF8C00"

        self.lbl_det_name.setText(f"<b>Name:</b> {data['name']}")
        self.lbl_det_status.setText(
            f"<b>Mapped:</b> <span style='color:{color}; font-weight:bold;'>{data['status']}</span> ({mapped} / {total})"
        )

        # Format AGL info
        agl_text = f"<span style='color:#00CED1;'>Yes ({agl})</span>" if agl > 0 else "No"
        self.lbl_det_agl.setText(f"<b>AGL Glyphs:</b> {agl_text}")

        # Format Unmapped info (Red if there is work to do, Green if done)
        unmapped_color = "#ff4444" if unmapped > 0 else "#228B22"
        self.lbl_det_unmapped.setText(
            f"<b>Unmapped:</b> <span style='color:{unmapped_color}; font-weight:bold;'>{unmapped}</span>")

        self.lbl_det_pages.setText(f"<b>Occurs on Pages:</b> {pages_str}")

    def apply_filters(self):
        search_text = self.search_input.text().lower()
        hide_100 = self.chk_hide_100.isChecked()
        selected_page = self.combo_page.currentData()

        effective_page_filter = None if search_text else selected_page

        first_visible = None
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            data = item.data(QtCore.Qt.UserRole)
            is_visible = (search_text in item.text().lower()) and \
                         (not (hide_100 and data['mapped'] == data['total'])) and \
                         (effective_page_filter is None or effective_page_filter in data['all_pages'])
            item.setHidden(not is_visible)
            if is_visible and first_visible is None: first_visible = item
        if first_visible and (not self.list_widget.currentItem() or self.list_widget.currentItem().isHidden()):
            self.list_widget.setCurrentItem(first_visible)

    def get_selected_font(self):
        item = self.list_widget.currentItem()
        return item.data(QtCore.Qt.UserRole)['target_page'], item.data(QtCore.Qt.UserRole)['name'] if item else None


class IntegratedRepairDialog(QDialog):
    def __init__(self, page_font_map, total_unique, ready_unique, parent=None):
        super().__init__(parent)
        self.setWindowTitle("PDF Repair")
        self.setMinimumSize(600, 450)
        self.setWindowModality(QtCore.Qt.WindowModal)

        self.main_layout = QVBoxLayout(self)

        self.setup_group = QGroupBox("Font check before repair")
        setup_layout = QVBoxLayout(self.setup_group)

        self.lbl_summary = QLabel(
            f"GlypRepair will repair <span style='color:#228B22; font-weight:bold;'>{ready_unique}</span> "
            f"out of <span style='font-weight:bold;'>{total_unique}</span> CFF fonts."
        )
        self.lbl_summary.setStyleSheet("font-size: 15px; margin-bottom: 5px;")
        setup_layout.addWidget(self.lbl_summary)

        self.tree = QTreeWidget()
        self.tree.setAlternatingRowColors(True)
        self.tree.setColumnWidth(1, 140)
        self.tree.itemDoubleClicked.connect(self.on_item_double_clicked)

        self.ready_count = 0
        for p_num in sorted(page_font_map.keys()):
            page_mapped = 0
            page_total = 0
            page_all_ok = True

            for f_info in page_font_map[p_num]:
                page_mapped += f_info['mapped']
                page_total += f_info['total']
                if f_info['total'] == 0 or f_info['mapped'] < f_info['total']:
                    page_all_ok = False

            page_item = QTreeWidgetItem([f"Page {p_num + 1}", ""])
            page_item.setData(0, QtCore.Qt.UserRole, (p_num, None))

            page_item.setSizeHint(0, QtCore.QSize(0, 30))

            color = "#228B22" if page_all_ok else "#FF8C00"
            page_item.setIcon(0, self._create_status_icon(color))

            f = page_item.font(0)
            f.setBold(True)
            page_item.setFont(0, f)

            page_item.setExpanded(True)
            self.tree.addTopLevelItem(page_item)

            pbar_page_widget = self._create_progress_bar_widget(page_mapped, page_total)
            self.tree.setItemWidget(page_item, 1, pbar_page_widget)

            for f_info in page_font_map[p_num]:
                is_ok = f_info['mapped'] >= f_info['total'] and f_info['total'] > 0
                font_item = QTreeWidgetItem([f_info['name'], ""])
                font_item.setData(0, QtCore.Qt.UserRole, (p_num, f_info['name']))

                font_item.setSizeHint(0, QtCore.QSize(0, 26))

                f_color = "#228B22" if is_ok else "#FF8C00"
                font_item.setIcon(0, self._create_status_icon(f_color))

                if is_ok:
                    self.ready_count += 1

                page_item.addChild(font_item)

                pbar_font_widget = self._create_progress_bar_widget(f_info['mapped'], f_info['total'])
                self.tree.setItemWidget(font_item, 1, pbar_font_widget)

        setup_layout.addWidget(self.tree)
        self.main_layout.addWidget(self.setup_group)

        self.log_group = QGroupBox("Repair progress")
        log_layout = QVBoxLayout(self.log_group)
        self.text_log = QTextEdit()
        self.text_log.setReadOnly(True)
        self.text_log.setStyleSheet(
            "background-color: #121212; color: #00ff00; font-family: Consolas; font-size: 12px;")
        log_layout.addWidget(self.text_log)
        self.log_group.setVisible(False)
        self.main_layout.addWidget(self.log_group)

        self.button_layout = QHBoxLayout()
        self.button_layout.setSpacing(10)

        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.setFixedHeight(40)
        self.btn_cancel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.btn_cancel.setStyleSheet("font-weight: bold; background-color: #444444; color: white; border-radius: 4px;")

        self.btn_repair = QPushButton(f"Start repair ({ready_unique} fonts)")
        self.btn_repair.setFixedHeight(40)
        self.btn_repair.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.btn_repair.setStyleSheet("font-weight: bold; background-color: #228B22; color: white; border-radius: 4px;")

        self.button_layout.addWidget(self.btn_cancel)
        self.button_layout.addWidget(self.btn_repair)
        self.main_layout.addLayout(self.button_layout)

        self.jump_target = None
        self.btn_cancel.clicked.connect(self.reject)
        self.btn_repair.clicked.connect(self.start_repair_flow)

    def _create_progress_bar_widget(self, mapped, total):
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(0)

        pbar = QProgressBar()
        pbar.setFixedHeight(16)

        is_ok = (total > 0 and mapped >= total)
        color = "#228B22" if is_ok else "#FF8C00"

        if total == 0:
            pbar.setMaximum(1)
            pbar.setValue(1)
            pbar.setFormat("0 / 0")
            color = "#888888"
        else:
            pbar.setMaximum(total)
            pbar.setValue(mapped)
            pbar.setFormat("%v / %m")

        pbar.setStyleSheet(f"""
            QProgressBar {{
                border: 1px solid #444;
                border-radius: 3px;
                background-color: #2a2a2a;
                text-align: center;
                color: white;
                font-weight: bold;
                font-size: 11px;
            }}
            QProgressBar::chunk {{
                background-color: {color};
                border-radius: 2px;
            }}
        """)

        layout.addWidget(pbar)
        return container

    def on_item_double_clicked(self, item, column):
        self.jump_target = item.data(0, QtCore.Qt.UserRole)
        if self.jump_target:
            self.done(QDialog.Accepted + 1)

    def start_repair_flow(self):
        self.switch_to_log_mode()
        self.accept()

    def switch_to_log_mode(self):
        self.setup_group.setVisible(False)
        self.log_group.setVisible(True)
        self.btn_repair.setVisible(False)
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.setText("Working...")
        QApplication.processEvents()

    def log(self, message):
        self.text_log.append(message)
        self.text_log.verticalScrollBar().setValue(self.text_log.verticalScrollBar().maximum())
        QApplication.processEvents()

    def finish(self):
        self.btn_cancel.setEnabled(True)
        self.btn_cancel.setText("Close")
        self.btn_cancel.setStyleSheet("font-weight: bold; background-color: #3d7eff; color: white; border-radius: 4px;")

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

# Main Application Window Class
class FontWidget(QMainWindow):
    ICON_SIZE_LARGE = 128
    ICON_SIZE_SMALL = 64
    PSV_PATH = "glyph_mappings.psv"  # Database file path

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
        self.setting_auto_save_on_switch = _get_bool("auto_save_on_switch", True)
        self.setting_auto_save_timer = _get_bool("auto_save_timer", False)
        self.setting_show_hex_input = _get_bool("show_hex_input", False)

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
        self.menu_structure = None
        self.current_font_glyph_names = []
        self.current_index = 0

        # Dictionaries for data storage
        self.user_glyph_to_char = {}  # Stores current session mappings
        self.font_cache = {}  # Caches extracted font data to avoid re-parsing
        self.known_glyph_hashes = set()  # Stores hashes already in the PSV database
        self.history_stack = []

        # Setup GUI components
        self._setup_menus()
        self._setup_ui()
        self.clear_ui_state()

        # Initialize a timer to handle delayed window snapping after user resizes
        self.resize_snap_timer = QtCore.QTimer(self)
        self.resize_snap_timer.setSingleShot(True)
        self.resize_snap_timer.timeout.connect(self.apply_snap_resize)

        # Window configuration
        self.setMinimumSize(1200, 810)
        self.resize(1200, 810)

        # Base size defines the starting point for the increments
        self.setBaseSize(1200, 810)

        # Force window to resize horizontally by 1px (freely) and vertically by exactly 68px (one list item)
        item_height = self.ICON_SIZE_SMALL + 4
        self.setSizeIncrement(1, item_height)

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

        self.current_pdf_action = toolbar.addAction("Repair PDF")
        current_pdf_icon = qta.icon('fa5s.save', color='white')
        self.current_pdf_action.setIcon(current_pdf_icon)
        self.current_pdf_action.setEnabled(False)
        self.current_pdf_action.triggered.connect(self.repair_current_pdf_100)

        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        toolbar.addWidget(spacer)

        self.font_progress = QProgressBar()
        self.font_progress.setFixedSize(150, 18)

        self.action_progress = toolbar.addWidget(self.font_progress)
        self.action_progress.setVisible(False)

        spacer_small = QWidget()
        spacer_small.setFixedWidth(15)
        toolbar.addWidget(spacer_small)

        self.lbl_toolbar_info = QLabel("Font - of -")
        self.lbl_toolbar_info.setStyleSheet("color: #aaaaaa; font-size: 13px; margin-right: 15px;")
        toolbar.addWidget(self.lbl_toolbar_info)

        spacer_small = QWidget()
        spacer_small.setFixedWidth(15)
        toolbar.addWidget(spacer_small)

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
        self.glyph_list.setVerticalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        self.glyph_list.setIconSize(QtCore.QSize(self.ICON_SIZE_LARGE, self.ICON_SIZE_LARGE))
        self.glyph_list.setSpacing(0)
        font = self.glyph_list.font()
        font.setFamily("Consolas")
        font.setPointSize(32)
        font.setBold(True)
        self.glyph_list.setFont(font)
        self.glyph_list.currentItemChanged.connect(self.on_list_item_changed)
        self.glyph_list.installEventFilter(self)
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
        preview_layout = QHBoxLayout(preview_group)

        self.canvas = GlyphQtWidget()
        self.label = QLabel("Select glyph")
        self.label.setStyleSheet("font-weight: bold; font-size: 24px; color: white;")
        self.label.setAlignment(QtCore.Qt.AlignCenter)

        preview_layout.addWidget(self.canvas)
        preview_layout.addWidget(self.label)

        mapping_group = QGroupBox("Mapping Tools")
        mapping_layout = QVBoxLayout(mapping_group)

        left_panel = QHBoxLayout()
        left_panel.setContentsMargins(0, 0, 0, 0)
        left_panel.setSpacing(10)

        self.suggestions_layout = QHBoxLayout()
        self.suggestions_layout.setAlignment(QtCore.Qt.AlignLeft)
        self.suggestions_layout.setContentsMargins(0, 0, 0, 0)
        self.suggestions_layout.setSpacing(6)

        self.lbl_no_suggestions = QLabel("No suggestions")
        self.lbl_no_suggestions.setStyleSheet("color: #ff4444; font-weight: bold; font-size: 36px; padding-left: 10px;")
        self.lbl_no_suggestions.setMinimumHeight(100)
        self.lbl_no_suggestions.setVisible(False)

        self.suggestions_layout.addWidget(self.lbl_no_suggestions)

        self.suggestion_buttons = []
        for _ in range(4):
            btn = QPushButton("")
            btn.setFixedSize(100, 100)
            font_sug = btn.font()
            font_sug.setPointSize(42)
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

        self.char_input = QLineEdit()
        self.char_input.setPlaceholderText("Character")
        self.char_input.setMaxLength(3)
        self.char_input.returnPressed.connect(self.save_glyph)
        self.char_input.setEnabled(False)
        self.char_input.setStyleSheet("font-family: 'Consolas', monospace; font-size: 32px; font-weight: bold; padding: 5px;")
        self.char_input.setMinimumHeight(50)
        self.char_input.installEventFilter(self)
        self.char_input.textChanged.connect(self.on_user_input_changed)

        self.unic_input = QLineEdit()
        self.unic_input.setPlaceholderText("Unicode Hex")
        self.unic_input.setMaxLength(5)

        hex_validator = QRegularExpressionValidator(QRegularExpression("[0-9a-fA-F]{0,5}"), self)
        self.unic_input.setValidator(hex_validator)

        self.unic_input.returnPressed.connect(self.save_glyph)
        self.unic_input.setEnabled(False)
        self.unic_input.setStyleSheet(
            "font-family: 'Consolas', monospace; font-size: 32px; font-weight: bold; padding: 5px;")
        self.unic_input.setMinimumHeight(50)
        self.unic_input.installEventFilter(self)
        self.unic_input.textChanged.connect(self.on_unic_input_changed)

        self.unic_input.setVisible(self.setting_show_hex_input)

        left_panel.addWidget(self.char_input)
        left_panel.addWidget(self.unic_input)

        user_inputs = QHBoxLayout()
        user_inputs.addLayout(left_panel)


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
        self.btn_glyph.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        self.btn_next_unmapped = QPushButton("Next Unmapped")
        self.btn_next_unmapped.setStyleSheet("font-weight: bold; padding: 5px; min-height: 30px;")
        self.btn_next_unmapped.setEnabled(False)
        self.btn_next_unmapped.clicked.connect(self.jump_to_next_unmapped)
        self.btn_next_unmapped.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        self.btn_prev_mapped = QPushButton("Previously Mapped")
        self.btn_prev_mapped.setStyleSheet("font-weight: bold; padding: 5px; min-height: 30px;")
        self.btn_prev_mapped.setEnabled(False)
        self.btn_prev_mapped.clicked.connect(self.go_back_in_history)
        self.btn_prev_mapped.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.btn_prev_mapped.setFocusPolicy(QtCore.Qt.NoFocus)

        self.btn_font = QPushButton("Save all to DB")
        self.btn_font.setStyleSheet("font-weight: bold; padding: 5px; min-height: 30px;")
        self.btn_font.setEnabled(False)
        self.btn_font.clicked.connect(self.submit_ToUnicode)
        self.btn_font.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        bottom_right_layout.addWidget(self.btn_next_unmapped)
        bottom_right_layout.addWidget(self.btn_prev_mapped)
        bottom_right_layout.addWidget(self.btn_special)
        bottom_right_layout.addWidget(self.btn_glyph)
        bottom_right_layout.addWidget(self.btn_font)

        right_panel.addLayout(bottom_right_layout)

        mapping_layout.addLayout(user_inputs, 1)
        mapping_layout.addLayout(right_panel, 0)

        # Create the Suggestions group box
        suggestions_group = QGroupBox("Suggestions")
        suggestions_group_layout = QVBoxLayout(suggestions_group)
        suggestions_group_layout.setContentsMargins(5, 5, 5, 5)
        suggestions_group_layout.addLayout(self.suggestions_layout)

        # Create a dedicated group box for the glyph list to maintain visual consistency
        list_group = QGroupBox("Glyph List")
        list_layout = QVBoxLayout(list_group)
        list_layout.setContentsMargins(5, 5, 5, 5)
        list_layout.addWidget(self.glyph_list)

        # Top Layout: Glyph List (Left) + Nav & Preview (Right)
        top_layout = QHBoxLayout()
        list_group.setMaximumWidth(800)
        top_layout.addWidget(list_group, 3)

        top_right_layout = QVBoxLayout()
        top_right_layout.addWidget(nav_group, 0)
        top_right_layout.addWidget(preview_group, 1)
        top_layout.addLayout(top_right_layout, 4)

        # Bottom Layout: Suggestions (Left) + Mapping Tools (Right)
        bottom_layout = QHBoxLayout()
        bottom_layout.addWidget(suggestions_group, 3)
        bottom_layout.addWidget(mapping_group, 4)

        # Main Vertical Layout combining Top and Bottom
        main_layout = QVBoxLayout(central)
        main_layout.addLayout(top_layout, 1)
        main_layout.addLayout(bottom_layout, 0)

        self.btn_prev_page.setFocusPolicy(QtCore.Qt.NoFocus)
        self.btn_next_page.setFocusPolicy(QtCore.Qt.NoFocus)
        self.btn_prev_font.setFocusPolicy(QtCore.Qt.NoFocus)
        self.btn_next_font.setFocusPolicy(QtCore.Qt.NoFocus)
        self.btn_select_page.setFocusPolicy(QtCore.Qt.NoFocus)
        self.btn_select_font.setFocusPolicy(QtCore.Qt.NoFocus)
        self.btn_glyph.setFocusPolicy(QtCore.Qt.NoFocus)
        self.btn_font.setFocusPolicy(QtCore.Qt.NoFocus)
        self.btn_special.setFocusPolicy(QtCore.Qt.NoFocus)
        self.btn_next_unmapped.setFocusPolicy(QtCore.Qt.NoFocus)
        self.btn_prev_mapped.setFocusPolicy(QtCore.Qt.NoFocus)

        self.shortcut_prev_font = QShortcut(QKeySequence("Ctrl+Left"), self)
        self.shortcut_prev_font.activated.connect(self.go_to_prev_font)

        self.shortcut_next_font = QShortcut(QKeySequence("Ctrl+Right"), self)
        self.shortcut_next_font.activated.connect(self.go_to_next_font)

    # Opens the settings dialog, applies changes, and saves them persistently
    def open_settings(self):
        dialog = SettingsDialog(self)
        if dialog.exec():
            # Check if critical settings were changed
            page_mode_changed = self.setting_page_mode != dialog.chk_page_mode.isChecked()
            timer_changed = self.setting_auto_save_timer != dialog.chk_auto_save_timer.isChecked()
            hex_visibility_changed = self.setting_show_hex_input != dialog.chk_show_hex_input.isChecked()

            # Update state variables
            self.setting_page_mode = dialog.chk_page_mode.isChecked()
            self.setting_auto_highlight = dialog.chk_auto_highlight.isChecked()
            self.setting_auto_jump_glyph = dialog.chk_auto_jump_glyph.isChecked()
            self.setting_auto_jump_font = dialog.chk_auto_jump_font.isChecked()
            self.setting_auto_save_100 = dialog.chk_auto_save_100.isChecked()
            self.setting_auto_save_on_switch = dialog.chk_auto_save_on_switch.isChecked()
            self.setting_auto_save_timer = dialog.chk_auto_save_timer.isChecked()
            self.setting_show_hex_input = dialog.chk_show_hex_input.isChecked()

            # Persist the new settings to the system
            self.settings_db.setValue("page_mode", self.setting_page_mode)
            self.settings_db.setValue("auto_highlight", self.setting_auto_highlight)
            self.settings_db.setValue("auto_jump_glyph", self.setting_auto_jump_glyph)
            self.settings_db.setValue("auto_jump_font", self.setting_auto_jump_font)
            self.settings_db.setValue("auto_save_100", self.setting_auto_save_100)
            self.settings_db.setValue("auto_save_on_switch", self.setting_auto_save_on_switch)
            self.settings_db.setValue("auto_save_timer", self.setting_auto_save_timer)
            self.settings_db.setValue("show_hex_input", self.setting_show_hex_input)

            # Apply runtime changes
            if page_mode_changed:
                self.update_navigation_labels()
            if timer_changed:
                self.toggle_auto_save_timer(self.setting_auto_save_timer)
            if hex_visibility_changed:
                self.unic_input.setVisible(self.setting_show_hex_input)
                if self.current_font_name and self.current_glyph_set:
                    is_agl = self.current_font_glyph_names[self.current_index] in EXTENDED_AGL
                    self.unic_input.setEnabled(not is_agl)

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
            self.current_page
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

    # Updates the progress bar in the toolbar with the current font's completion status
    # It calculates the live state independently, leaving font_cache untouched for the menus
    def update_progress_bar(self):
        if not self.current_font_glyph_names or self.current_page is None:
            self.action_progress.setVisible(False)
            return

        self.action_progress.setVisible(True)

        # Load base statistics and hashes from the static cache
        info = self.font_cache.get((self.current_page, self.current_font_name), {})
        total = info.get('glyph_count', 0)
        agl_count = info.get('agl_count', 0)
        hashes_dict = info.get('glyph_hashes', {})

        # Calculate live mapped characters for the progress bar only
        current_session_mapped = set()
        for gname in self.current_font_glyph_names:
            if gname in self.user_glyph_to_char:
                current_session_mapped.add(gname)
            elif gname in EXTENDED_AGL:
                current_session_mapped.add(gname)
            elif hashes_dict.get(gname) in self.known_glyph_hashes:
                current_session_mapped.add(gname)

        actual_mapped = len(current_session_mapped)

        # Keep the static cache instantly synchronized with the live UI session
        if (self.current_page, self.current_font_name) in self.font_cache:
            self.font_cache[(self.current_page, self.current_font_name)]['mapped_count'] = actual_mapped

        # Update the visual progress bar widget
        self.font_progress.setMaximum(total)
        self.font_progress.setValue(actual_mapped)

        _, color = self._get_status_info(actual_mapped, total, agl_count)

        self.font_progress.setStyleSheet(f"""
            QProgressBar {{
                border: 1px solid #555;
                border-radius: 4px;
                background-color: #2a2a2a;
                text-align: center;
                color: white;
                font-weight: bold;
                font-size: 11px;
            }}
            QProgressBar::chunk {{
                background-color: {color};
                border-radius: 3px;
            }}
        """)
        self.font_progress.setFormat(f"{actual_mapped} / {total}")

        if self.setting_page_mode:
            self.update_navigation_labels()

    def _get_status_info(self, mapped, total, agl_count=0):
        if total == 0:
            return "—", "#888888" # Gray
        perc = (mapped / total) * 100

        if perc >= 100:
            if agl_count > 0:
                return "100%", "#00CED1"  # Teal/Cyan
            else:
                return "100%", "#228B22"  # Green
        elif perc > 0 or agl_count > 0:
            if agl_count > 0:
                return f"{int(perc)}%", "#3d7eff"  # Blue
            else:
                return f"{int(perc)}%", "#FF8C00"  # Orange
        return "0%", "#888888" # Gray

    # Helper method to create a colored dot icon for the page navigation button
    def _create_status_icon(self, color_str):
        size = 14
        pix = QPixmap(size, size)
        pix.fill(QtCore.Qt.transparent)
        p = QtGui.QPainter(pix)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        p.setPen(QtCore.Qt.NoPen)

        # Draw a two-color pie chart for 100% complete with AGL, or a solid dot otherwise
        if color_str.upper() == "#00CED1":
            p.setBrush(QtGui.QBrush(QtGui.QColor("#3d7eff")))
            p.drawPie(2, 2, size - 4, size - 4, 90 * 16, 180 * 16)

            p.setBrush(QtGui.QBrush(QtGui.QColor("#228B22")))
            p.drawPie(2, 2, size - 4, size - 4, 270 * 16, 180 * 16)
        else:
            p.setBrush(QtGui.QBrush(QtGui.QColor(color_str)))
            p.drawEllipse(2, 2, size - 4, size - 4)

        p.end()
        return QIcon(pix)

    # Resets the UI elements when no font is loaded
    def clear_ui_state(self):
        self.glyph_list.clear()
        self.label.setText("No font loaded")
        self.canvas.draw_glyph(None, None, None, None)
        self.char_input.clear()
        self.char_input.setEnabled(False)
        self.unic_input.clear()
        self.unic_input.setEnabled(False)
        self.btn_glyph.setEnabled(False)
        self.btn_font.setEnabled(False)
        if hasattr(self, 'suggestion_buttons'):
            for btn in self.suggestion_buttons:
                btn.setEnabled(False)

        # Hide progress bar
        if hasattr(self, 'font_progress'):
            self.action_progress.setVisible(False)

        # Reset navigation labels
        self.btn_select_font.setText("No font loaded")
        self.btn_select_page.setText("Page: -")
        self.lbl_toolbar_info.setText("Font - of -")
        self.nav_page_widget.setVisible(False)
        self.unsaved_changes = False
        self._update_window_title()

        if hasattr(self, 'current_pdf_action'):
            self.current_pdf_action.setEnabled(False)

        if hasattr(self, 'suggestion_buttons'):
            for btn in self.suggestion_buttons:
                btn.setText("")
                btn.setEnabled(False)
                btn.setVisible(False)

        if hasattr(self, 'lbl_no_suggestions'):
            self.lbl_no_suggestions.setVisible(False)

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

            page_mapped = 0
            page_total = 0
            page_agl = 0
            for fname in fonts_on_page:
                info = self.font_cache.get((self.current_page, fname), {})
                page_total += info.get('glyph_count', 0)
                page_mapped += info.get('mapped_count', 0)
                page_agl += info.get('agl_count', 0)

            _, color_code = self._get_status_info(page_mapped, page_total, page_agl)
            self.btn_select_page.setIcon(self._create_status_icon(color_code))

            all_pages = sorted(self.menu_structure.keys())
            page_idx = all_pages.index(self.current_page) + 1
            total_pages = len(all_pages)
            self.btn_select_page.setText(f"Page {self.current_page + 1} ({page_idx}/{total_pages})")

        else:
            self.btn_select_page.setIcon(QIcon())

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

    # Dynamically resizes list items to show the selected one larger
    def on_list_item_changed(self, current, previous):
        if previous:
            # Shrink the previously selected item
            pix_large = previous.data(QtCore.Qt.UserRole + 2)
            if pix_large:
                pix_small = pix_large.scaled(
                    self.ICON_SIZE_SMALL, self.ICON_SIZE_SMALL,
                    QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation
                )
                previous.setIcon(QIcon(pix_small))
                previous.setSizeHint(QtCore.QSize(0, self.ICON_SIZE_SMALL + 4))

        if current:
            # Enlarge the newly selected item
            pix_large = current.data(QtCore.Qt.UserRole + 1)
            if pix_large:
                current.setIcon(QIcon(pix_large))
                current.setSizeHint(QtCore.QSize(0, self.ICON_SIZE_LARGE + 4))
            name = current.data(QtCore.Qt.UserRole)
            if name in self.current_font_glyph_names:
                new_index = self.current_font_glyph_names.index(name)

                if new_index != self.current_index:
                    self.current_index = new_index
                    self.show_glyph()

    # Core Logic: Saves the mapping for a single glyph
    def save_glyph(self):
        text_input = self.char_input.text().strip()
        unic_input = self.unic_input.text().strip().lower()
        glyph_name = self.current_font_glyph_names[self.current_index]

        unicode_hex = ""
        agn = ""
        display = ""

        # Priority is given to Hex input if it is valid
        if unic_input:
            # Enforce the 4 to 5 characters rule
            if len(unic_input) < 4:
                QMessageBox.warning(self, "Invalid Length", "Unicode hex code must be at least 4 characters long.")
                return

            unicode_hex = unic_input.zfill(4)  # Ensure at least 4 characters
            try:
                ch = chr(int(unicode_hex, 16))
                display = "[space]" if ch == " " else ch
                agn = UV2AGL.get(int(unicode_hex, 16), "")
            except ValueError:
                QMessageBox.warning(self, "Invalid Unicode",
                                    f"The hex value '{unic_input}' is not a valid Unicode character.")
                return
        elif not text_input:
            text_input = " "
            unicode_hex = "0020"
            agn = "space"
            display = "[space]"
        elif len(text_input) == 1:
            unicode_hex = format(ord(text_input), '04x')
            agn = UV2AGL.get(ord(text_input), "")
            display = text_input
        else:
            if text_input in self.KNOWN_LIGATURES:
                unicode_hex = self.KNOWN_LIGATURES[text_input]
                agn = UV2AGL.get(int(unicode_hex, 16), text_input)
                display = text_input
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
        item.setText(f" → {display}")
        item.setForeground(QtGui.QColor("#228B22"))  # Set to green

        self.char_input.clear()
        self.unic_input.clear()

        self.update_progress_bar()

        # Calculate completion accurately including DB hashes
        info = self.font_cache.get((self.current_page, self.current_font_name), {})
        hashes_dict = info.get('glyph_hashes', {})

        mapped_count = 0
        for g in self.current_font_glyph_names:
            if g in self.user_glyph_to_char:
                mapped_count += 1
            elif g in EXTENDED_AGL:
                mapped_count += 1
            elif hashes_dict.get(g) in self.known_glyph_hashes:
                mapped_count += 1

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
            current_pos = (self.current_page, self.current_font_name, self.current_index)
            if not hasattr(self, 'history_stack'):
                self.history_stack = []
            if not self.history_stack or self.history_stack[-1] != current_pos:
                self.history_stack.append(current_pos)

            # Check remaining glyphs in the current font
        if self.current_font_glyph_names:
            for i in range(self.current_index + 1, len(self.current_font_glyph_names)):
                gname = self.current_font_glyph_names[i]
                if gname not in self.user_glyph_to_char and gname not in EXTENDED_AGL:
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

        # Check next fonts
        for p, fname in ordered_seq:
            if p == self.current_page and fname == self.current_font_name:
                continue

            info = self.font_cache.get((p, fname), {})
            mapped = info.get('mapped_count', 0)
            agl_c = info.get('agl_count', 0)
            total = info.get('glyph_count', 0)

            # Check if font has any non-AGL and unmapped glyphs left
            if (mapped + agl_c) < total:
                if self.unsaved_changes:
                    self.save_to_db()

                self.load_font(p, fname)
                for i, gname in enumerate(self.current_font_glyph_names):
                    if gname not in self.user_glyph_to_char and gname not in EXTENDED_AGL:
                        self.current_index = i
                        self.show_glyph()
                        return

        # Wrap around to the beginning of the current font
        if self.current_font_glyph_names:
            for i in range(0, self.current_index):
                gname = self.current_font_glyph_names[i]
                if gname not in self.user_glyph_to_char and gname not in EXTENDED_AGL:
                    self.current_index = i
                    self.show_glyph()
                    return

        QMessageBox.information(self, "Finished", "Great! No more unmapped non-AGL glyphs found.")

    def go_back_in_history(self):
        if not hasattr(self, 'history_stack') or not self.history_stack:
            QMessageBox.information(self, "Info", "Historie je prázdná, není kam se vrátit.")
            return

        prev_page, prev_font, prev_index = self.history_stack.pop()

        if self.current_page != prev_page or self.current_font_name != prev_font:
            if getattr(self, 'unsaved_changes', False):
                self.save_to_db()
            self.load_font(prev_page, prev_font)

        if self.current_font_glyph_names and 0 <= prev_index < len(self.current_font_glyph_names):
            self.current_index = prev_index
            self.show_glyph()

    # Opens a web helper for finding symbols
    def open_special(self):
        webbrowser.open_new_tab("https://www.vertex42.com/ExcelTips/unicode-symbols.html")

    # Calculates or retrieves MD5 hash of the glyph shape
    def get_glyph_hash(self, glyph_name):
        if hasattr(self, 'current_page') and hasattr(self, 'current_font_name'):
            cache = self.font_cache.get((self.current_page, self.current_font_name), {})
            cached_hashes = cache.get('glyph_hashes', {})
            if glyph_name in cached_hashes:
                return cached_hashes[glyph_name]

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

    # Loads a specific font from the PDF into memory and UI
    def load_font(self, page, font_name):
        # Save any unsaved progress before switching to a new font/page if setting is enabled
        if self.setting_auto_save_on_switch and getattr(self, 'unsaved_changes', False):
            self.save_to_db()

        self.current_page = page
        self.current_font_name = font_name
        self._update_window_title()

        # Check cache first to avoid slow PDF extraction
        cache = self.font_cache.get((page, font_name))
        if not cache:
            self.statusBar().showMessage(f"Cache empty", 5000)

        try:
            self.setEnabled(False)
            QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
            QApplication.processEvents()

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
            self.char_input.setEnabled(True)
            self.unic_input.setEnabled(True)
            self.btn_glyph.setEnabled(True)
            self.btn_font.setEnabled(True)
            self.btn_next_unmapped.setEnabled(True)
            self.btn_prev_mapped.setEnabled(True)

            if hasattr(self, 'suggestion_buttons'):
                for btn in self.suggestion_buttons:
                    btn.setEnabled(True)

            self.statusBar().showMessage(f"Loaded: {font_name} (Page {page + 1})", 5000)

            if not self.current_font_glyph_names:
                self.clear_ui_state()

            # Update dynamic navigation labels
            self.update_navigation_labels()
            self.update_progress_bar()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error while loading font:\n{e}")

        finally:
            self.setEnabled(True)
            self.glyph_list.setFocus()
            QApplication.restoreOverrideCursor()

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
            pen = QtPen(glyphSet)
            glyphSet['.notdef'].draw(pen)
            if not pen.path.isEmpty():
                rect = pen.path.boundingRect()
                notdef_baseline = rect.top()
                notdef_topline = rect.bottom()

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

    # Generates a thumbnail image of a glyph natively via Qt (Hyper-optimized)
    def generate_icon(self, glyph_name, size=(128, 128), draw_lines=False):
        pix = QPixmap(size[0], size[1])
        pix.fill(QtCore.Qt.white)

        if not hasattr(self, 'current_glyph_set') or glyph_name not in self.current_glyph_set:
            return pix

        # Draw the glyph path using our fast QtPen
        glyph = self.current_glyph_set[glyph_name]
        pen = QtPen(self.current_glyph_set)
        glyph.draw(pen)

        if pen.path.isEmpty():
            return pix

        # Determine reference heights for scaling
        if self.notdef_baseline is not None and self.notdef_topline is not None:
            ref_min = self.notdef_baseline
            ref_max = self.notdef_topline
        else:
            ref_max = getattr(self.current_font, 'FontBBox', [0, 0, 0, 1000])[3]
            ref_min = getattr(self.current_font, 'FontBBox', [0, -200, 0, 0])[1]

        ref_height = max(ref_max - ref_min, 1)
        ref_midpoint = (ref_max + ref_min) / 2

        # Bounding box for horizontal centering
        br = pen.path.boundingRect()
        center_x = br.center().x()

        # Set up QPainter
        painter = QtGui.QPainter(pix)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)

        # Center in the pixmap
        painter.translate(size[0] / 2.0, size[1] / 2.0)

        # Scale the path (Y must be negative because Qt's Y axis points down, unlike PDF)
        scale = (size[1] * 0.45) / ref_height
        painter.scale(scale, -scale)

        # Move the specific glyph to the exact visual center
        painter.translate(-center_x, -ref_midpoint)

        # Draw the solid black shape
        painter.setBrush(QtGui.QBrush(QtGui.QColor("black")))
        painter.setPen(QtCore.Qt.NoPen)
        painter.drawPath(pen.path)

        # Draw optional alignment lines for the selected item
        if draw_lines:
            line_pen = QtGui.QPen()
            line_pen.setStyle(QtCore.Qt.DotLine)
            line_pen.setWidthF(0)  # 0 means a 1-pixel cosmetic line independent of zoom scale

            painter.setBrush(QtCore.Qt.NoBrush)
            if self.notdef_baseline is not None:
                line_pen.setColor(QtGui.QColor("blue"))
                painter.setPen(line_pen)
                painter.drawLine(QtCore.QLineF(-10000, self.notdef_baseline, 10000, self.notdef_baseline))
                painter.drawLine(QtCore.QLineF(-10000, self.notdef_topline, 10000, self.notdef_topline))
            else:
                line_pen.setColor(QtGui.QColor("red"))
                painter.setPen(line_pen)
                painter.drawLine(QtCore.QLineF(-10000, 0, 10000, 0))

        painter.end()
        return pix

    # Fills the QListWidget with glyph thumbnails
    def populate_glyph_list(self):
        w = self.glyph_list
        w.blockSignals(True)
        w.clear()

        for name in self.current_font_glyph_names:
            # Generate the clean version for the unselected small state
            pix_clean = self.generate_icon(name, size=(self.ICON_SIZE_LARGE, self.ICON_SIZE_LARGE), draw_lines=False)

            # Create a small version for the default unselected state
            pix_small = pix_clean.scaled(
                self.ICON_SIZE_SMALL, self.ICON_SIZE_SMALL,
                QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation
            )

            # Generate the version with guidelines for the large selected state
            pix_large_lines = self.generate_icon(name, size=(self.ICON_SIZE_LARGE, self.ICON_SIZE_LARGE), draw_lines=True)

            item = QListWidgetItem(QIcon(pix_small), "")
            item.setData(QtCore.Qt.UserRole, name)

            # Cache the large pixmap with lines in the item itself using UserRole + 1
            item.setData(QtCore.Qt.UserRole + 1, pix_large_lines)

            # Cache the clean small pixmap so we can restore it later using UserRole + 2
            item.setData(QtCore.Qt.UserRole + 2, pix_small)

            # Set default small height
            item.setSizeHint(QtCore.QSize(0, self.ICON_SIZE_SMALL + 4))

            # If already mapped in database, show result
            if name in self.user_glyph_to_char:
                ch = chr(int(self.user_glyph_to_char[name]["unicode_hex"], 16))
                disp = "[space]" if ch.isspace() else ch
                item.setText(f" → {disp}")
                item.setForeground(QtGui.QColor("#228B22"))
            elif name in EXTENDED_AGL:
                ch = chr(EXTENDED_AGL[name])
                disp = "[space]" if ch.isspace() else ch
                item.setText(f" → {disp}")
                item.setForeground(QtGui.QColor("#3d7eff"))  # Blue text to signify AGL mapped
            else:
                item.setText(f" {name}")
                item.setForeground(QtGui.QColor("#888888"))

            w.addItem(item)

        w.blockSignals(False)

    # Updates the main canvas area with the selected glyph
    def show_glyph(self):
        name = self.current_font_glyph_names[self.current_index]
        self.canvas.draw_glyph(self.current_glyph_set, name, self.notdef_topline, self.notdef_baseline)

        # Retrieve mapping info if available
        mapping = self.user_glyph_to_char.get(name, {})
        uhex = mapping.get("unicode_hex", "None")
        agn = mapping.get("AGN", "None")

        # If it's an AGL glyph, prioritize showing its inherent AGL value
        is_agl = name in EXTENDED_AGL
        if is_agl:
            uhex = format(EXTENDED_AGL[name], '04x').upper()
            agn = name

        ch = chr(int(uhex, 16)) if uhex != "None" else "None"
        if ch == " ": ch = "[space]"

        # Update Information Label using HTML formatting
        html = f"""
                        <div style="text-align: center; margin-top: 10px;">
                            <span style="color: #aaaaaa; font-size: 28px;">Glyph Name</span><br>
                            <span style="font-family: Consolas, monospace; font-size: 28px; font-weight: bold; color: white;">{name}</span>
                            <hr width="95%" color="#aaaaaa" style="margin-top: 6px; margin-bottom: 6px;">

                            <span style="color: #aaaaaa; font-size: 28px;">Character</span><br>
                            <span style="font-family: Consolas, monospace; font-size: 28px; font-weight: bold; color: white;">{ch}</span>
                            <hr width="95%" color="#aaaaaa" style="margin-top: 6px; margin-bottom: 6px;">

                            <span style="color: #aaaaaa; font-size: 28px;">Unicode</span><br>
                            <span style="font-family: Consolas, monospace; font-size: 28px; font-weight: bold; color: white;">{uhex}</span>
                            <hr width="95%" color="#aaaaaa" style="margin-top: 6px; margin-bottom: 6px;">

                            <span style="color: #aaaaaa; font-size: 28px;">Adobe Glyph List</span><br>
                            <span style="font-family: Consolas, monospace; font-size: 28px; font-weight: bold; color: white;">{agn}</span>
                        </div>
                        """
        self.label.setText(html)

        # Ensure the item is selected and visible in list
        item = self.glyph_list.item(self.current_index)
        if item:
            self.glyph_list.setCurrentItem(item)

            # Force the list to immediately recalculate heights after dynamic resize
            self.glyph_list.doItemsLayout()

            # Align the selected item exactly to the bottom edge of the viewport
            self.glyph_list.scrollToItem(item, QListWidget.ScrollHint.EnsureVisible)

        # Lock inputs and disable suggestions if it's a standard AGL glyph
        if is_agl:
            self.char_input.setEnabled(False)
            self.char_input.setFocusPolicy(QtCore.Qt.NoFocus)
            self.unic_input.setEnabled(False)
            self.unic_input.setFocusPolicy(QtCore.Qt.NoFocus)

            self.btn_glyph.setEnabled(False)
            self.char_input.setPlaceholderText("AGL Auto")
            self.unic_input.setPlaceholderText("AGL Auto")
            for btn in self.suggestion_buttons:
                btn.setEnabled(False)
                btn.setVisible(False)
        else:
            self.char_input.setEnabled(True)
            self.unic_input.setEnabled(self.setting_show_hex_input)

            self.btn_glyph.setEnabled(True)
            self.char_input.setPlaceholderText("Character")
            self.unic_input.setPlaceholderText("Unicode Hex")
            self.update_suggestions_ui(name, self.current_font_name)
            self.char_input.setFocus()

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
            self.menu_structure = {}
            self.font_cache.clear()

            with fitz.open(file_path) as doc:
                first_page = first_name = None

                # Iterate through all pages
                for page_num in range(len(doc)):
                    page = doc.load_page(page_num)
                    cff_names_on_page = []

                    # Analyze fonts on each page
                    for font in page.get_fonts(full=True):
                        try:
                            xref = font[0]
                            name, ext, _, buffer = doc.extract_font(xref)
                            # Only process CFF fonts
                            if ext and ext.lower() == "cff":

                                if has_tounicode(doc, xref):
                                    continue

                                cff_names_on_page.append(name)

                                if (page_num, name) not in self.font_cache:
                                    tmp_font = CFFFontSet()
                                    tmp_font.decompile(BytesIO(buffer), None)
                                    glyph_set = tmp_font.topDictIndex[0].CharStrings

                                    # Fiter out .notdef
                                    valid_glyph_names = [g for g in glyph_set.keys() if g != '.notdef']
                                    total_glyphs = len(valid_glyph_names)

                                    current_font_hashes = {}
                                    agl_count = 0
                                    for gname in valid_glyph_names:
                                        if gname in EXTENDED_AGL:
                                            agl_count += 1

                                        glyph = glyph_set[gname]
                                        pen = SignaturePen(glyph_set)
                                        glyph.draw(pen)

                                        # Detect completely empty glyphs and assign a special hash
                                        if not pen.signature:
                                            ghash = md5("EMPTY_SPACE".encode('utf-8')).hexdigest()
                                        else:
                                            sig = pen.get_signature()
                                            ghash = md5(sig.encode('utf-8')).hexdigest()
                                        current_font_hashes[gname] = ghash

                                    mapped_count = self.calculate_font_mapped_count(page_num, name, current_font_hashes)

                                    #Cache the data
                                    self.font_cache[(page_num, name)] = {
                                        'glyph_count': total_glyphs,
                                        'mapped_count': mapped_count,
                                        'agl_count': agl_count,
                                        'glyph_hashes': current_font_hashes,
                                        'data': buffer
                                    }

                                if first_page is None:
                                    first_page, first_name = page_num, name
                        except Exception as e:
                            print(f"Error parsing font {name}: {e}")

                    if cff_names_on_page:
                        self.menu_structure[page_num] = cff_names_on_page

            if first_page is not None:
                self.load_font(first_page, first_name)
                self.current_pdf_action.setEnabled(True)
            else:
                self.clear_ui_state()
                QMessageBox.information(
                    self,
                    "Nothing to repair",
                    "This document does not contain any fonts to be repaired.\n\n"
                )
                self.statusBar().showMessage("Nothing to repair", 5000)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error while loading PDF:\n{e}")
            self.clear_ui_state()

    # Helper method to robustly calculate mapped glyphs for any given font
    def calculate_font_mapped_count(self, page, font_name, glyph_hashes):
        mapped_count = 0
        clean_font_name = font_name.split('+', 1)[-1] if '+' in font_name else font_name

        # If calculating for the currently active font, include live unsaved mappings
        is_current = (page == self.current_page and font_name == self.current_font_name)

        for gname, h in glyph_hashes.items():
            if gname in EXTENDED_AGL:
                mapped_count += 1
            elif is_current and gname in getattr(self, 'user_glyph_to_char', {}):
                mapped_count += 1
            elif (h, clean_font_name, gname) in getattr(self, 'exact_db_matches', set()):
                mapped_count += 1
            elif h in self.known_glyph_hashes:
                mapped_count += 1

        return mapped_count

    # Refreshes menu statistics after a DB update
    def update_statistics(self):
        self.load_db_cache()
        for (p, fname), info in self.font_cache.items():
            hashes_dict = info.get('glyph_hashes', {})
            if not hashes_dict: continue

            # Use the robust calculation method
            info['mapped_count'] = self.calculate_font_mapped_count(p, fname, hashes_dict)

    def load_db_cache(self):
        hash_counts = {}
        space_hash = md5("EMPTY_SPACE".encode('utf-8')).hexdigest()
        hash_counts[space_hash] = {"0020"}

        self.db_records = []
        #Track exact matches for specific fonts to solve global hash collisions
        self.exact_db_matches = set()

        self.global_db_map = {}
        self.global_db_map[space_hash] = [{"unicode_hex": "0020", "font_name": "", "GlyphName": "space", "AGN": "space"}]

        path = self.PSV_PATH
        if os.path.exists(path):
            try:
                with open(path, 'r', encoding='utf-8', newline='') as f:
                    reader = csv.DictReader(f, delimiter='|', quotechar='"')
                    if "glyph_hash" in reader.fieldnames:
                        for row in reader:
                            h = row["glyph_hash"]
                            u = row["unicode_hex"]
                            # Clean the font name from subset prefixes (e.g., ABCDEF+)
                            fname = row.get("font_name", "").split('+', 1)[-1] if '+' in row.get("font_name",
                                                                                                 "") else row.get(
                                "font_name", "")
                            gname = row.get("GlyphName", "")

                            if h not in hash_counts:
                                hash_counts[h] = set()
                            hash_counts[h].add(u)

                            self.db_records.append(row)
                            self.exact_db_matches.add((h, fname, gname))

                            if h not in self.global_db_map:
                                self.global_db_map[h] = []
                            self.global_db_map[h].append(row)
            except Exception as e:
                print(f"DB Cache Error: {e}")

        self.known_glyph_hashes = {h for h, unics in hash_counts.items() if len(unics) == 1}

    # Generates suggestions based on GlyphName and fuzzy matching of font_name
    def get_suggestions(self, glyph_name, font_name, current_hash=None):
        if not hasattr(self, 'db_records') or not self.db_records or not glyph_name or not font_name:
            return []

        matches = []
        for row in self.db_records:
            db_glyph_name = row.get("GlyphName", "")
            glyph_sim = difflib.SequenceMatcher(None, glyph_name, db_glyph_name).ratio()

            if current_hash and row.get("glyph_hash") == current_hash:
                matches.append((1.0, glyph_sim, row.get("unicode_hex")))

            elif db_glyph_name == glyph_name:
                matches.append((0.5, glyph_sim, row.get("unicode_hex")))

        matches.sort(key=lambda x: (x[0], x[1]), reverse=True)

        suggestions = []
        for _, _, hex_val in matches:
            try:
                char = chr(int(hex_val, 16))
                # Add unique characters until we have 4 (for our 4 buttons)
                if char not in suggestions:
                    suggestions.append(char)
                if len(suggestions) >= 4:
                    break
            except (ValueError, TypeError):
                pass

        return suggestions

    # Refreshes the suggestion buttons above the text input
    def update_suggestions_ui(self, glyph_name, font_name):
        current_hash = self.get_glyph_hash(glyph_name)
        suggestions = self.get_suggestions(glyph_name, font_name, current_hash)
        self.active_suggestions_count = len(suggestions)

        if self.active_suggestions_count == 0:
            self.lbl_no_suggestions.setVisible(True)
        else:
            self.lbl_no_suggestions.setVisible(False)

        for i, btn in enumerate(self.suggestion_buttons):
            if i < len(suggestions):
                char = suggestions[i]

                # Qt uses '&' for keyboard shortcuts. To display a literal '&', it must be escaped as '&&'.
                display_char = char.replace('&', '&&')

                btn.setText(display_char)
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
                    "font-family: 'Consolas', monospace; border: 3px solid #3d7eff; background-color: white; border-radius: 4px; color: black;")
            else:
                # Default style - dark gray border
                btn.setStyleSheet(
                    "font-family: 'Consolas', monospace; border: 1px solid #555; background-color: white; border-radius: 4px; color: black;")

    # Removes highlight if the user starts typing manually
    def on_user_input_changed(self, text):
        if text and self.current_suggestion_idx != -1:
            self.set_suggestion_highlight(-1)
        elif not text and self.setting_auto_highlight and self.active_suggestions_count > 0:
            # Re-highlight the first button if the user deletes their text
            self.set_suggestion_highlight(0)

    def on_unic_input_changed(self, text):
        if text and self.current_suggestion_idx != -1:
            self.set_suggestion_highlight(-1)
        elif not text and self.setting_auto_highlight and self.active_suggestions_count > 0:
            self.set_suggestion_highlight(0)

        # Catches keyboard events in the char_input field for suggestion navigation
    def eventFilter(self, obj, event):
        if event.type() == QtCore.QEvent.KeyPress:

            if event.modifiers() & QtCore.Qt.ControlModifier:
                if event.key() == QtCore.Qt.Key_Left:
                    self.go_to_prev_font()
                    return True
                elif event.key() == QtCore.Qt.Key_Right:
                    self.go_to_next_font()
                    return True

            if obj in (self.char_input, self.unic_input, self.glyph_list):
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

            if obj in (self.char_input, self.unic_input):
                if not self.setting_auto_highlight:
                    return super().eventFilter(obj, event)

                # Left arrow
                if event.key() == QtCore.Qt.Key_Left:
                    if obj.text():
                        return False
                    if self.active_suggestions_count > 0:
                        new_idx = max(0, self.current_suggestion_idx - 1)
                        if self.current_suggestion_idx == -1: new_idx = 0
                        self.set_suggestion_highlight(new_idx)
                        return True

                # Right arrow
                elif event.key() == QtCore.Qt.Key_Right:
                    if obj.text():
                        return False
                    if self.active_suggestions_count > 0:
                        new_idx = min(self.active_suggestions_count - 1, self.current_suggestion_idx + 1)
                        if self.current_suggestion_idx == -1: new_idx = 0
                        self.set_suggestion_highlight(new_idx)
                        return True

                # Enter or Return key
                elif event.key() in (QtCore.Qt.Key_Return, QtCore.Qt.Key_Enter):
                    # Apply highlighted suggestion ONLY if the text box is completely empty
                    # AND a valid suggestion is currently highlighted
                    if not obj.text().strip() and self.current_suggestion_idx >= 0:
                        try:
                            btn = self.suggestion_buttons[self.current_suggestion_idx]
                            if btn.isVisible():
                                self.apply_suggestion(btn.suggestion_char)
                                return True  # Block the event, we handled it
                        except IndexError:
                            pass  # Failsafe in case active_suggestions_count is out of sync

                    # If text box is NOT empty, let the normal returnPressed signal handle it
                    return False

        # Všechny ostatní nestřežené eventy propustíme zpět domů
        return super().eventFilter(obj, event)

    # Automatically fills input and triggers the save mechanism
    def apply_suggestion(self, char):
        self.char_input.setText(char)
        self.save_glyph()

    # Saves current session work to the PSV file
    def save_to_db(self):
        path = self.PSV_PATH
        fieldnames = ["glyph_hash", "font_name", "GlyphName", "unicode_hex", "AGN"]

        existing_data = {}
        # Read existing data first to preserve it
        if os.path.exists(path):
            try:
                with open(path, 'r', encoding='utf-8', newline='') as f:
                    reader = csv.DictReader(f, delimiter='|', quotechar='"')
                    if "glyph_hash" in reader.fieldnames:
                        for row in reader:
                            key = (row["glyph_hash"], row.get("font_name", ""), row.get("GlyphName", ""))
                            existing_data[key] = row
            except Exception:
                pass

        count_new = 0
        current_font_name = self.current_font_name or "unknown"

        # Update existing data with new mappings
        for gname, data in self.user_glyph_to_char.items():
            # Skip saving strictly AGL glyphs to the database
            if gname in EXTENDED_AGL:
                continue

            g_hash = data.get("glyph_hash") or self.get_glyph_hash(gname)

            if g_hash:
                key = (g_hash, current_font_name, gname)
                existing_data[key] = {
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
                writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter='|', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                writer.writeheader()
                for row in existing_data.values():
                    writer.writerow(row)

            # Refresh application state
            self.load_db_cache()
            self.update_statistics()
            self.statusBar().showMessage(f"Saved. Total DB size: {len(existing_data)}", 3000)

            self.unsaved_changes = False

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save:\n{e}")

    # Checks the database for any glyphs in the current font that we already know
    def load_mappings_for_current_font(self):
        self.user_glyph_to_char = {}

        if not hasattr(self, 'global_db_map'):
            self.load_db_cache()
        db_map = self.global_db_map

        current_clean_font = self.current_font_name.split('+', 1)[
            -1] if self.current_font_name and '+' in self.current_font_name else (self.current_font_name or "")

        cached_hashes = self.font_cache.get((self.current_page, self.current_font_name), {}).get('glyph_hashes', {})

        # Check each glyph in current font against DB
        for name in self.current_font_glyph_names:

            # Absolute AGL priority
            if name in EXTENDED_AGL and name != '.notdef':
                continue

            g_hash = cached_hashes.get(name) or self.get_glyph_hash(name)

            if g_hash and g_hash in db_map:
                rows = db_map[g_hash]

                exact_match = None
                for r in rows:
                    db_clean_font = r.get("font_name", "").split('+', 1)[-1] if '+' in r.get("font_name", "") else r.get("font_name", "")
                    if r.get("GlyphName") == name and db_clean_font == current_clean_font:
                        exact_match = r
                        break

                if exact_match:
                    self.user_glyph_to_char[name] = {
                        "glyph_hash": g_hash,
                        "unicode_hex": exact_match["unicode_hex"],
                        "AGN": exact_match["AGN"]
                    }
                else:
                    unique_hexes = list(set(r["unicode_hex"] for r in rows))
                    if len(unique_hexes) == 1:
                        row = next(r for r in rows if r["unicode_hex"] == unique_hexes[0])
                        self.user_glyph_to_char[name] = {
                            "glyph_hash": g_hash,
                            "unicode_hex": row["unicode_hex"],
                            "AGN": row["AGN"]
                        }

        # Special handling for .notdef
        if '.notdef' in self.current_glyph_set:
            nhash = cached_hashes.get('.notdef') or self.get_glyph_hash('.notdef')
            if nhash in db_map:
                row = db_map[nhash][0]
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

    # Overrides the default resize event to trigger the snapping timer
    # This prevents fighting with the OS window manager while the user is actively dragging
    def resizeEvent(self, event):
        super().resizeEvent(event)
        # Restart the timer on every pixel change; it will only fire when resizing stops for 500ms
        self.resize_snap_timer.start(500)

    # Calculates the nearest valid height based on list item increments and applies it
    def apply_snap_resize(self):
        current_height = self.height()
        current_width = self.width()

        base_height = 810
        item_height = 68  # 64px icon + 4px margin

        # Calculate how many steps we are away from the base height
        diff = current_height - base_height
        steps = round(diff / item_height)

        # Determine the exact target height
        target_height = base_height + (steps * item_height)

        # Apply the resize only if the window is not already at the perfect height
        if current_height != target_height:
            self.resize(current_width, target_height)

    def repair_current_pdf_100(self):
        if not self.pdf_path: return
        if getattr(self, 'unsaved_changes', False): self.save_to_db()

        page_font_map = {}
        all_unique_xrefs = set()
        ready_xrefs_set = set()

        with fitz.open(self.pdf_path) as doc_temp:
            for (p_num, f_name), info in self.font_cache.items():
                if p_num not in page_font_map: page_font_map[p_num] = []
                found_xref = None
                for f in doc_temp.load_page(p_num).get_fonts(full=True):
                    name, ext, _, _ = doc_temp.extract_font(f[0])
                    if ext == "cff" and name == f_name:
                        found_xref = f[0]
                        break

                f_stats = {'name': f_name, 'mapped': info.get('mapped_count', 0), 'total': info.get('glyph_count', 0),
                           'xref': found_xref}
                page_font_map[p_num].append(f_stats)

                if found_xref:
                    all_unique_xrefs.add(found_xref)
                    if f_stats['total'] > 0 and f_stats['mapped'] >= f_stats['total']:
                        ready_xrefs_set.add(found_xref)

        total_unique_cff_fonts = len(all_unique_xrefs)
        ready_xrefs = list(ready_xrefs_set)

        dialog = IntegratedRepairDialog(page_font_map, total_unique_cff_fonts, len(ready_xrefs), self)
        result = dialog.exec()

        if result == QDialog.Accepted + 1:
            p_num, f_name = dialog.jump_target
            if f_name:
                self.load_font(p_num, f_name)
            else:
                fonts_on_page = self.menu_structure.get(p_num, [])
                if fonts_on_page:
                    self.load_font(p_num, fonts_on_page[0])
            self.statusBar().showMessage(f"Jump to font/page {f_name}", 3000)
            return

        if result == QDialog.Accepted:
            dialog.show()

            try:
                dialog.log("=========================================")
                dialog.log("          PDF Repair started...          ")
                dialog.log("=========================================")

                db_map = load_db(self.PSV_PATH)
                custom_flags = fitz.TEXT_PRESERVE_LIGATURES | fitz.TEXT_INHIBIT_SPACES | fitz.TEXT_USE_CID_FOR_UNKNOWN_UNICODE | fitz.TEXT_PRESERVE_WHITESPACE

                doc_vizual = fitz.open(self.pdf_path)
                font_cache_local = {}

                dialog.log("[INFO] Visual order scanning...")
                vizualni_sekvence = load_sequence(doc_vizual, custom_flags, db_map, font_cache_local, rezim="vizual")
                vizualni_sekvence = {x: seq for x, seq in vizualni_sekvence.items() if x in ready_xrefs}

                ready_fonts_count = len(vizualni_sekvence)
                dialog.log(
                    f"[INFO] Found {ready_fonts_count} fonts out of {total_unique_cff_fonts} for reconscruction.\n")

                dialog.log("[1/3] DUMMY ToUnicode injection")
                dialog.log("-" * 40)
                dummy_cmap_str = generate_dummy_tounicode()

                for idx, xref in enumerate(vizualni_sekvence.keys(), 1):
                    dummy_xref = doc_vizual.get_new_xref()
                    doc_vizual.update_object(dummy_xref, "<<>>")
                    doc_vizual.update_stream(dummy_xref, dummy_cmap_str.encode("utf-8"))
                    doc_vizual.xref_set_key(xref, "ToUnicode", f"{dummy_xref} 0 R")

                    font_name = font_cache_local.get(xref, {}).get("name", "Not found")
                    dialog.log(f"  [{idx}/{ready_fonts_count}] Font '{font_name}'")
                    dialog.log(f"      -> New DUMMY xref: {dummy_xref}")

                dialog.log("\n[INFO] Saving PDF to cache ...")
                dummy_pdf_bytes = doc_vizual.tobytes()
                doc_vizual.close()

                dialog.log("\n[2/3] ChatacterCode extraction")
                dialog.log("-" * 40)
                doc_dummy = fitz.open("pdf", dummy_pdf_bytes)
                interni_sekvence = load_sequence(doc_dummy, custom_flags, db_map, font_cache_local, rezim="dummy")
                doc_dummy.close()
                dialog.log(f"  -> Extraction succeeded for {len(interni_sekvence)} fonts.")

                dialog.log("\n[3/3] Final ToUnicode mapping")
                dialog.log("-" * 40)
                doc_final = fitz.open(self.pdf_path)
                success_count = 0

                for idx, (xref, v_seq) in enumerate(vizualni_sekvence.items(), 1):
                    i_seq = interni_sekvence.get(xref, [])
                    font_name = font_cache_local.get(xref, {}).get("name", "Not found")

                    dialog.log(f"  [{idx}/{ready_fonts_count}] Processing font: '{font_name}'")
                    dialog.log(f"      -> Characters detected - Visual: {len(v_seq)} | Internal: {len(i_seq)}")

                    if len(v_seq) == len(i_seq):
                        mapping = {}
                        for v_hex, i_id in zip(v_seq, i_seq):
                            if i_id not in mapping:
                                mapping[i_id] = v_hex

                        real_cmap = generate_real_tounicode(mapping)
                        new_xref = doc_final.get_new_xref()
                        doc_final.update_object(new_xref, "<<>>")
                        doc_final.update_stream(new_xref, real_cmap.encode("utf-8"))
                        doc_final.xref_set_key(xref, "ToUnicode", f"{new_xref} 0 R")

                        success_count += 1
                        dialog.log(f"      -> ToUnicode generated (Size: {len(real_cmap)} bytes)")
                        dialog.log(f"      -> Injected under xref: {new_xref}")
                        dialog.log(f"      -> [SUCCESS] {len(mapping)} unique glyphs mapped")
                    else:
                        dialog.log(f"      -> [ERROR] Sequence lengths do not match, skipping font...")

                dialog.log("\n[INFO] Cleanup and file saving to disk...")
                base, ext = os.path.splitext(self.pdf_path)

                if success_count == total_unique_cff_fonts:
                    out_path = f"{base}_Repaired{ext}"
                else:
                    out_path = f"{base}_Partially_Repaired{ext}"

                doc_final.save(out_path)
                doc_final.close()

                dialog.log("\n" + "=" * 45)
                dialog.log("             Repair finished                ")
                dialog.log("=" * 45)
                dialog.log(f"Repaired file: {os.path.basename(out_path)}")
                dialog.log(f"Fonts reconstructed: {success_count} / {total_unique_cff_fonts}")
                if success_count == total_unique_cff_fonts:
                    dialog.log("[STATUS] All fonts in pdf were repaired successfully.")
                else:
                    dialog.log("[STATUS] Finished with errors, some fonts skipped...")
                dialog.log("=" * 45)

                dialog.finish()
                dialog.exec()

            except Exception as e:
                dialog.log(f"\n[CRITICAL ERROR] Something went wrong:\n{str(e)}")
                dialog.finish()
                dialog.exec()

def load_db(psv_path="glyph_mappings.psv"):
    db_map = {}
    space_hash = md5("EMPTY_SPACE".encode('utf-8')).hexdigest()
    db_map[space_hash] = [{"unicode_hex": "0020", "font_name": "", "GlyphName": "space"}]

    if not os.path.exists(psv_path):
        return db_map

    try:
        with open(psv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter='|')
            for row in reader:
                g_hash = row.get("glyph_hash", "")
                u_hex = row.get("unicode_hex", "")
                if not u_hex or not g_hash: continue
                if g_hash == space_hash and u_hex != "0020": continue

                f_name = row.get("font_name", "").split('+')[-1]
                g_name = row.get("GlyphName", "")

                if g_hash not in db_map: db_map[g_hash] = []
                db_map[g_hash].append({"unicode_hex": u_hex, "font_name": f_name, "GlyphName": g_name})
    except Exception as e:
        print(f"DB ERROR ({e}).")
    return db_map

def get_standalone_glyph_hash(glyph_set, g_name):
    if not glyph_set or g_name not in glyph_set: return None
    try:
        glyph = glyph_set[g_name]
        pen = SignaturePen(glyph_set)
        glyph.draw(pen)
        sig = pen.get_signature()
        if not sig: sig = "EMPTY_SPACE"
        return md5(sig.encode('utf-8')).hexdigest()
    except Exception:
        return None

def find_best_unicode(g_hash, g_name, f_name, db_map):
    if g_hash not in db_map: return None
    records = db_map[g_hash]
    space_hash = md5("EMPTY_SPACE".encode('utf-8')).hexdigest()
    if g_hash == space_hash: return "0020"

    clean_f_name = f_name.split('+')[-1] if f_name else ""
    for r in records:
        if r["GlyphName"] == g_name and r["font_name"] == clean_f_name: return r["unicode_hex"]
    for r in records:
        if r["GlyphName"] == g_name: return r["unicode_hex"]
    unique_hexes = set(r["unicode_hex"] for r in records)
    if len(unique_hexes) == 1: return list(unique_hexes)[0]
    return None

def get_differences(doc, xref):
    try:
        enc_obj = doc.xref_get_key(xref, "Encoding")
        raw_enc = doc.xref_object(xref) if enc_obj[0] == "dict" else doc.xref_object(int(enc_obj[1].split()[0]))
        diff_match = re.search(r'/Differences\s*\[(.*?)\]', raw_enc, re.DOTALL)
        if diff_match:
            tokens = re.findall(r'/[^\s\[\]/]+|\d+', diff_match.group(1))
            res, curr = {}, -1
            for t in tokens:
                if t.isdigit():
                    curr = int(t)
                elif t.startswith('/') and curr != -1:
                    res[curr] = t[1:]
                    curr += 1
            return res
    except:
        pass
    return {}

def generate_dummy_tounicode():
    cmap = [
        "/CIDInit /ProcSet findresource begin", "12 dict begin", "begincmap",
        "/CIDSystemInfo << /Registry (Adobe) /Ordering (UCS) /Supplement 0 >> def",
        "/CMapName /Adobe-Identity-UCS def", "/CMapType 2 def",
        "1 begincodespacerange", "<00> <FF>", "endcodespacerange"
    ]
    for start in range(0, 256, 100):
        chunk_size = min(100, 256 - start)
        cmap.append(f"{chunk_size} beginbfchar")
        for i in range(start, start + chunk_size):
            cmap.append(f"<{i:02X}> <E0{i:02X}>")
        cmap.append("endbfchar")
    cmap.extend(["endcmap", "CMapName currentdict /CMap defineresource pop", "end", "end"])
    return "\n".join(cmap)

def generate_real_tounicode(mapping_dict):
    cmap = [
        "/CIDInit /ProcSet findresource begin", "12 dict begin", "begincmap",
        "/CIDSystemInfo << /Registry (Adobe) /Ordering (UCS) /Supplement 0 >> def",
        "/CMapName /Adobe-Identity-UCS def", "/CMapType 2 def",
        "1 begincodespacerange", "<00> <FF>", "endcodespacerange"
    ]
    items = list(mapping_dict.items())
    chunks = [items[i:i + 100] for i in range(0, len(items), 100)]
    for chunk in chunks:
        cmap.append(f"{len(chunk)} beginbfchar")
        for cid, u_hex in chunk:
            cmap.append(f"<{cid:02X}> <{u_hex}>")
        cmap.append("endbfchar")
    cmap.extend(["endcmap", "CMapName currentdict /CMap defineresource pop", "end", "end"])
    return "\n".join(cmap)

def load_sequence(doc, flags, db_map, font_cache, rezim="vizual"):
    sekvence = {}
    for page in doc:
        fonts_on_page = {f[0]: f[3] for f in page.get_fonts()}
        raw_text = page.get_text("rawdict", flags=flags)

        for block in raw_text.get("blocks", []):
            if block.get("type") != 0: continue
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    f_name = span.get("font")
                    xref = next((x for x, n in fonts_on_page.items() if (n == f_name or f_name in n)), None)

                    if not xref: continue
                    if xref not in sekvence: sekvence[xref] = []

                    if rezim == "vizual" and xref not in font_cache:
                        _, ext, _, buffer = doc.extract_font(xref)
                        if ext == "cff":
                            try:
                                cff = CFFFontSet()
                                cff.decompile(BytesIO(buffer), None)
                                glyph_set = cff[0].CharStrings
                                pdf_diffs = get_differences(doc, xref)
                                try:
                                    internal_enc = {gid: n for gid, n in enumerate(cff.getGlyphOrder())}
                                except:
                                    internal_enc = {gid: gname for gid, gname in enumerate(glyph_set.keys())}
                                font_cache[xref] = {"glyph_set": glyph_set, "diffs": pdf_diffs, "enc": internal_enc, "name": f_name}
                            except:
                                font_cache[xref] = None
                        else:
                            font_cache[xref] = None

                    for char_obj in span.get("chars", []):
                        raw_char = char_obj.get("c", "")
                        if not raw_char: continue

                        if rezim == "vizual":
                            b = ord(raw_char) % 256
                            if xref in font_cache and font_cache[xref]:
                                f_data = font_cache[xref]
                                g_name = f_data["diffs"].get(b, f_data["enc"].get(b, ".notdef"))
                                g_hash = get_standalone_glyph_hash(f_data["glyph_set"], g_name)
                                u_hex = find_best_unicode(g_hash, g_name, f_data["name"], db_map)
                                sekvence[xref].append(u_hex if u_hex else "003F")
                        else:
                            val = ord(raw_char)
                            if 0xE000 <= val <= 0xE0FF:
                                sekvence[xref].append(val - 0xE000)
                            else:
                                sekvence[xref].append(val % 256)
    return sekvence


def calculate_file_hash(filepath):
    try:
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                md5().update(chunk)
        return md5().hexdigest()
    except Exception as e:
        print(f"Error while loading font {filepath}: {e}")
        return None


def load_file_hash_db(db_path):
    valid_hashes = set()
    if not os.path.exists(db_path):
        return valid_hashes

    with open(db_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('|')
            if parts and len(parts[0]) == 32:
                valid_hashes.add(parts[0].strip().lower())
    return valid_hashes


def headless_repair(pdf_path, glyph_db_map, verbose=False):
    def vprint(msg):
        if verbose:
            print(msg)

    try:
        doc = fitz.open(pdf_path)
        all_unique_xrefs = set()
        ready_xrefs = set()
        font_cache_local = {}

        vprint("  [DEBUG] Phase 0: Scanning for fonts in PDF...")
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            for f in page.get_fonts(full=True):
                xref = f[0]
                if xref in all_unique_xrefs: continue

                name, ext, _, buffer = doc.extract_font(xref)
                if ext and ext.lower() == "cff" and not has_tounicode(doc, xref):
                    all_unique_xrefs.add(xref)

                    cff = CFFFontSet()
                    cff.decompile(BytesIO(buffer), None)
                    glyph_set = cff.topDictIndex[0].CharStrings
                    valid_glyph_names = [g for g in glyph_set.keys() if g != '.notdef']

                    total_glyphs = len(valid_glyph_names)
                    mapped_count = 0

                    for g_name in valid_glyph_names:
                        if g_name in EXTENDED_AGL:
                            mapped_count += 1
                        else:
                            g_hash = get_standalone_glyph_hash(glyph_set, g_name)
                            u_hex = find_best_unicode(g_hash, g_name, name, glyph_db_map)
                            if u_hex: mapped_count += 1

                    font_cache_local[xref] = {
                        "name": name, "glyph_set": glyph_set,
                        "mapped": mapped_count, "total": total_glyphs
                    }

                    if total_glyphs > 0 and mapped_count >= total_glyphs:
                        ready_xrefs.add(xref)
                        vprint(f"    -> Ready: Font '{name}' - {mapped_count}/{total_glyphs} chars")
                    else:
                        vprint(f"    -> Nekompletní: Font '{name}' - {mapped_count}/{total_glyphs} chars")

        if not all_unique_xrefs:
            print("  [SKIP] Document does not contain repairabel CFF fonts")
            doc.close()
            return

        if not ready_xrefs:
            print(f"  [SKIP] Found {len(all_unique_xrefs)} CFF fonts, none are 100% mapped.")
            doc.close()
            return

        print(f"  [INFO] Repairing {len(ready_xrefs)} of {len(all_unique_xrefs)} CFF fonts...")

        custom_flags = fitz.TEXT_PRESERVE_LIGATURES | fitz.TEXT_INHIBIT_SPACES | fitz.TEXT_USE_CID_FOR_UNKNOWN_UNICODE | fitz.TEXT_PRESERVE_WHITESPACE

        vprint("  [DEBUG] Visual sequence loading...")
        vizualni_sekvence = load_sequence(doc, custom_flags, glyph_db_map, font_cache_local, rezim="vizual")
        vizualni_sekvence = {x: seq for x, seq in vizualni_sekvence.items() if x in ready_xrefs}

        vprint("  [DEBUG] Step 1/3: DUMMY ToUnicode injection")
        dummy_cmap_str = generate_dummy_tounicode()
        for xref in vizualni_sekvence.keys():
            dummy_xref = doc.get_new_xref()
            doc.update_object(dummy_xref, "<<>>")
            doc.update_stream(dummy_xref, dummy_cmap_str.encode("utf-8"))
            doc.xref_set_key(xref, "ToUnicode", f"{dummy_xref} 0 R")
            vprint(f"    -> Dummy xref {dummy_xref} for font {xref} created")

        dummy_pdf_bytes = doc.tobytes()
        doc.close()

        vprint("  [DEBUG] Step 2/3: Internal ID extraction")
        doc_dummy = fitz.open("pdf", dummy_pdf_bytes)
        interni_sekvence = load_sequence(doc_dummy, custom_flags, glyph_db_map, font_cache_local, rezim="dummy")
        doc_dummy.close()

        # Finální oprava
        vprint("  [DEBUG] Step 3/3: Final ToUnicode creation")
        doc_final = fitz.open(pdf_path)
        success_count = 0

        for xref, v_seq in vizualni_sekvence.items():
            i_seq = interni_sekvence.get(xref, [])
            if len(v_seq) == len(i_seq):
                mapping = {i_id: v_hex for v_hex, i_id in zip(v_seq, i_seq)}
                real_cmap = generate_real_tounicode(mapping)
                new_xref = doc_final.get_new_xref()
                doc_final.update_object(new_xref, "<<>>")
                doc_final.update_stream(new_xref, real_cmap.encode("utf-8"))
                doc_final.xref_set_key(xref, "ToUnicode", f"{new_xref} 0 R")
                success_count += 1
                vprint(
                    f"    -> [OK] Font {xref} repaired. Mapped {len(mapping)} unique chars out of {len(v_seq)}")
            else:
                vprint(
                    f"    -> [ERROR] Skipping font {xref}! Length mismatch (Visual: {len(v_seq)} | Internal: {len(i_seq)})")

        base, ext = os.path.splitext(pdf_path)
        if success_count == len(all_unique_xrefs):
            out_path = f"{base}_Repaired{ext}"
        else:
            out_path = f"{base}_Partially_Repaired{ext}"

        doc_final.save(out_path)
        doc_final.close()
        print(f"  [SUCCESS] Saved as {os.path.basename(out_path)} ({success_count}/{len(all_unique_xrefs)} fonts)")

    except Exception as e:
        print(f"  [ERROR] Failed in processing: {e}")


def run_cli_mode(args):
    print(f"=== CLI Glyph Repair ===")

    if not args.hash_db:
        print("[ERROR] Parameter -d (--hash-db) with path to database missing")
        #print("Run with: python main.py C:\\folder -m -d hashes.psv")
        sys.exit(1)

    valid_hashes = load_file_hash_db(args.hash_db)
    if not valid_hashes:
        print(f"[ERROR] PSV database '{args.hash_db}' missing or empty.")
        sys.exit(1)

    print(f"[INFO] Loaded {len(valid_hashes)} validation hashes.")

    target_files = []
    if args.multiple:
        if not os.path.isdir(args.target):
            print(f"[ERROR] Target '{args.target}' is not a folder (required for -m).")
            sys.exit(1)

        for root, dirs, files in os.walk(args.target):
            for file in files:
                if file.lower().endswith('.pdf'):
                    target_files.append(os.path.join(root, file))
            if not args.recursive:
                break
    else:
        if not os.path.isfile(args.target):
            print(f"[ERROR] File '{args.target}' not found")
            sys.exit(1)
        target_files.append(args.target)

    if not target_files:
        print("[INFO] No PDFs found.")
        sys.exit(0)

    glyph_db_map = load_db(FontWidget.PSV_PATH)
    print(f"[INFO] Mapping {len(target_files)} files...\n")

    for pdf_path in target_files:
        print(f"-> {os.path.basename(pdf_path)}")

        if valid_hashes:
            file_hash = calculate_file_hash(pdf_path)
            if file_hash not in valid_hashes:
                print(f"  [BLOCKED] File hash not in database.")
                continue

        headless_repair(pdf_path, glyph_db_map, verbose=args.verbose)


def apply_dark_theme(app):
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
            border-bottom: 1px solid #333;
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
            background-color: #3d3d3d;
        }
        QToolBar QToolButton:pressed {
            background-color: #3d7eff;
        }
    """)


def run_gui_mode():
    app = QApplication(sys.argv)
    apply_dark_theme(app)

    window = FontWidget()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PDF Glyph Repair Tool - GUI & CLI")

    parser.add_argument("target", nargs='?', help="Path for file/folder to repair")

    parser.add_argument("-m", "--multiple", action="store_true", help="Folder path, repairs multiple files")
    parser.add_argument("-r", "--recursive", action="store_true", help="Repair recursively (-m required)")
    parser.add_argument("-d", "--hash-db", dest="hash_db", help="Path to PSV file containing md5 hashes")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enables verbose output")

    args = parser.parse_args()

    if args.target:
        run_cli_mode(args)

    else:
        run_gui_mode()