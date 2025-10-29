import sys
import webbrowser

import fitz
import matplotlib.patches as patches
from PySide6 import QtCore
from fontTools.cffLib import CFFFontSet
from fontTools.pens.basePen import BasePen
from PySide6.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, QListWidget, QListWidgetItem
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.path import Path
from io import BytesIO

# ======= Extract fonts from PDF =======
def extract_cff_fonts(pdf_path, page, font_name):
    doc = fitz.open(pdf_path)
    page = doc.load_page(page)
    fonts = page.get_fonts(full=True)
    for font in fonts:
        name, ext, type, buffer = doc.extract_font(font[0])
        if ext.lower() == "cff" and name == font_name:
            doc.close()
            return buffer
    doc.close()
    raise ValueError(f"Font '{font_name}' is not in PDF or isn't CFF.")

# ======= Matplotlib Pen =======
class MatplotlibPen(BasePen):
    def __init__(self, glyphSet):
        super().__init__(glyphSet)
        self.vertices = []
        self.codes = []

    def _moveTo(self, p):
        self.vertices.append(p)
        self.codes.append(Path.MOVETO)

    def _lineTo(self, p):
        self.vertices.append(p)
        self.codes.append(Path.LINETO)

    def _curveToOne(self, p1, p2, p3):
        self.vertices.extend([p1, p2, p3])
        self.codes.extend([Path.CURVE4, Path.CURVE4, Path.CURVE4])

    def _closePath(self):
        if self.vertices:
            self.vertices.append(self.vertices[-1])
            self.codes.append(Path.CLOSEPOLY)

# ======= Canvas Widget for Glyph =======
class GlyphCanvas(FigureCanvas):
    def __init__(self, font, parent=None):
        self.font = font
        self.fig = Figure(figsize=(4, 4))
        super().__init__(self.fig)
        self.ax = self.fig.add_subplot(111)
        self.ax.axis('off')
        self.glyphSet = None
        self.glyph_name = None
        self.max_glyph_height = 1.0

    def draw_glyph(self, glyphSet, glyph_name, max_glyph_height):
        self.ax.clear()
        self.ax.axis('off')
        self.glyphSet = glyphSet
        self.glyph_name = glyph_name
        self.max_glyph_height = max_glyph_height

        glyph = glyphSet[glyph_name]
        pen = MatplotlibPen(glyphSet)
        glyph.draw(pen)

        if pen.vertices:
            xs, ys = zip(*pen.vertices)
            min_x, max_x = min(xs), max(xs)
            width = max_x - min_x

            ascent = getattr(self.font, 'FontBBox', [0, 0, 0, 1000])[3]
            descent = getattr(self.font, 'FontBBox', [0, -200, 0, 0])[1]
            font_height = ascent - descent

            scale = 0.8 / font_height

            top_margin = 0.05 * font_height * scale
            bottom_margin = 0.05 * font_height * scale

            vertices = [((x - min_x - width / 2) * scale,
                         (y - descent) * scale + bottom_margin)
                        for x, y in pen.vertices]

            path = Path(vertices, pen.codes)
            patch = patches.PathPatch(path, facecolor='black', lw=1)
            self.ax.add_patch(patch)

            self.ax.set_xlim(-0.5, 0.5)
            self.ax.set_ylim(0, font_height * scale + top_margin + bottom_margin)

        self.ax.set_aspect('equal')
        self.ax.autoscale_view()
        self.draw()

# ======= Main PySide Widget =======
class FontWidget(QWidget):
    def __init__(self, font_data):
        super().__init__()
        self.font = CFFFontSet()
        self.font.decompile(BytesIO(font_data), None)
        self.topDict = self.font.topDictIndex[0]
        self.glyphSet = self.topDict.CharStrings
        self.glyph_names = list(self.glyphSet.keys())
        self.current_index = 0
        self.glyph_to_char = {}

        # ===== Glyph list =====
        self.glyph_list = QListWidget()
        for name in self.glyph_names:
            self.glyph_list.addItem(QListWidgetItem(name))
        self.glyph_list.itemClicked.connect(self.on_glyph_clicked)

        # ===== Right side =====
        self.canvas = GlyphCanvas(self.topDict)
        self.label = QLabel("")
        self.label.setStyleSheet("font-weight:bold; font-size:40px;")
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.user_input = QLineEdit()
        self.user_input.setMaxLength(1)
        self.user_input.returnPressed.connect(self.save_glyph)
        self.btn_special = QPushButton("Special chars")
        self.btn_special.clicked.connect(self.open_special)
        self.btn_glyph = QPushButton("Save glyph")
        self.btn_glyph.clicked.connect(self.save_glyph)
        self.btn_font = QPushButton("Update font")
        self.btn_font.clicked.connect(self.submit_ToUnicode)

        # ===== Layout builder =====
        input_layout = QHBoxLayout()
        input_layout.addWidget(self.user_input)
        input_layout.addWidget(self.btn_special)
        input_layout.addWidget(self.btn_glyph)
        input_layout.addWidget(self.btn_font)

        right_layout = QVBoxLayout()
        right_layout.addWidget(self.canvas)
        right_layout.addWidget(self.label)
        right_layout.addLayout(input_layout)

        main_layout = QHBoxLayout()
        main_layout.addWidget(self.glyph_list, 2)
        main_layout.addLayout(right_layout, 5)
        self.setLayout(main_layout)

        self.setMinimumSize(800, 600)
        self.setWindowTitle("Glyph Repair")

        # ===== Glyph max height calculations =====
        self.max_glyph_height = 1.2
        for name in self.glyph_names:
            glyph = self.glyphSet[name]
            pen = MatplotlibPen(self.glyphSet)
            glyph.draw(pen)
            if pen.vertices:
                ys = [y for x, y in pen.vertices]
                height = max(ys) - min(ys)
                if height > self.max_glyph_height:
                    self.max_glyph_height = height

        self.show_glyph()

    # ======= Render current glyph =====
    def show_glyph(self):
        glyph_name = self.glyph_names[self.current_index]
        self.canvas.draw_glyph(self.glyphSet, glyph_name, self.max_glyph_height)
        display_name = glyph_name
        if glyph_name in self.glyph_to_char:
            glyph_uni = chr(int(self.glyph_to_char[glyph_name], 16))
            display_name += f" [{glyph_uni}]"
        self.label.setText(display_name)

    # ======= Skip to next glyph =====
    def show_next(self):
        self.current_index = (self.current_index + 1) % len(self.glyph_names)
        self.show_glyph()

    # ======= Navigate glyph list =====
    def on_glyph_clicked(self, item):
        glyph_name = item.text().split()[0]
        if glyph_name in self.glyph_names:
            self.current_index = self.glyph_names.index(glyph_name)
            self.show_glyph()

    #TODO ======= Temporary special character solution =====
    def open_special(self):
        webbrowser.open_new_tab("https://www.vertex42.com/ExcelTips/unicode-symbols.html")

    # ======= Save current glyph =====
    def save_glyph(self):
        char = self.user_input.text()
        if not char:
            return
        glyph_name = self.glyph_names[self.current_index]
        self.glyph_to_char[glyph_name] = format(ord(char), '04x')
        item = self.glyph_list.item(self.current_index)
        item.setText(f"{glyph_name} [{char}]")
        self.user_input.clear()
        self.show_next()

    # ======= Submit repaired font =====
    def submit_ToUnicode(self):
        print(self.glyph_to_char)
        sys.exit(0)

# ======= Run =======
if __name__ == "__main__":
    app = QApplication()
    pdf_path = "AR sample/Sample.pdf"
    font_name = "GKCMAE+Arial068.313"
    try:
        font_data = extract_cff_fonts(pdf_path, 0, font_name)
        window = FontWidget(font_data)
        window.show()
        sys.exit(app.exec())
    except ValueError as e:
        print(f"Chyba: {e}")
        sys.exit(1)