import csv
from hashlib import md5
import os
import sys
import webbrowser
from numpy import asarray
from io import BytesIO

import fitz
import matplotlib.patches as patches
from PySide6 import QtCore, QtWidgets, QtGui
from PySide6.QtGui import QImage, QPixmap, QIcon
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QListWidget, QListWidgetItem, QMainWindow, QFileDialog, QToolButton
)
from fontTools.agl import UV2AGL
from fontTools.cffLib import CFFFontSet
from fontTools.pens.basePen import BasePen
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.path import Path


# SECTION Extraction from pdf
# Function extracts raw font data from pdf file
def extract_cff_fonts(pdf_path, page, font_name):
    # Open file using PyMuPDF
    with fitz.open(pdf_path) as doc:
        page_obj = doc.load_page(page)  # Load page as object
        fonts = page_obj.get_fonts(full=True)  # Extract fonts from page
        # Loop through every font extracted
        for font in fonts:
            name, ext, _, buffer = doc.extract_font(font[0])  # Extract name, type, and raw binary data of the font
            # Filter only fonts saved in cff format with the desired name
            if ext and ext.lower() == "cff" and name == font_name:
                # Return extracted data
                return buffer
        # Raise error if no font with the given name in CFF format was found
        raise ValueError(f"Font '{font_name}' not found or not CFF.")


# Function extracts font data and matches pages and fonts used
def extract_pdf_data(pdf_path):
    font_map = {}  # Initialize a dictionary for page number and fonts used
    # Open file using PyMuPDF
    with fitz.open(pdf_path) as doc:
        # Loop through pages in documents
        for page in doc:
            cff_names = []  # Initialize list of font names in each page
            fonts = page.get_fonts(full=True)  # Extract fonts from page
            # Loop through every font
            for font in fonts:
                name, ext, _, buffer = doc.extract_font(font[0])  # Extract name, font format, and raw binary data of the font
                # Add only fonts in cff format to cff_name list
                if ext and ext.lower() == "cff":
                    cff_names.append(name)
            # If there are any names in list, match them to page number
            if cff_names:
                font_map[page.number] = cff_names
        return font_map  # Return complete map of pages and fonts used


# SECTION MatplotlibPen
# Custom pen that inherits from fontTools' BasePen to record outlines as Matplotlib Path data
class MatplotlibPen(BasePen):
    # Initializing variables to store vertices and Matplotlib path codes
    def __init__(self, glyphset):
        super().__init__(glyphset)
        self.vertices = []
        self.codes = []

    # Defining a moveTo method which adds a point to vertices list and path code MOVETO to codes list
    def _moveTo(self, p):
        self.vertices.append(p)
        self.codes.append(Path.MOVETO)

    # Defining a lineTo method which adds a point to vertices list and path code LINETO to codes list
    def _lineTo(self, p):
        self.vertices.append(p)
        self.codes.append(Path.LINETO)

    # Defining a curveToOne method which adds list of 3 points to vertices list and list of 3 CURVE4 path code to codes list
    def _curveToOne(self, p1, p2, p3):
        self.vertices.extend([p1, p2, p3])
        self.codes.extend([Path.CURVE4, Path.CURVE4, Path.CURVE4])

    # Defining a closePath method
    def _closePath(self):
        # Check if there are any points in vertices list
        if self.vertices:
            self.vertices.append(self.vertices[0])  # Add first point to vertices list
            self.codes.append(Path.CLOSEPOLY)  # Add path code CLOSEPOLY to codes list


# SECTION Matplotlib canvas
# Custom canvas that inherits from matplotlib's FigureCanvas to display individual glyphs
class GlyphCanvas(FigureCanvas):
    # Initializing variables for use in other methods
    def __init__(self, font):
        self.font = font  # Store the font object for later use
        self.fig = Figure(figsize=(4, 4))  # Defining size of figure and creating it
        super().__init__(self.fig)  # Initializing FigureCanvas using created figure
        self.ax = self.fig.add_subplot()  # Create an Axes object used for drawing glyphs

    # Method used to draw glyph from a set of glyphs defined by its name,
    def draw_glyph(self, glyphset, glyph_name, notdef_max_y, notdef_min_y):
        ax = self.ax  # Using initialized Axes object
        ax.clear()  # Clear previously drawn glyphs
        ax.axis('off')  # Hide axes

        # Check whether the glyph set exists and contains the requested glyph
        if not glyphset or glyph_name not in glyphset:
            # Add text to canvas describing that there is no glyph to draw
            ax.text(0.5, 0.5, "No glyph", ha='center', va='center', fontsize=48, color='dimgray', weight='bold',
                    style='italic')
            self.draw()  # Render text
            return  # Exit the method early

        glyph = glyphset[glyph_name]  # Retrieve the glyph object corresponding to the given name
        pen = MatplotlibPen(glyphset)  # Create a MatplotlibPen to record the glyph's outlines
        glyph.draw(pen)  # Use the pen to draw the glyph's outline data into vertices and path codes

        # Check, if the glyph is empty
        if not pen.vertices:
            # Add text to canvas describing that there is empty glyph
            ax.text(0.5, 0.5, "Empty glyph,\nmost likely space", ha='center', va='center', fontsize=48, color='dimgray',
                    weight='bold', style='italic')
            self.draw()  # Render text
            return  # Exit the method early

        xs, ys = zip(*pen.vertices)  # Separate vertices into x and y coordinate arrays
        min_x, max_x = min(xs), max(xs)  # Get minimum and maximum coordinates on x-axis
        width = max_x - min_x  # Calculate glyph width

        ascent = getattr(self.font, 'FontBBox', [0, 0, 0, 1000])[
            3]  # Get top y-coordinate from FontBoundingBox; fallback to 1000 if missing
        descent = getattr(self.font, 'FontBBox', [0, -200, 0, 0])[
            1]  # Get bottom y-coordinate from FontBBox; fallback to -200 if missing
        font_height = ascent - descent  # Calculate maximum height of all glyphs in font
        scale = 0.8 / font_height  # Scale glyph to 80% of the canvas height
        bottom_margin = 0.05 * font_height * scale  # Add bottom margin

        # Check, if there are baseline values given to method
        if notdef_min_y is not None and notdef_max_y is not None:
            # Convert .notdef baseline coordinates into scaled canvas positions
            min_y = (notdef_min_y - descent) * scale + bottom_margin
            max_y = (notdef_max_y - descent) * scale + bottom_margin
            # Draw horizontal reference lines based on the .notdef glyph
            ax.axhline(y=min_y, color='blue', linestyle=':', linewidth=1.5)
            ax.axhline(y=max_y, color='blue', linestyle=':', linewidth=1.5)
        else:
            # Draw a fallback baseline at the bottom margin
            ax.axhline(y=bottom_margin, color='red', linestyle=':', linewidth=1.5)

        vertices = []  # New list of transformed vertices
        for x, y in pen.vertices:
            x_transformed = (x - min_x - width / 2) * scale  # Center the glyph horizontally and scale
            y_transformed = (y - descent) * scale + bottom_margin  # Shift vertically and scale
            vertices.append((x_transformed, y_transformed))  # Add transformed point to the new list

        path = Path(vertices, pen.codes)  # Use new vertices list and codes to define a path object
        patch = patches.PathPatch(path, facecolor='black', lw=1)  # Create a glyph image
        ax.add_patch(patch)  # Add glyph to canvas

        ax.set_xlim(-0.5, 0.5)  # Limit horizontal drawing range
        ax.set_ylim(0, font_height * scale * 1.1)  # Limit vertical drawing range
        ax.set_aspect('equal')  # Keep equal scaling for x and y axes to preserve glyph proportions
        self.draw()  # Render glyph


# SECTION Main window
# Custom window that inherits from PySide6 QMainWindow
class FontWidget(QMainWindow):
    ICON_SIZE = 64  # Size of icons in glyph list in pixels
    CSV_PATH = "glyph_mappings.csv"

    # Initializing variables for use in other methods
    def __init__(self):
        super().__init__()  # Initializing QMainWindow
        self.pdf_path = None  # Path to the currently loaded PDF file
        self.current_page = None  # Currently selected page number
        self.current_font_name = None  # Currently selected font name
        self.current_font = None  # Current font object
        self.current_glyph_set = None  # Dictionary of glyph objects (topDict)
        self.current_font_glyph_names = []  # List of glyph names (CharStrings)
        self.current_index = 0  # Index of currently visible glyph
        self.user_glyph_to_char = {}  # Dictionary of glyph names and character information
        self.font_cache = {}  # Cached font data
        self.db_cache = {}  # Cached glyph-to-character mappings from database

        self._setup_menus()  # Initialize application menus
        self._setup_ui()  # Set up main UI layout and widgets

        self.clear_ui_state()  # Reset UI
        self.setMinimumSize(1000, 700)  # Set minimum window size
        self.setWindowTitle("Glyph Repair")  # Set application window title
        self.statusBar().showMessage("Select PDF to repair")  # Add message to status bar

    # Initialize menu bar
    def _setup_menus(self):
        menubar = self.menuBar()  # Create menuBar object
        pdf_menu = menubar.addMenu("PDF")  # Add pdf menu

        open_action = pdf_menu.addAction("Open PDF")  # Add action Open PDF to pdf menu
        open_action.setIcon(QIcon.fromTheme("folder-open"))  # Set icon for action
        open_action.triggered.connect(self.open_pdf)  # Connect the action to open_pdf method

        self.export_action = pdf_menu.addAction("Save PDF")  # Add action Save PDF to pdf menu
        self.export_action.setIcon(QIcon.fromTheme("document-save"))  # Set icon for action
        self.export_action.setEnabled(False)  # Disable action initially
        self.export_action.triggered.connect(self.save_pdf)  # Connect the action to save_pdf method

        exit_action = pdf_menu.addAction("Exit")  # Add action Exit to pdf menu
        exit_action.setIcon(QIcon.fromTheme("window-close"))  # Set icon for action
        exit_action.triggered.connect(self.close)  # Close window

        self.pages_menu = menubar.addMenu("Pages")  # Add pages menu
        self.fonts_menu = menubar.addMenu("Fonts")  # Add fonts menu
        self._menu_placeholder(self.pages_menu)  # Add placeholder item for Pages menu
        self._menu_placeholder(self.fonts_menu)  # Add placeholder item for Pages menu

    # Create placeholder actions
    def _menu_placeholder(self, menu):
        placeholder = menu.addAction("No file loaded")  # Add text as action
        placeholder.setIcon(QIcon.fromTheme("sync-error"))  # Set icon for action
        placeholder.setEnabled(False)  # Disable action

    # Initialize UI elements
    def _setup_ui(self):
        central = QWidget()  # Create QWidget
        self.setCentralWidget(central)  # Add widget as Central Widget

        glyph_list = self.glyph_list = QListWidget()  # Glyph list widget
        canvas = self.canvas = GlyphCanvas(None)  # Canvas for displaying glyphs
        label = self.label = QLabel("Select glyph")  # Label showing glyph information
        user_input = self.user_input = QLineEdit()  # Character input field
        btn_special = self.btn_special = QPushButton("Special chars")  # Button for special characters
        btn_glyph = self.btn_glyph = QPushButton("Save glyph")  # Button for saving individual glyphs
        btn_font = self.btn_font = QPushButton("Update font")  # Button for updating whole font
        btn_prev = self.btn_prev_font = QToolButton()  # Button for switching to previous font
        btn_next = self.btn_next_font = QToolButton()  # Button for switching to next font
        lbl_font = self.lbl_font = QLabel("No font loaded")  # Label showing current font name

        # Glyph list
        glyph_list.setIconSize(QtCore.QSize(self.ICON_SIZE, self.ICON_SIZE))  # Set icon size of menu items
        glyph_list.setSpacing(0)  # Remove spacing between list items
        font = glyph_list.font()  # Use default font
        font.setPointSize(20)  # Set font size
        font.setBold(True)  # Make text bold
        glyph_list.setFont(font)  # Apply font to list

        # Information label
        label.setStyleSheet("font-weight: bold; font-size: 48px; color: white;")  # Label weight, size and color
        label.setAlignment(QtCore.Qt.AlignCenter)  # Center label text

        # Character input field
        user_input.setPlaceholderText("Enter glyph")  # Placeholder text
        user_input.setMaxLength(1)  # Limit to one character
        user_input.returnPressed.connect(self.save_glyph)  # Enter triggers saving the glyph

        # Buttons
        btn_special.clicked.connect(self.open_special)  # Connect character button to open_special
        btn_glyph.clicked.connect(self.save_glyph)  # Connect save glyph button to save_glyph
        btn_font.clicked.connect(self.submit_ToUnicode)  # Connect update font button to submit_ToUnicode
        btn_prev.clicked.connect(self.go_to_prev_font)  # Connect previous font button to go_to_prev_font
        btn_next.clicked.connect(self.go_to_next_font)  # Connect next font button to go_to_next_font

        # Previous and next buttons
        btn_prev.setArrowType(QtCore.Qt.LeftArrow)  # Left arrow
        btn_next.setArrowType(QtCore.Qt.RightArrow)  # Right arrow
        btn_prev.setFixedSize(40, 40)  # Button size
        btn_next.setFixedSize(40, 40)  # Button size
        btn_prev.setToolTip("Previous font")  # Set tooltip
        btn_next.setToolTip("Next font")  # Set tooltip

        # Current font label
        font = lbl_font.font()  # Use default font
        font.setPointSize(14)  # Size of font in list
        font.setBold(True)  # Text weight
        lbl_font.setFont(font)  # Use defined font information
        lbl_font.setAlignment(QtCore.Qt.AlignCenter)  # Align label to center
        lbl_font.setMinimumWidth(180)  # Set minimum width of label

        # Build layout
        # Font navigation
        nav = QHBoxLayout()  # Horizontal layout
        nav.addWidget(btn_prev)  # Add previous font button
        nav.addWidget(lbl_font)  # Add font name label
        nav.addWidget(btn_next)  # Add next font button
        nav.setSpacing(6)  # Set spacing between elements

        # User input and save buttons
        inputs = QHBoxLayout()  # Horizontal layout
        inputs.addWidget(user_input)  # Add user input field
        inputs.addWidget(btn_special)  # Add special characters button
        inputs.addWidget(btn_glyph)  # Add save individual glyph button
        inputs.addWidget(btn_font)  # Add update font button

        # Right side of layout
        right = QVBoxLayout()  # Vertical layout
        right.addLayout(nav)  # Add font navigation layout
        right.addWidget(canvas)  # Add glyph canvas
        right.addWidget(label)  # Add font info label
        right.addLayout(inputs)  # Add User input layout

        # Main window layout
        main = QHBoxLayout(central)  # Horizontal layout
        main.addWidget(glyph_list, 2)  # Add glyph list as 2/7 of window width
        main.addLayout(right, 5)  # Add right side layout as 5/7 of window width

    # Clear UI
    def clear_ui_state(self):
        self.glyph_list.clear()  # Remove all items from glyph list
        self.label.setText("No font loaded")  # Label text
        self.canvas.draw_glyph(None, None, None, None)  # Clear glyph canvas
        self.user_input.clear()  # Clear user input field
        self.user_input.setEnabled(False)  # Disable user input field
        self.btn_glyph.setEnabled(False)  # Diable save glyph button
        self.btn_font.setEnabled(False)  # Diable update font button
        self.lbl_font.setText("No font loaded")  # Current font text

    # Navigate through fonts
    def go_to_prev_font(self):
        self._navigate_font(-1)  # Move to previous font

    def go_to_next_font(self):
        self._navigate_font(1)  # Move to next font

    # Cycle through fonts
    def _navigate_font(self, step):
        # Skip if there is no pdf selected
        if not self.pdf_path:
            return

        fontList = []  # New list of all fonts, for indexing
        # For each action (font name) in fonts menu
        for font in self.fonts_menu.actions():
            # If action is enabled (check for placeholders)
            if font.isEnabled():
                fontList.append(font)  # Add action (font name) tp actions

        # If there are no fonts, exit
        if not fontList:
            return

        cur_page = self.current_page  # Currently showing page index
        cur_name = self.current_font_name  # Currently showing font name

        idx = 0  # Placeholder index
        # Loop through each font and find id of currently shown font in fontList
        for i, font in enumerate(fontList):
            if font.data() == (cur_page, cur_name):
                idx = i
                break

        next_idx = (idx + step) % len(fontList)  # Calculate the next index, wrapping around if necessary

        # Get page and font name for the next font and load it
        page, name = fontList[next_idx].data()
        self.load_font(page, name)

    # Helper method to move to the next glyph
    def show_next(self):
        # Only proceed if there are glyphs loaded
        if self.current_font_glyph_names:
            self.current_index = (self.current_index + 1) % len(
                self.current_font_glyph_names)  # Increment current glyph index, wrap around to 0 if at the end
            self.show_glyph()  # Display the glyph at the new index

    # Helper method to select a glyph when clicked in the glyph list
    def on_glyph_clicked(self, item):
        name = item.data(QtCore.Qt.UserRole)  # Retrieve glyph name from built-in UserRole variable
        # Check if name is in list of names
        if name in self.current_font_glyph_names:
            self.current_index = self.current_font_glyph_names.index(
                name)  # Update the current glyph index to match the clicked glyph
            self.show_glyph()  # Display the newly selected glyph

    # Save current glyph to user_glyph_to_char dict
    def save_glyph(self):
        char = self.user_input.text().strip() or " "  # Get char from user input field, or set to space if empty

        glyph_name = self.current_font_glyph_names[self.current_index]  # Get the name of the currently displayed glyph
        unicode_hex = format(ord(char),
                             '04x')  # Convert the user-entered character to a 4-digit hexadecimal Unicode string
        agn = UV2AGL.get(ord(char),
                         "")  # Look up the Adobe Glyph List name for this character, or return empty string if not found

        # Save data to user_glyph_to_char
        self.user_glyph_to_char[glyph_name] = {
            "font_hash": self.get_font_hash(),  # Save a font hash
            "font_name": self.current_font_name,  # Save font name
            "unicode_hex": unicode_hex,  # Save hexadecimal unicode representation
            "AGN": agn  # Save adobe glyph list name
        }

        item = self.glyph_list.item(self.current_index)  # Item object of currently selected glyph in list
        display = "[space]" if char == " " else char  # Set message to user-defined character
        item.setText(f" → {display}")  # Set text of item in list
        item.setForeground(QtGui.QColor("#228B22"))  # Set color of item in list

        self.user_input.clear()  # Clear user input field
        self.show_next()  # Skip to next glyph

    # Method for opening website with special characters for now TODO
    def open_special(self):
        webbrowser.open_new_tab("https://www.vertex42.com/ExcelTips/unicode-symbols.html")

    # Get hast from font data
    def get_font_hash(self):
        # Check if there are any font_data to hash
        if not hasattr(self, 'font_data') or not self.font_data:
            return "unknown"
        return md5(self.font_data).hexdigest()  # User MD5 algorithm

    # Method to save mappings to database
    def submit_ToUnicode(self):
        self.save_to_db()  # Save to database
        total = len(self.current_font_glyph_names)  # Get statistic of all glyphs in font
        # Get statistic of all glyphs mapped
        mapped = 0
        for g in self.current_font_glyph_names:
            if g in self.user_glyph_to_char:
                mapped += 1
        self.statusBar().showMessage(f"Saved: {mapped}/{total} glyphs", 3000)  # Show statistics in status bar
        self.go_to_next_font()  # Skip to next font

    # Method for loading fonts
    def load_font(self, page, font_name):
        self.current_page = page  # Set current page
        self.current_font_name = font_name  # Set current font name
        self.lbl_font.setText(font_name)  # Set current font label

        cache = self.font_cache.get((page, font_name))  # Load cache data for page and font name
        # Check if there is anything in cache, or there is any unknow hash
        if not cache or cache['hash'] == "unknown":
            QtWidgets.QMessageBox.critical(self, "Error", "Font cant be loaded from cache.")  # Show error box
            return  # Exit the method early

        try:
            font_data = cache.get('data') or extract_cff_fonts(self.pdf_path, page,
                                                               font_name)  # Get data from cache or extract from pdf
            # Fill variables
            self.font_data = font_data  # Set font data
            self.reload_font(font_data)  # Reload font using font data

            self.user_glyph_to_char = {}  # Initialize user_glyph_to_char dict
            self.load_mappings_for_current_font()  # Load mapping
            self.populate_glyph_list()  # Populate glyph list
            self.show_glyph()  # Render first glyph

            self.user_input.setEnabled(True)  # Enable user input field
            self.btn_glyph.setEnabled(True)  # Enable save glyph button
            self.btn_font.setEnabled(True)  # Enable update font button
            self.statusBar().showMessage(f"Loaded: {font_name} (Page {page + 1})",
                                         5000)  # Show information on status bar
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Error while loading font:\n{e}")  # Error box

    # Method for reloading font
    def reload_font(self, font_data):
        font = CFFFontSet()  # Initialize library
        font.decompile(BytesIO(font_data), None)  # Add data to font variable
        topDict = font.topDictIndex[0]  # Get topDict information
        glyphSet = topDict.CharStrings  # Get glyphset using charstrings table
        glyph_names = list(glyphSet.keys())  # Get list of names from glyphset

        notdef_baseline = notdef_topline = None  # Initialize notdef vertical size variables
        # Check if .notdef name is in glyphset
        if '.notdef' in glyphSet:
            pen = MatplotlibPen(glyphSet)  # Create a MatplotlibPen
            glyphSet['.notdef'].draw(pen)  # Use pen to get information about .notdef glyph
            # If .notdef is not empty character
            if pen.vertices:
                _, ys = zip(*pen.vertices)  # Add all points y values to list
                notdef_baseline = min(ys)  # Determine minimum y value as baseline
                notdef_topline = max(ys)  # Determine maximum y value as reference point

        glyph_names = [name for name in glyph_names if
                       name != '.notdef']  # Do not add .notdef to font_glyph_names to be rendered

        # Fill variables
        self.current_font = topDict
        self.current_glyph_set = glyphSet
        self.current_font_glyph_names = glyph_names
        self.canvas.font = topDict
        self.notdef_baseline = notdef_baseline
        self.notdef_topline = notdef_topline
        self.current_index = 0

    # Generate status icon used for menus in given color
    def create_status_icon(self, color):
        size = 12  # Size of the icon
        pix = QPixmap(size, size)  # Creation of pixmap
        pix.fill(QtCore.Qt.transparent)  # Transparent background
        p = QtGui.QPainter(pix)  # Using pixmap as a base for painter
        p.setRenderHint(QtGui.QPainter.Antialiasing)  # Enable antialiasing
        p.setBrush(QtGui.QBrush(QtGui.QColor(color)))  # Add brush in desired color
        p.drawEllipse(2, 2, size - 4, size - 4)  # Draw a circle icon
        p.end()  # End painter
        return QIcon(pix)  # Return pixmap as icon

    # Generate glyphs as icons for menu list
    def generate_glyph_pixmap(self, glyph_name, size=(64, 64)):
        fig = Figure(figsize=(1.0, 1.0), dpi=200)  # Defining size of figure and creating it
        ax = fig.add_subplot()  # Create an Axes object used for drawing glyphs
        ax.axis('off')  # Hide axes

        glyph = self.current_glyph_set[glyph_name]  # Get desired glyph as object
        pen = MatplotlibPen(self.current_glyph_set)  # Create a MatplotlibPen to record the glyph's outlines
        glyph.draw(pen)  # Convert data into vertices and codes

        # If the glyph isn't empty
        if pen.vertices:
            xs, ys = zip(*pen.vertices)  # Separate vertices into x and y coordinate lists
            min_x, max_x = min(xs), max(xs)  # Get minimum and maximum coordinates on x-axis
            min_y, max_y = min(ys), max(ys)  # Get minimum and maximum coordinates on y-axis
            width = max_x - min_x  # Calculate glyph width
            height = max_y - min_y  # Calculate glyph height
            scale = min(0.8 / max(width, height, 1), 0.8)  # Calculate scale of object, fall back to 0.8 if greater

            vertices = []  # New list of transformed vertices
            for x, y in pen.vertices:
                x_transformed = (x - min_x - width / 2) * scale  # Center the glyph horizontally and scale
                y_transformed = (y - min_y - height / 2) * scale  # Center the glyph vertically and scale
                vertices.append((x_transformed, y_transformed))  # Add transformed point to the new list
            path = Path(vertices, pen.codes)  # Use new vertices list and codes to define a path object
            patch = patches.PathPatch(path, facecolor='black', lw=0.5)  # Create a glyph image
            ax.add_patch(patch)  # Add glyph on the axes
            ax.set_xlim(-1.0, 1.0)  # Limit horizontal drawing range
            ax.set_ylim(-1.0, 1.0)  # Limit vertical drawing range
            ax.set_aspect('equal')  # Keep equal scaling for x and y axes to preserve glyph proportions

        canvas = FigureCanvasAgg(fig)  # Create canvas on figure
        canvas.draw()  # Render glyph
        buf = canvas.buffer_rgba()  # Saving image as data
        arr = asarray(buf)  # Saves buffer data as array
        img = QImage(arr.data, arr.shape[1], arr.shape[0], QImage.Format_RGBA8888)  # Use array data to create image
        pix = QPixmap.fromImage(img)  # Create pixmap from image
        return pix.scaled(*size, QtCore.Qt.KeepAspectRatio,
                          QtCore.Qt.SmoothTransformation)  # Return scaled down version of pixmap

    # Fill glyph list with glyph icons
    def populate_glyph_list(self):
        w = self.glyph_list  # Reference to the QListWidget
        w.clear()  # Clear list
        # For each name in list, add a glyph icon
        for name in self.current_font_glyph_names:
            pix = self.generate_glyph_pixmap(name, size=(self.ICON_SIZE, self.ICON_SIZE))  # Generating glyph icon
            item = QListWidgetItem(QIcon(pix), "")  # Creating each item with icon and empty label
            item.setData(QtCore.Qt.UserRole, name)  # Adding glyph name to internal variable
            item.setSizeHint(QtCore.QSize(0, self.ICON_SIZE + 4))  # Sizing the item

            # If the glyph is mapped
            if name in self.user_glyph_to_char:
                ch = chr(int(self.user_glyph_to_char[name]["unicode_hex"], 16))  # Decode unicode character
                disp = "[space]" if ch == " " else ch  # Display [space] if the character is empty
                item.setText(f" → {disp}")  # Set text
                item.setForeground(QtGui.QColor("#228B22"))  # Set color to green
            else:
                item.setText(f" {name}")  # Add only glyph name
                item.setForeground(QtGui.QColor("#888888"))  # Set color to gray

            w.addItem(item)  # Add item to list
        w.itemClicked.connect(self.on_glyph_clicked)  # Connect clicking to on_glyph_clicked

    # Glyph canvas and label handler
    def show_glyph(self):
        name = self.current_font_glyph_names[self.current_index]  # Current glyph name
        self.canvas.draw_glyph(self.current_glyph_set, name, self.notdef_topline,
                               self.notdef_baseline)  # Render desired glyph

        mapping = self.user_glyph_to_char.get(name, {})  # Get desired mapping
        uhex = mapping.get("unicode_hex", "None")  # Retrieve hex code
        agn = mapping.get("AGN", "None")  # Retrieve Adobe Glyph List name
        ch = chr(int(uhex, 16)) if uhex != "None" else "None"  # Convert hex to character, unless missing

        if ch == " ": ch = "[space]"  # Set to [space] if empty

        # Build label text
        lines = [f"<b>Gcode:</b> {name}", f"<b>Znak:</b> {ch}",
                 f"<b>Unicode:</b> {uhex}", f"<b>AGL:</b> {agn}"]
        self.label.setText("<br>".join(lines))  # Set label text

        item = self.glyph_list.item(self.current_index)  # Retrieve current item
        # If it exists
        if item:
            self.glyph_list.setCurrentItem(item)  # Set current item as selected in list
            self.glyph_list.scrollToItem(item, QListWidget.EnsureVisible)  # Always visible glyph in list

    # Build pages menu
    def build_pages_menu(self, menu_data):
        menu = self.pages_menu  # QMenu variable
        menu.clear()  # Reset menu
        # Check if there are any data
        if not menu_data:
            a = menu.addAction("No CFF font")  # Add action as text
            a.setEnabled(False)  # Disable action
            return  # Exit method early

        # Loop through pages
        for page_num in sorted(menu_data.keys()):
            font_names = menu_data[page_num]  # Get names of all fonts on page
            if not font_names:  # Skip if no fonts
                continue

            page_mapped = page_total = 0  # Initialize counters for page statistics
            # Calculate page statistics
            for name in font_names:
                info = self.font_cache.get((page_num, name), {})  # Get font info from cache
                total = info.get('glyph_count', 0)  # Count all fonts for statistics
                hash_val = info.get('hash', "unknown")  # Retrieve font hash from cache
                mapped = len(self.db_cache.get(hash_val, set()))  # Count mapped glyphs

                # If this font is currently loaded, count mapped glyphs directly from current_glyph_set
                if (self.current_page == page_num and self.current_font_name == name and hasattr(self,
                                                                                                 'current_glyph_set') and self.current_glyph_set):
                    mapped = sum(1 for g in self.current_glyph_set.keys() if g in self.user_glyph_to_char)

                page_mapped += mapped  # Add to page statistics
                page_total += total  # Add to page statistics

            # Create icon and status based on mapping percentage
            if page_total == 0:
                icon = self.create_status_icon("#888888")  # Gray status icon
                status = "—"  # Set status information
            else:
                perc = page_mapped / page_total * 100  # Calculate percentage
                if perc >= 100:
                    icon = self.create_status_icon("#228B22")  # Green status icon
                    status = "100%"  # Set fill percentage
                elif perc > 0:
                    icon = self.create_status_icon("#FF8C00")  # Amber status icon
                    status = f"{int(perc)}%"  # Set fill percentage
                else:
                    icon = self.create_status_icon("#888888")  # Gray status icon
                    status = "—"  # Set status information

            page_menu = menu.addMenu(icon, f"Page {page_num + 1} [{status}]")  # Set pages icon and label

            # Filling page submenus
            for name in font_names:
                info = self.font_cache.get((page_num, name), {})  # Get font info from cache
                total = info.get('glyph_count', 0)  # Count all fonts for statistics
                hash_val = info.get('hash', "unknown")  # Retrieve font hash from cache
                mapped = len(self.db_cache.get(hash_val, set()))  # Count mapped glyphs

                # If this font is currently loaded, count mapped glyphs directly from current_glyph_set
                if self.current_page == page_num and self.current_font_name == name and hasattr(self,
                                                                                                'current_glyph_set') and self.current_glyph_set:
                    mapped = sum(1 for g in self.current_glyph_set.keys() if g in self.user_glyph_to_char)

                # Create icon and status based on mapping percentage for fonts
                if total == 0:
                    icon = self.create_status_icon("#888888")  # Gray status icon
                    status = "—"  # Set status information
                else:
                    perc = mapped / total * 100  # Calculate percentage
                    if perc >= 100:
                        icon = self.create_status_icon("#228B22")  # Green status icon
                        status = "100%"  # Set fill percentage
                    elif perc > 0:
                        icon = self.create_status_icon("#FF8C00")  # Amber status icon
                        status = f"{int(perc)}%"  # Set fill percentage
                    else:
                        icon = self.create_status_icon("#888888")  # Gray status icon
                        status = "—"  # Set status information

                action = page_menu.addAction(icon, f"{name} [{status}]")  # Set pages icon and label
                action.setData((page_num, name))  # Store data for loading fonts
                action.triggered.connect(
                    lambda checked, p=page_num, f=name: self.load_font(p, f))  # Connect clicking to loading font

    # Build fonts menu
    def build_fonts_menu(self, menu_data):
        menu = self.fonts_menu  # QMenu variable
        menu.clear()  # Reset menu

        unique = {}  # Dictionary to store unique valid fonts
        # Collect unique fonts across all pages
        for page_num, names in menu_data.items():
            for name in names:
                if name in unique:  # Skip if font already added
                    continue
                info = self.font_cache.get((page_num, name), {})  # Get font info from cache
                if info.get('glyph_count', 0) == 0:  # Skip fonts with no glyphs
                    continue
                # Store font info for later use
                unique[name] = {
                    'page': page_num,
                    'hash': info.get('hash'),
                    'glyph_count': info.get('glyph_count')
                }

        # If no valid fonts found, show disabled message
        if not unique:
            a = menu.addAction("No valid CFF fonts")  # Add action as text
            a.setEnabled(False)  # Disable action
            return  # Exit method early

        # Loop through unique fonts
        for name, data in unique.items():
            h = data['hash']  # Font hash
            total = data['glyph_count']  # Total glyphs in font
            mapped = len(self.db_cache.get(h, set()))  # Count mapped glyphs from DB cache

            # If this font is currently loaded, compute mapped using actual current_glyph_set (includes .notdef)
            if self.current_font_name == name and hasattr(self, 'current_glyph_set') and self.current_glyph_set:
                mapped = sum(1 for g in self.current_glyph_set.keys() if g in self.user_glyph_to_char)

            # Calculate mapping percentage
            perc = mapped / total * 100 if total else 0
            # Determine status icon and text based on mapping percentage
            if perc >= 100:
                icon = self.create_status_icon("#228B22")  # Green status icon
                status = "100%"
            elif perc > 0:
                icon = self.create_status_icon("#FF8C00")  # Amber status icon
                status = f"{perc:.0f}%"  # Rounded percentage
            else:
                icon = self.create_status_icon("#888888")  # Gray status icon
                status = "—"

            # Count how many times this font occurs across all pages
            occurrences = sum(1 for ns in menu_data.values() for n in ns if n == name)
            # Add action to menu
            action = menu.addAction(icon, f"{name} [{status}]")  # Font name with status
            action.setData((data['page'], name))  # Store page and font name for loading
            action.setToolTip(f"Mapped: {mapped}/{total} glyphs | Occurrences: {occurrences}")  # Tooltip info
            action.triggered.connect(
                lambda checked, p=data['page'], f=name: self.load_font(p, f))  # Connect action click to load font

    # Opening pdf file
    def open_pdf(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select PDF file to repair", "",
                                                   "PDF Files (*.pdf)")  # Use PySide6 to get file path
        # Exit if no file selected
        if not file_path:
            return

        self.pdf_path = file_path  # Save to variable
        self.statusBar().showMessage("Loading PDF and fonts...", 0)  # Show message on status bar
        QApplication.processEvents()  # Continues to load gui

        try:
            menu_data = extract_pdf_data(file_path)  # Get data from pdf
            self.font_cache.clear()  # Clear cache
            # Open file using PyMuPDF
            with fitz.open(file_path) as doc:
                first_page = first_name = None  # Initialize variables
                # Loop through each page
                for page_num in range(len(doc)):
                    page = doc.load_page(page_num)  # Get page object
                    # Loop through each font on page
                    for font in page.get_fonts(full=True):
                        try:
                            name, ext, _, buffer = doc.extract_font(
                                font[0])  # Extract name, font format, and raw binary data of the _font
                            # Is the font in cff format
                            if ext and ext.lower() == "cff":
                                font_hash = md5(buffer).hexdigest()  # Hash font data
                                tmp = CFFFontSet()  # Initialize library
                                tmp.decompile(BytesIO(buffer), None)  # Add data to temporary variable
                                glyph_count = len(tmp.topDictIndex[0].CharStrings)  # Count number of glyphs

                                # Add information to cache
                                self.font_cache[(page_num, name)] = {
                                    'hash': font_hash,
                                    'glyph_count': glyph_count,
                                    'data': buffer
                                }
                                # Check if first page was changed
                                if first_page is None:
                                    first_page, first_name = page_num, name  # Set variables to first occurrences
                                self.export_action.setEnabled(True)  # Enable Save PDF action
                        except:
                            # Error fallback
                            self.font_cache[(page_num, name)] = {
                                'hash': "unknown", 'glyph_count': 0
                            }

            self.load_db_cache()  # Load data from DB cache
            self.build_pages_menu(menu_data)  # Build pages menu
            self.build_fonts_menu(menu_data)  # Build fonts menu

            # Check if first page is set
            if first_page is not None:
                self.load_font(first_page, first_name)  # Load first font saved on first page
            else:
                self.clear_ui_state()  # Clear UI
                self.statusBar().showMessage("No CFF fonts in PDF", 3000)  # Set status bar message

        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Error while loading PDF:\n{e}")  # Error box
            self.clear_ui_state()  # Clear UI

    # Method which will save data to new PDF document
    def save_pdf(self):
        QtWidgets.QMessageBox.information(self, "Save PDF", "TBD")
        return

    # Load data from DB to cache
    def load_db_cache(self):
        self.db_cache.clear()  # Clear cache
        path = "glyph_mappings.csv"  # Path to csv DB
        # Check if DB exists, end early if not
        if not os.path.exists(path):
            return
        try:
            # Open DB
            with open(path, 'r', encoding='utf-8', newline='') as f:
                # Loop through data
                for row in csv.DictReader(f, delimiter='|', quotechar='"'):
                    h = row["font_hash"]  # Set hash to variable
                    g = row["Gcode"]  # Set glyph name to variable
                    self.db_cache.setdefault(h, set()).add(g)  # Add hash and glyph name to db_cache
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Error while loading DB cache:\n{e}")  # Error box

    # Save gathered data to DB
    def save_to_db(self):
        path = self.CSV_PATH  # Path to DB
        fieldnames = ["font_hash", "font_name", "Gcode", "unicode_hex", "AGN"]  # Predefined DB column names

        existing = {}  # Initialize dictionary
        # Check, if DB exists
        if os.path.exists(path):
            try:
                # Open DB
                with open(path, 'r', encoding='utf-8', newline='') as f:
                    # Loop through data
                    for row in csv.DictReader(f, delimiter='|', quotechar='"'):
                        key = (row["font_hash"], row["Gcode"])  # Get tuple of hash and glyph name
                        existing[key] = row  # Add whole row with tuple as a key
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Error", f"Error while loading DB:\n{e}")  # Error box

        font_hash = self.get_font_hash()  # Get hash of font
        font_name = self.current_font_name or "unknown"  # Get name of font

        # Check, if notdef is in glyphset
        if hasattr(self, 'current_glyph_set') and self.current_glyph_set and '.notdef' in self.current_glyph_set:
            # Check if notdef is in mapping
            if '.notdef' not in self.user_glyph_to_char:
                # Add information to mapping
                self.user_glyph_to_char['.notdef'] = {
                    "font_hash": font_hash,
                    "font_name": font_name,
                    "unicode_hex": "FFFD",
                    "AGN": "notdef"
                }

        # Loop through mapping
        for gname, data in self.user_glyph_to_char.items():
            key = (font_hash, gname)  # Get a key
            # Add items to existing dictionary
            existing[key] = {
                "font_hash": font_hash,
                "font_name": font_name,
                "Gcode": gname,
                "unicode_hex": data["unicode_hex"],
                "AGN": data["AGN"]
            }

        try:
            # Open DB
            with open(path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter='|', quotechar='"',
                                        quoting=csv.QUOTE_MINIMAL)  # Set up csv writer
                writer.writeheader()  # Add header to csv
                # Write all data in existing dictionary to file
                for row in existing.values():
                    writer.writerow(row)

            # refresh in-memory db cache and menus
            self.db_cache.clear()
            self.load_db_cache()

            data = extract_pdf_data(self.pdf_path)  # Get data from pdf
            self.build_pages_menu(data)  # Rebuild pages menu
            self.build_fonts_menu(data)  # Rebuild fonts menu
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to save:\n{e}")

    # Load mappings from DB for current font
    def load_mappings_for_current_font(self):
        font_hash = self.get_font_hash()  # Get hash of current font
        # End if there are no data
        if font_hash == "unknown":
            return

        self.user_glyph_to_char = {}  # Initialize mapping dictionary

        csv_path = self.CSV_PATH  # Path to DB
        # Check if there is DB
        if os.path.exists(csv_path):
            try:
                # Open DB
                with open(csv_path, 'r', encoding='utf-8', newline='') as f:
                    # Loop through rows
                    for row in csv.DictReader(f, delimiter='|', quotechar='"'):
                        # Check if hash and glyph name are in glyphset
                        if row["font_hash"] == font_hash and row["Gcode"] in self.current_glyph_set:
                            # Add data to user_glyph_to_char mapping
                            self.user_glyph_to_char[row["Gcode"]] = {
                                "font_hash": font_hash,
                                "font_name": row["font_name"],
                                "unicode_hex": row["unicode_hex"],
                                "AGN": row["AGN"]
                            }
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Error", f"Error while loading DB:\n{e}")

        # Add notdef to user_glyph_to_char if not there already
        if '.notdef' in self.current_glyph_set and '.notdef' not in self.user_glyph_to_char:
            self.user_glyph_to_char['.notdef'] = {
                "font_hash": font_hash,
                "font_name": self.current_font_name,
                "unicode_hex": "FFFD",
                "AGN": "notdef"
            }


# Main application starter
if __name__ == "__main__":
    app = QApplication()  # QApplication initialization
    window = FontWidget()  # FontWidget initialization
    window.show()  # Rendering main window
    sys.exit(app.exec())  # Close app on exit
