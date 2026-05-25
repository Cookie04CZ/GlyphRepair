## This program was originally created to repair text encoding in popular czech hobby magazines. If you wish to apply it to other PDF files, please skip to the [next chapter](#what-glyphrepair-does).

# Keywords

PDF copy-paste gibberish, mojibake, PDF font encoding repair, Type1 font, toUnicode table, Python, Windows executable

# Oprava textu v časopisech AMARO

Většina elektronických vydání (PDF souborů) časopisů A-Radio Praktická Elektronika a Konstrukční elektronika je špatně vygenerovaná, takže v nich nejde hledat ani kopírovat text. To je v dnešním informačním věku dost výrazná vada. Tento Python skript to umí opravit, přičemž je koncipován tak, aby jeho použití bylo co nejjednodušší. Abyste nemuseli instalovat Python a nezbytné knihovny, je zde připaven hotový spustitelný program "opravAR.exe", ve kterém už vše je. Skript dokáže opravit pouze originální PDF časopisy z CD a DVD, které byly vydány firmou AMARO, všechny ostatní soubory ignoruje. Díky tomu je "blbuvzdorný" a nebude nic dělat, pokud ho třeba omylem spustíte jinde, než jste chtěli. Pokud originální CD či DVD s časopisy nemáte, můžete si je koupit zde:

http://www.aradio.cz/cdrom.html

Pointa skriptu je, že každý čtenář si může svoji sbírku opravit sám - na časopisy se vztahuje autorský zákon a nelze je volně šířit. Postup použití je následující:

1. Někam na pevný disk z CD/DVD zkopírujte všechny soubory, které chcete opravit. Nejlepší je zachovat původní adresářovou strukturu po jednotlivých ročnících, skript automaticky hledá ve všech podadresářích.

2. Zde z Githubu si stáhněte a do stejného adresáře uložte soubory skriptu. Pro funkci jsou nezbytné jen 4 soubory: [opravAR.exe](opravAR.exe), [magazine_hash.json](magazine_hash.json), [Type1toUnicode.exe](Type1toUnicode.exe) a [to_unicode.json](to_unicode.json). Stahování se bohužel nespustí automaticky, u každého souboru musíte kliknout na "Download raw file" vpravo nad obsahem souboru. Alternativně můžete stáhnout všechny soubory najednou jako ZIP archív, dělá se to zeleným tlačítkem Code -> Download ZIP.

3. Spusťte opravAR.exe. Pravděpodobně se objeví [modré okno s varováním SmartScreen](https://github.com/user-attachments/assets/a067c6a6-d85b-4f68-8123-b2fc8f61d345), to musíte potvrdit. Tato okna se liší podle verze Windows, buď je tam přímo tlačítko "Přesto spustit" nebo nejdřív musíte kliknout na "Další informace". Samotná oprava pak typicky zabere několik minut, podle počtu souborů a výkonu PC. Skript nemá žádné GUI, výsledky jeho činnosti se zobrazují pouze v konzoli příkazové řádky. Většinou se v ní zobrazují pouze zelené řádky se statistikou oprav, u některých ročníků tam jsou i oranžové řádky s varováními. Ty můžete ignorovat. Nikdy by se však neměly objevit červené řádky s chybami.

4. Na disku se objeví opravené PDF soubory s koncovkou _repaired a také adresáře s podrobnějšími logy o průběhu opravy. Logy a původní PDF z CD/DVD poté můžete smazat. To uděláte nejsnadněji tak, že je nejdřív seřadíte podle data, originální PDF soubory jsou vždy starší než opravené.

Skript byl vyvíjen a testován pouze na Windows 10, funkci na jiných OS neznáme. Zde je pro ukázku jedna stránka (snad se firma AMARO nebude zlobit) před a po opravě, zkuste si z nich vykopírovat text:

[T1tU_sample.zip](https://github.com/user-attachments/files/16921939/T1tU_sample.zip)

Oprava podporovaných časopisů je téměř stoprocentní, včetně řecké abecedy a jiných "exotických" znaků. Většinou nejdou opravit jen stránky s reklamami, ale ty jsou irelevantní. Je nutné zdůraznit, že **skript umí opravit pouze časopisy A-Radio Praktická Elektronika (2000-současnost), Konstrukční elektronika (2000-2011) a Electus (2000-2007).** Tyto časopisy mají totálně špatné kódování, takže text je v nich "rozsypaný čaj" a nejde v nich vůbec vyhledávat. Skript opravuje PE do roku 2022, postupem času ho budeme aktualizovat. Zde je přehled, co skript aktuálně (verze 0.4.0) umí či neumí opravit:

![Prehled_AR_v040](https://github.com/xgmitt00-220814/Type1toUnicode/assets/169207159/4dafd779-fbe8-4540-8648-d66c8e9a8c9d)

Časopisy Amatérské rádio (řada A + řada B, později Stavebnice a konstrukce) sice také mají špatné kódování, ale projevuje se to jen v některých PDF prohlížečkách. Hlavně v Adobe Readeru se nesprávně kopírují české znaky, naštěstí jiné prohlížečky (SumatraPDF, Mozilla Firefox, Evince, Google Chrome a jeho klony) je dekódují správně. Při jejich čtení se proto Adobe Readeru vyhýbejte, je obecně dost háklivý na správnou syntaxi PDF. Kódování Amatérských rádií by šlo sice také šlo opravit,  ale byla by to zbytečná práce.

Je nejasné, proč všechny ty časopisy mají špatné kódování textu. Nicméně je/bylo to **Amatérské** radio a ten amatérizmus se holt projevil i tímto způsobem. Naštěstí **po přechodu na nový grafický design od PE 04/2023 je už kódování správně** a v časopisech jde konečně normálně hledat bez ohledu na prohlížečku.

Opravný skript vzniknul v rámci diplomové práce ["Skripty pro hromadnou úpravu fontů v PDF dokumentech"](https://hdl.handle.net/11012/246071) na [Ústavu telekomunikací](https://www.utko.fekt.vut.cz/) na [Vysokém učení technickém v Brně](https://www.vut.cz/). Tento český a anglický návod byl vytvořen vedoucím práce. Pokud vás zajímá, jak skript interně funguje, přečtěte si tu diplomku (slovensky) nebo anglický návod níže.

 # What GlyphRepair does
This program is designed to repair wrong text encoding ("mojibake") in PDF files -- text looks fine on screen, but you get only gibberish when you try to copy+paste it. This may be fixed via OCR (Optical Character Recognition), but it always recognizes some characters wrong, particularly in multi-lingual and/or scientific texts which contain special symbols. OCR also usually destroys ("flattens") original vector content of the source PDF, which is generally undesirable. GlyphRepair works around these limitations:

* Meaning of each character (glyph) is manually defined (mapped) by user, which allows for 100% text fidelity.
* GlyphRepair preserves original document data, only adds new text encoding tables to it.

However, these advantages come at a price. The program is designed to work only with older PDF documents which use so-called [Type 1 fonts](https://en.wikipedia.org/wiki/PostScript_fonts#Type_1). And manual mapping of the characters can become a time-consuming task, although GlyphRepair auto-suggests them.
 
 # Before you start

Do you really need to permanently fix your PDF files? Or do you merely need to copy some text? If so, there may be a faster way: **[open-source viewer Evince](https://wiki.gnome.org/Apps/Evince) can return meaningful text even on files that are completely garbled in other PDF viewers** (we tested Adobe Reader, Sumatra PDF, PDF-XChange Viewer, Mozilla Firefox, Google Chrome and others). It's probably because Evince internally uses some sort of heuristics. Nevertheless, even Evince will usually correctly copy only standard ASCII characters (codes 32 to 126); special characters for foreign languages may still be garbled. And unfortunately, Evince is currently available only on Linux.

# How to run GlyphRepair

You should download the program and glyph_mappings.psv database and put them to the same directory. You can run Python code directly or use Windows executable we compiled. The executable already contains all the necessary packages, so it runs right out the box. You will probably encounter [blue SmartScreen filter warning](https://github.com/user-attachments/assets/a067c6a6-d85b-4f68-8123-b2fc8f61d345) when you run it for the first time. These warnings vary between Windows versions, either there is "Run anyway" button or you need to click on "More information" first.

If you want to run Python code, you have to install following packages:
* NumPy 2.0.2 https://numpy.org/
* PyMuPDF 1.26.5 https://pymupdf.readthedocs.io/en/latest/#
* PySide6 6.10.3 https://doc.qt.io/qtforpython-6/PySide6/QtWidgets/index.html
* Matplotlib 3.9.4 https://matplotlib.org/
* qtawesome 1.4.2 https://github.com/spyder-ide/qtawesome
* colorama 0.4.6 https://github.com/tartley/colorama

All can be installed with pip
```
pip3 install NumPy
pip3 install PyMuPDF
pip3 install PySide6
pip3 install Matplotlib
pip3 install qtawesome
pip3 install colorama
```
Note the program was developed and tested only with these package versions and only on Windows. We have no idea if it works on other operating systems. 

# Testing whether your PDF file can be repaired

As we mentioned earlier, GlyphRepair currently supports only one PDF font type which is prevalent in older PDF files. That's because old files are also the most likely to have wrong text encoding. Therefore, GlyphRepair automatically detects and displays only fonts which it can actually repair. The easiest method is to simply load your file into GlyphRepair and click to the middle of font selector to see list of fonts:

<p>
<img width="1202" height="855" alt="Analysis base" src="https://github.com/user-attachments/assets/679622f7-dccd-4f1f-8105-991d49296870" />
<p>

 You should switch the page filter to "All pages" to see all fonts in the document. Most of them will probably have grey or orange dots next to them, which means all of some of their characters are unknown (unmapped):

<p>
<img width="652" height="532" alt="Analysis font list" src="https://github.com/user-attachments/assets/94d41779-0d58-4ddf-b972-a59ceb79aadc" />
<p>

If your file doesn't contain any repairable fonts, you will get message "This document does not contain any fonts that can be repaired."

# Mapping your first document

You can test GlyphRepair on this sample 2-page document, provided under [fair use](https://en.wikipedia.org/wiki/Fair_use) doctrine:

[GR_Sample.zip](https://github.com/user-attachments/files/28198468/GR_Sample.zip)

Note the sample will appear as fully mapped if you load it with default glyph_mappings.psv database from main repository. Therefore, the sample also contains much smaller glyph_mappings.psv which contains mapping only for one font. Simply overwrite it; you can always download full database again.

GlyphRepair is most effective if you need to repair multiple documents that come from the same source and/or contain the same fonts. It builds a database of unique graphemes (glyphs) that are visible in the document and you have to assign (map) which characters they represent. You have to do this mapping only once for a given font -- GlyphRepair automatically recognizes glyphs you've previously mapped, even in different documents. Glyphs are displayed vertically in the work area on the left. Simply enter their meaning into Character field and press Enter. You may enter up to 3 characters for each glyph, which is sometimes needed for so-called [ligatures](https://en.wikipedia.org/wiki/Ligature_(writing)). If you make a mistake, simply click on the glyph in the work area and re-enter character for it. You can also move up and down the glyph list with cursor keys.

<p>
<img width="1202" height="855" alt="First document base" src="https://github.com/user-attachments/assets/9dc13089-4a5b-48be-be9e-37926183bdb9" />
<p>

You have to assign character mapping for **all** fonts you wish to repair, because even slightly different glyphs are regarded as unique. Fortunately, glyphs in real-world documents have their internal names which are (usually) not random. These Glyph Names are displayed in the work area and replaced with actual character once you map them. The program analyzes existing Glyph Names in the database and automatically suggests the most probable mapping. These suggestions are displayed below the work area:

<p>
<img width="1202" height="855" alt="First document prompter" src="https://github.com/user-attachments/assets/58e57a5d-ed25-4153-b7be-278eb6088f6f" />
<p>

If the suggested character matches the glyph, simply press Enter to confirm it. GlyphRepair offers up to 4 suggestions; you may choose between them with left and right cursor keys (+Enter) or simply click on them. That immediately maps them. In ideal case, you will have to map each character only once and then it will be auto-suggested for all other fonts. If you do it right, you will just keep pressing the Enter key, only ocassionally stopping to fill the gaps. The working area is designed to streamline the process: the active glyph is usually right above the suggestions, so you'll just gaze into the bottom-left corner and press Enter. Therefore, **try to do the initial mapping as accuarately as you can,** because mistakes may then propagate to other fonts! Also, **you have to map all glyphs in a given font, otherwise program will be unable to repair it**. This is necessary due to internal limitations of the repair method.

Notice that GlyphRepair automatically recognizes and maps spaces (U+0020). If you encounter an unmapped empty glyph, it usually means it's no-break space (NBSP, U+00A0) or other special character. But that's rarely relevant when you copy+paste text from PDF, so you may map them to ordinary spaces (U+0020). Such special characters are sometimes hard to enter into the Character field. However, you may find their [Unicode](https://en.wikipedia.org/wiki/Unicode) encoding on various web pages. GlyphRepair allows to enter Unicode directly if you enable it in settings:

<p>
<img width="452" height="626" alt="Settings Unicode hex enable" src="https://github.com/user-attachments/assets/a92ad410-a5d4-4178-9518-1c424827db68" />
</p>

This will display another input field which accepts only string of 4 or 5 hexadecimal characters (0-9, a-f, A-F). In other words, you must **not** enter the U+ prefix that's customary with Unicode.

<p>
<img width="1202" height="855" alt="First document Unicode enabled" src="https://github.com/user-attachments/assets/7b7abc75-50c3-420d-a243-ce87ec42498a" />
</p>

You track your mapping progress near top of the window. You can click and switch the progress bar between 4 modes: glyphs on current font, glyphs on current page, glyphs in entire document or finished page counter:

<p>
<img width="262" height="197" alt="Mapping progress bars" src="https://github.com/user-attachments/assets/f0420f3f-e403-4dd2-93c1-10c1dfa4b2f3" />
</p>

When you map all glyphs in entire document, you'll get this message window:

<p>
<img width="344" height="129" alt="All glyphs mapped" src="https://github.com/user-attachments/assets/d74477a3-f031-43cc-9d1c-d0bd98aaecf3" />
</p>

## Deciding between uppercase and lowercase characters

Whenever possible, GlyphRepair displays two thin blue guidelines that should help you decide whether glyphs are uppercase or lowercase characters. Another clue may be glyph order in the work area, because they as we'll [explain later](#how-glyphrepair-works-internally), the order frequently represents first words in the given font. Unfortunately, there are also fonts which have undefined character height, so displaying the blue lines is impossible. In these cases, GlyphRepair displays only glyph's baseline in red. That makes the decision harder for certain characters like C, O, S, V, X or Z:

<img width="827" height="446" alt="Character height guidelines" src="https://github.com/user-attachments/assets/f1e2fdb4-8bab-4ad7-8ecd-84943ebd73aa" />

## Where to find and copy special characters

GlyphRepair has button Special Characters which links to a [web page of common scientific, typographical and dingbat symbols](https://www.vertex42.com/ExcelTips/unicode-symbols.html), so you could easily copy and paste them. But of course there are many other sites with searchable lists of Unicode characters:

* https://unicodeplus.com/
* https://www.compart.com/en/unicode
* https://unicode.org/charts/

However, you may find all the characters you need in once place. Type 1 fonts are usually based on legacy character sets, so it could be useful to check your file's metadata for clues what they may be. Legacy character sets [varied between languages and operating systems](https://en.wikipedia.org/wiki/Code_page), but usually they're easy to guess. In our case, the magazines were authored in a Windows program and therefore they are based on Windows-1250 code page for Central European languages. So we simply found a web page with a table of all Windows-1250 characters, like [this one](https://cs.wikipedia.org/wiki/Windows-1250#Mapov%C3%A1n%C3%AD_do_Unik%C3%B3du).

# Glyph database and its impact on auto-suggestion

As we already mentioned, the database is stored in glyph_mappings.psv file. It currently contains about 38 thousand glyphs, most of them for Arial, Times New Roman and Courier fonts by [Monotype Corporation](https://en.wikipedia.org/wiki/Monotype_Imaging), which used to be bundled with legacy Adobe products. If your documents use other fonts, the auto-suggestion feature may offer wrong characters. If it keeps happening, it may be best if you start with a blank database. Also, the program will work a bit faster when the database is smaller. You can simply delete the glyph_mappings.psv file, because GlyphRepair will create an empty database if it doesn't find it upon start.

By default, GlyphRepair auto-saves the database whenever you finish mapping a font or entire document. This auto-saving feature can be disabled in the Settings. You can also save the database manually at any time with Save All to DB button.

# Mapping another document

When you load another similar document into GlyphRepair, you will notice that many glyphs in the work area are already green. That means you've already mapped and saved them into the database. However, real-world documents rarely use exactly the same set of glyphs, so you need to find and map the missing ones. For that, you have to use the Next Unmapped button. In practice, GlyphRepair will make large jumps in the work area or even skip entire fonts, making manual navigation difficult. If you think you've made a mistake, use Previously Mapped button to easily go back. Be aware that this glyph mapping history is remembered only for currently opened document.

<p>
<img width="1202" height="855" alt="Another document base" src="https://github.com/user-attachments/assets/e8e91226-7299-4972-9aa3-d859883a51d6" />
</p>


# Mapping only selected pages or fonts

Real-world documents may contain dozens of fonts, which means you'd have to map hundreds or even thousands of glyphs. In many cases, that's unneccessary, because only a few fonts hold the bulk of useful text. Or you may want to repair only specific pages and ignore all others. GlyphRepair is designed to help you with that, because you can enable Page Mode Navigation in the settings. If you do so, new page selector will appear above font selector:

<p>
<img width="1202" height="855" alt="Page mode base" src="https://github.com/user-attachments/assets/8232fb25-5411-4cc9-9c5e-c354d17fc91b" />
</p>

The font selector will now cycle only through fonts which are used on the selected page. You may quickly switch between pages if you click in the middle of the page selector and simply choose page in the menu that appears. The menu also displays how many fonts are used on every page and colors indicate their mapping status:

<p>
<img width="452" height="532" alt="Page mode select menu" src="https://github.com/user-attachments/assets/c5c68f5a-0a8a-4a9e-a919-1da6815ffdf0" />
</p>

Note that Next Unmapped button is **not** limited to selected page -- it always searches all fonts within the document and will automatically jump to other pages, even with Page Mode Navigation enabled.

If you want to repair only a specific font, you don't need the Page Mode Navigation. Instead, you can switch to it with font selector as explained in previous chapters. The real trick is to know which font you actually need to select, because most PDF viewers don't tell you which parts of the text use which font. You will have to use a 3rd party program for that. We've been using two programs, [Infix PDF Editor](https://infix-pdf-editor.en.softonic.com/) and [PDF-XChange Editor](https://www.pdf-xchange.com/product/downloads/enduser/pdf-xchange-editor). Both have free trial versions. They both work similarly -- activate their text editing mode, click somewhere into text and they will display its font:

<p>
<img width="880" height="570" alt="Select font in Infix" src="https://github.com/user-attachments/assets/eff8d2dc-4191-488c-aef6-3d7287ba870d" />
</p>
<p>
<img width="1026" height="716" alt="Select font in XChange" src="https://github.com/user-attachments/assets/97777af8-ad07-4cba-85f2-bd52db3c3957" />
</p>


 # Saving repaired document

**Important! Always load and repair only original documents!** While you can re-load documents which were already processed by GlyphRepair, it may lead to unpredictable results. You mapping work is actually saved in glyph_mappings.psv, so there is no need to also save unfinished documents.

As you've probably guessed, you have to use Repair PDF button in the top left corner. Remember: you have to always map **all** glyphs in a font, otherwise the repair algorithm will skip it. That's why the repair menu displays list of all pages again, with their mapping status indicated by colors:

<p>
<img width="552" height="432" alt="Repair with missing glyphs" src="https://github.com/user-attachments/assets/0daa912d-cf6d-4893-a23d-f26eeec3dce2" />
</p>

The big Repair button will be green only if all fonts are 100% mapped. If it's orange, it will repair only fonts that have all glyphs mapped. Repaired file will be saved to the same directory as source file; program will attach suffixes _Repaired or _Partially_Repaired to their name.

 **TBD**

 # What is AGL?

You may notice that GlyphRepair automatically skips glyphs and fonts that are displayed blue in the GUI. These are Type 1 fonts whose glyphs have special naming scheme [Adobe Glyph List](https://en.wikipedia.org/wiki/Adobe_Glyph_List). Theoretically, glyph named "Aacute" should always represent letter "Á" and so on, you can see [their full list here](https://github.com/adobe-type-tools/agl-aglfn/blob/master/glyphlist.txt). Unfortunately, it's not always true in practice, so we played with the idea that GlyphRepair could remap AGL glyphs, too. But the current GlyphRepair version **does not** support it and leaves all such fonts untouched.

<p>
<img width="1202" height="855" alt="AGL font example" src="https://github.com/user-attachments/assets/95c85c04-2f77-444c-82a0-b6267f72c30d" />
</p>

 # How GlyphRepair works internally 

Glyphs in Type 1 fonts are stored as vector instructions in PostScript language. Even visually very similar glyphs have slight differences in vector coordinates, which can be detected. GlyphRepair extracts raw binary data from each font, decodes them into separate PostScript chunks and then calculates MD5 hash from them. The reason for this is threefold:
1. Even slight difference in glyph PostScript instructions will result in a completely different hash.
2. The resulting hash always has the same length, which is useful for storing them in database.
3. Many fonts are copyrighted, so you can't store and distibute their original data, anyway.

While you're creating new mappings for a document, the data flow is:

**Read glyph data -> calculate MD5 hash -> pair the hash with user-assigned character -> store Unicode code for the character into database.**

There's a reason why mapped characters are stored as Unicode. In old PDF files with Type 1 fonts, glyphs are just graphical symbols that may or may not contain information about which character they actually represent. Moreover, PDF supports several schemes to reduce overall file size, so it typically stores only glyphs that are needed to render the given document. These are called "embedded subset" fonts. Another file size reduction comes from character ordering. In embedded subset fonts, characters are ordered by their appearance in the text. In other words, every font has different character order. Suppose you have a document that starts with word "OUROBOROS", then characters in its font will get these character codes (CC):

| Letter | O |U |R |O |B |O |R |O |S |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Character code (CC) | 1 |2 |3 |1 |4 |1 |3 |1 |5 |

Notice that CC for letter "O" gets repeated every time it's needed. These character codes are linked with glyphs, so the renderer knows what to draw at each code position. Glyphs have their own Glyph Names (GN) which may be linked to CCs like this:

| Letter | O | U | R | B | S |
|:---:|:---:|:---:|:---:|:---:|:---:|
| Character code (CC) | 1 | 2 | 3 | 4 | 5 |
| Glyph name (GN) | G79 | G85 | G82 | G66 | G83 |

Unlike Adobe Glyph List, such Glyph Names don't reliably convey which character they actually represent. **That's the real reason why you get only gibberish when you try to copy+paste from some PDF documents.** Moreover, Type 1 fonts are limited to about 220 characters, which quickly became insufficient for modern documents. So in 1996, Adobe introduced ToUnicode tables into PDF version 1.2. These are separate tables that link character codes with their [Unicode](https://en.wikipedia.org/wiki/Unicode) equivalent. For OUROBOROS, the ToUnicode table would look like this:

| Letter | O | U | R | B | S |
|:---:|:---:|:---:|:---:|:---:|:---:|
| Character code (CC) | 1 | 2 | 3 | 4 | 5 |
| ToUnicode | 004F | 0055 | 0052 | 0042 | 0053 |

GlyphRepair fixes document encoding by creating and injecting new ToUnicode tables for each font. In simplified form, it works like this:

**Read glyph data -> calculate MD5 hash -> look up the hash in database -> read associated Unicode from database -> create CC-Unicode pair**

When all pairs are created, they are compiled into a ToUnicode table and injected into the PDF. This ensures that the document's original contents are preserved. Another advantage is that repaired file size increases only slightly.

# glyph_mappings.psv database format

The glyph database is designed to be human-readable and editable. It's a [Pipe-Separated Values](https://docs.amperity.com/reference/format_psv.html) file where each row represents one glyph. There are 5 columns:

* MD5 hash computed from glyph's PostScript data.
* Name of glyph's source font when mapping was done.
* Glyph Name copied from the source font.
* Unicode of character, assigned (mapped) by user.
* Adobe Glyph List name, if the mapped character has one. The program calculates these solely to ease searching; they aren't otherwise used.

In practice, it looks like this:
```
0089549e52807f487abfda979c833058|AOGNPO+Arial.tu.n.0150|G48|0030|zero
0089e72a1831b9725ce36eb6a0852e60|GOBEDD+Arial035|G35|0023|numbersign
008db3d884e87e669a851081b6847bb1|BIKJJG+Arial.tu.n.095.813|G110|006e|n
008e368c96dc9b9c2940f04a776bb12d|AJELJP+Arial.tu.n.050|G98|0062|b
0090901c94af61bbe3aafc2b123585ef|AJFICH+Arial035.813|G47|002f|slash
0093634fbd7c628df8598147529265de|AFGFPL+Arial.tu.n.0111.125|G106|006a|j
```
GlyphRepair has no way to "unmap" or "forget" fonts. If you want to delete them from the database, you have to search and delete their rows in glyph_mappings.psv. You can easily import the file to MS Excel or other similar program to search, sort or otherwise modify it.

## Why so many font names start with strings like ABCDEF+ ?

As we mentioned in previous chapter, Type 1 embedded subset fonts store only glyphs that are needed to render the given document. The PDF standard stipulates that names of such fonts must start with so-called Font Subset Tag, which is a string of 6 random uppercase letters A-Z, followed by +. You should ignore these tags when searching the fonts, as they don't convey any meaningful information.


 # Repairing multiple documents at once

GlyphRepair has a command line interface that allows you to repair many files at once. You can display help if you run it with -h or --help option:
```
usage: GlyphRepair_v19.py [-h] [-m] [Target directory] [-r] [-d HASH_DB] [-v]

PDF Glyph Repair Tool - GUI & CLI

positional arguments:
  Target directory      Path to target file or directory

optional arguments:
  -h, --help            show this help message and exit
  -m, --multiple        Enables automated repair of multiple input files
  -r, --recursive       Also repairs files in subdirectories (-m required)
  -d HASH_DB, --hash-db HASH_DB
                        Path to hash database of known input files
  -v, --verbose         Enables verbose output
```
For example, if you want to repair all PDF files in given directory and subdirectiories, you can use
```
GlyphRepair -m "c:\PDFs are here" -r -d known_docs.psv
```

**Note that -d option is mandatory**, because you know how it is: most people have a mess in their PCs. We originally devised the program to repair popular hobby magazines and we had to make sure it doesn't touch any other PDF files it may find. Therefore, the multiple repair function has a safety feature built in it: **it repairs only files whose MD5 hashes are stored in known_docs.psv database**. This PSV file has only two columns, MD5 hash and name of the source PDF file. Note that only the MD5 hash decides whether a file will be repaired; you may leave the file name column blank. 
```
6693b059e2f51b9f48b52bc9e2355cd8|_PE11_2002.pdf
496aabde27329d0055411f571af03656|_PE12_2002.pdf
62909c8922a57254ef82cd18f076573d|Electus2003.pdf
56d32e4d771ce352d59dbb862e8032cf|KE01_2003.pdf
```
On Windows, you can use a system utility to calculate the MD5 hash:
```
certutil -hashfile ABCD.pdf md5
```
To hash multiple PDF files in a directory, you can use
```
forfiles /m *.pdf /c "cmd /c certutil -hashfile @file md5" 
```
You can also include subdirectories with ```forfiles /s``` option. We currently don't have any program or utility to collect or filter the resulting MD5 hashes, so'll have to do it manually in a text editor etc. During repair, the console displays progress bars and font statistics which are similar to GUI:

<p>
<img width="544" height="182" alt="Multiple repair result" src="https://github.com/user-attachments/assets/ee0660c1-9781-441d-bc5d-7906e6ad55cb" />
</p>

You can further expand these results if you run GlyphRepair with -v or --verbose option.

 # Other ways to fix your documents, but with lower fidelity

If you don't need to preserve document's fidelity, garbled text can be permanently fixed via OCR. Each page is rendered as ordinary bitmap image (it's called "flattening") and then fed to OCR. However, most OCR algorithms still struggle with diacritics, math and/or non-latin characters, so the extracted text usually contains errors. Also, vector graphics may not be preserved, depending on how smart the OCR algorithm is. That's usually highly undesirable and may significantly increase file size. 

There **are** programs that can fix garbled PDFs via OCR while (mostly) preserving vector contents of the document. One of them is open source [Ghostscript by Artifex Software](https://www.ghostscript.com/). A few years ago, they [enhanced its PDFwrite output device](https://ghostscript.readthedocs.io/en/latest/Devices.html#vector-pdf-output-with-ocr-unicode-cmaps) so it behaves similarly to Type1toUnicode. Internally, it uses (also open source) [Tesseract OCR](https://tesseract-ocr.github.io/tessdoc/Installation.html) to recognize the text. Sadly, Artifex doesn't advertise this feature much and there were several bugs that prevented it from working properly. You need to install Ghostscript 10.06.0 or newer if you want to use it. It comes bundled with its own copy of Tesseract OCR, so you don't need to install it separately. On Windows, you'll need to [set up TESSDATA_PREFIX environment variable](https://ghostscript.readthedocs.io/en/latest/Devices.html#ocr-text-output) to a directory with [Tesseract language files of your choice](https://tesseract-ocr.github.io/tessdoc/Data-Files.html). Ghostscript has many options and usually requires several parameters to do what you want. For example, to repair our sample file, it's necessary to use
```
gswin64c -dNOPAUSE -sDEVICE=pdfwrite -sUseOCR=AsNeeded -sOCRLanguage="ces" -dAutoFilterColorImages=false -dColorImageFilter=/FlateEncode -dAutoFilterGrayImages=false -dGrayImageFilter=/FlateEncode  -sOutputFile=sample_repaired_Ghostscript.pdf sample.pdf -c quit
```
Notice the -sUseOCR=AsNeeded parameter. As explained in the [link above](https://ghostscript.readthedocs.io/en/latest/Devices.html#vector-pdf-output-with-ocr-unicode-cmaps), this directs Ghostscript to recognize only text which doesn't have valid Unicode representation. That's the best option in our case, because "our" magazines usually contain at least a few pages of properly encoded text. Therefore, it would be counter-productive to process it with (unreliable) OCR. You can download our [sample fixed with Ghostscript here](https://github.com/user-attachments/files/22067056/sample_repaired_Ghostscript.pdf). If you copy text from it and compare it with sample repaired by our script, you will notice that several characters were recognized wrong. This is mostly caused by how Ghostscript and Tesseract work together; if you're interested, one [Ghostscript developer explained it here](https://bugs.ghostscript.com/show_bug.cgi?id=708548#c1). It also has one nasty side effect: if a character is recognized wrong, the same error then appears in all other instances of that character. But of course that's still better than completely garbled text.

**Be warned that using Ghostscript has many other caveats.** Ghostscript actually completely rebuilds the input file, resulting in an entirely new PDF that only closely resembles the original one. In other words, even though it preserves vector content, it's still much more destructive than Type1toUnicode. This is apparent even in our simple sample, for example the main body font is slightly thicker. Even worse changes can occur with bitmap images, because by default, Ghostscript [recompresses them to optimize file size](https://ghostscript.com/blog/optimizing-pdfs.html). It's complicated to suppress this behavior, but in most cases it can be done via "Distiller Parameters". In our example, there are 4 "Filter" parameters that disable recompression for lossless grayscale and color images. Recompression is disabled by default for lossy (JPEG/DCT) images, but it can still kick in for large images (see PassThroughJPEGImages parameter). [Full list of these Distiller Parameters is here](https://ghostscript.readthedocs.io/en/latest/VectorDevices.html#distiller-parameters), but in our experience they can lead to counter-intuitive results. Be sure to examine your output files closely if you decide to give Ghostscript a try.



# Known limitations and issues

* Type1toUnicode doesn't just inject new toUnicode tables into existing files. The PyPDF library actually completely rebuilds them, so all PDF objects get new IDs, page tree will have different hierarchy etc. While Type1toUnicode preserves metadata, other PDF settings in the root are lost. It's possible other data (attachments, multimedia objects) may get lost, too. **Double check the output files!**

* When log reports a missing glyph definition, the glyph may be on a different page than indicated in the log. That's because one font may used on multiple pages and the script lists only the first occurence of the font (i.e. not page where the missing glyph is actually used).

* Type1toUnicode can repair only Type1 fonts that have complete Differences table, i.e. every character (glyph) must be replaced. That's not typical.

* toUnicode table must be completely missing, Type1toUnicode can't be used to replace existing one.

* If JSON mapping table is incomplete, Type1toUnicode replaces undefined characters with spaces (U+0020). If you then copy text from the "repaired" file, it looks deceptively "clean", i.e. without any garbled characters.

* There are no safeguards against duplicate font/glyph names or other user errors in the JSON file. The script does report its syntax errors, however.

* In retrospect, JSON wasn't the best choice to store the mapping data, because it doesn't allow for comments. It would be nice to see what letter each GN-Unicode pair actually represents...

* We've seen documents with multiple fonts whose names were empty strings (even though PDF standard prohibits it). Type1toUnicode's logs and final statistic will count them incorrectly.

* Real-world documents may contain other PDF standard violations which can cause erratic behaviour. Your mileage may vary.

# Script opravAR for user-friendly repair

As mentioned at the beginning, Type1toUnicode was originally created to repair encoding in popular czech hobby magazines. Of course, the magazines are copyrighted. So the idea is that every subscriber can download these scripts and repair their own copies of the magazines. But we had to make sure it would repair only the magazines and nothing else. We couldn't rely on file names, because let's be honest here, most people's HDD is a mess. Thus opravAR searches all directories and subdirectories for PDF files, computes their SHA-256 hash and compares it with a list of known (repairable) magazines. This list is stored in [magazine_hash.json](magazine_hash.json). If a hash match is found, opravAR calls Type1toUnicode which performs the actual repair.

XXXXXXXXXXXXXXXXXXXXx TBD

If you want to use opravAR, you'll need SHA-256 hashes of files you want to repair. On Windows, you can use its built-in certification utility
```
certutil -hashfile ABCD.pdf sha256
```
To hash multiple files at once, you can use
```
forfiles /m *.pdf /c "cmd /c certutil -hashfile @file sha256" 
```
BTW, "oprav" means "repair" in Czech and "AR" is abbreaviation of magazines' main title "A-Radio". They're similar to "Popular Electronics" or "Elektor", except they're even older, as they're being published continuously since 1952. 

# The gory details
Type1toUnicode allows you to analyze fonts within your PDF files (-v switch), but before it existed, we had to do it some other way. Probably the most user-friendly program is [PDFtalk Snooper](https://pdftalk.de/doku.php?id=pdftalksnooper). It has GUI that can display entire internal PDF object tree and even decode some of the objects. If you want to analyze a font, you need to select the appropriate page and then expand Resources -> Font -> Font Name. Here is how it looks like for [the sample file](https://github.com/user-attachments/files/16921939/T1tU_sample.zip):

![Snooper_tree](https://github.com/xgmitt00-220814/Type1toUnicode/assets/169207159/fe797bbb-9f00-419b-8d70-1f5af301084c)

Sadly, it seems PDFtalk Snooper's development has stalled and it outright crashed (unhandled exception) on about 1/5 of files we tried. (Maybe the authors would fix them if enough people asked?) Despite that, it's a very handy tool and we've been using it extensively.

As we mentioned in the [analysis chapter](#analyzing-your-pdf-files), Type1toUnicode automatically skips all fonts that don't meet certain criteria. These deserve to be explained in greater detail, so let's list them:

* Must be Type1 font.
* Its toUnicode table must be missing.
* FontDescriptor must exist and must contain FirstChar and LastChar entries.
* All characters in the font must be replaced via Differences table.

Read [this article](https://www.gnostice.com/nl_article.asp?id=383) to give you some idea about supported font types and encoding combinations. Displaying font type is easy and many PDF viewers can do it. Here is how it looks in Adobe Reader XI:

![Reader_XI_fonts](https://github.com/xgmitt00-220814/Type1toUnicode/assets/169207159/8c56af8e-1313-4a6d-8641-d6085b8f3841)

Obviously, if a font is not Type1 font, Type1toUnicode will print "Font has other type than Type1 -> skipping" into the (verbose) log. If a font already has a toUnicode table, then it prints "ToUnicode already exists -> skipping" into the log.

Like explained in previous chapters, Type1toUnicode requires glyph names (GNs) and their order to work, but they're surprisingly hard to obtain. Normally they're hidden in the font's raw data (PDFStream) which is embedded within the PDF file. Type1 fonts are in PostScript language and the stream is normally decoded with a PostScript interpreter. However, such intepreters usually aren't designed to collect glyph names and pass them to an "outside" program. Glyph names and their data are stored in /CharStrings dictionary, so we would need to *somehow* loop through all the entries and get what we need. There also are other complications, such as undefined characters etc. PDFtalk Snooper doesn't help much here - you can view the data if you expand Font Name -> FontDescriptor -> FontFile:PDFStream -> Internal sub-tree, but they look like gibberish. So we'd either have to "bend" some existing PostScript interpreter for our purposes or write our own parser.

![Snooper_internal](https://github.com/xgmitt00-220814/Type1toUnicode/assets/169207159/961f82a6-cd31-4222-86bb-c1272b903ca5)

Fortunately, there was another way. Most of the czech magazines used fonts whose characters were completely replaced via Differences table. Differences table was originally devised as a supplement for legacy encodings like WinANSI or MacRoman. These were limited to 255 characters, which is not nearly enough to cover most Latin-script languages (let alone other languages). If other characters were required, they could be replaced via Differences table. Such fonts are then regarded as having "custom encoding", like in the font list above. Normally, the number of replacements is low, but for some arcane reason, all characters were replaced in the magazines. Such Differences tables contain GNs in the same order as CCs, making it was unnecessary to decode font data. If a font has Differences table, you can view it if you expand its Encodning subtree in PDFtalk Snooper:

![Snooper_Differences](https://github.com/xgmitt00-220814/Type1toUnicode/assets/169207159/cd3b540d-bef0-44f7-b60c-41e2e490a5b1)

One last obstacle remained - how to check that all characters were really replaced, again without decoding font data. This is where FirstChar and LastChar entries come in. If LastChar-FirstChar matches the numbner of entries in the Differences table, then it means all characters were replaced. If they're not, the script prints "no ToUnicode but Differences incomplete -> skipping" into the log. In some fonts, FirstChar, LastChar and/or entire FontDescriptor is missing ([PDF standard allows it](https://www.verypdf.com/document/pdf-format-reference/txtidx0413.htm)). Type1toUnicode can't repair such fonts and prints "FontDescriptor entries missing -> skipping" messages in the (verbose) logs. In PDFtalk Snooper, FirstChar and LastChar are displayed here:

![Snooper_first_last](https://github.com/xgmitt00-220814/Type1toUnicode/assets/169207159/3811fff2-16ca-4423-b379-d8930032c0f4)


# Possible further work
Just some ideas in case someone wants to build upon this...

* Type1toUnicode now requires Differences table for all glyphs in the font. Theoretically, it could be modified to also fix fonts that employ the original idea behind Differences, i.e. take some standard encoding like WinANSI and replace only a few glyphs within it. There are many unknowns, for example how to reliably identify the original encoding and GNs within it.

* Manual glyph identification via Infix PDF Editor is still too laborious. Feed the glyphs into OCR and construct JSON mapping automatically? User could only confirm and/or manually correct whatever glyphs OCR doesn't get right.

* Use image similarity algorithm and/or OCR to automatically cross-compare GNs between multiple documents?

* Or even better: use OCR/user input to identify the glyphs, but store their hash. Hash would be computed from the glyphs' binary data. Build big database of such glyph hashes. Script would then construct toUnicode tables based on the hashes, not GNs. It would be possible to fully automatically repair even documents where GNs are arbitrary. (Many fonts are copyrighted, so only glyph hashes can be distributed legally). Unfortunately, glyph data may change with font's size (height) because of [hinting](https://en.wikipedia.org/wiki/Font_hinting) and other typographical "tricks". It happens extensively in the czech magazines, which means that every font size would need separate hash table. That would be very laborious to make.

# Credits
The scripts were developed as part of master's thesis ["Skripty pro hromadnou úpravu fontů v PDF dokumentech"](https://hdl.handle.net/11012/246071) at [Brno University of Technology](https://www.vut.cz/en/), Faculty of Electrical Engineering and Communications, [Dept. of Telecommunications](https://www.utko.fekt.vut.cz/en). This czech and english manual was created by thesis advisor.

