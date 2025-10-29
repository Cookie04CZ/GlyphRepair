**This script was originally created to repair encoding in popular czech hobby magazines. If you wish to apply it to other PDF files, please skip to the [next chapter](#before-you-start).**

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

 # Before you start

You're probably here because you have a PDF file with garbled text ("mojibake") - it looks fine on screen, but you get only gibberish when you try to copy+paste it. There are many reasons why text encoding can be wrong in PDF files and our script "Type1toUnicode" can repair only one case. Using the script properly can become a time-consuming task, but you may spare yourself the hassle. Do you really need to permanently fix your PDF files? Or do you merely need to copy some text? If so, there may be a faster way: we've accidentally discovered **that [open-source viewer Evince](https://wiki.gnome.org/Apps/Evince) can return meaningful text even on files that are completely garbled in other PDF viewers** (we tested Adobe Reader, Sumatra PDF, PDF-XChange Viewer, Mozilla Firefox, Google Chrome and others). It's probably because Evince internally uses some sort of heuristics. Nevertheless, even Evince will usually correctly copy only standard ASCII characters (codes 32 to 126); special characters for foreign languages may still be garbled.

Type1toUnicode can permanently fix documents with 100% fidelity and with minimal file size increase, but the PDF files must meet several conditions. Here is a PDF sample before and after repair, provided under [fair use doctrine](https://en.wikipedia.org/wiki/Fair_use):

[T1tU_sample.zip](https://github.com/user-attachments/files/16921939/T1tU_sample.zip)

 # Other ways to fix your documents, but with lower fidelity

If you don't need to preserve document's fidelity, garbled text can be permanently fixed via OCR. Each page is rendered as ordinary bitmap image (it's called "flattening") and then fed to OCR. However, most OCR algorithms still struggle with diacritics, math and/or non-latin characters, so the extracted text usually contains errors. Also, vector graphics may not be preserved, depending on how smart the OCR algorithm is. That's usually highly undesirable and may significantly increase file size. 

There **are** programs that can fix garbled PDFs via OCR while (mostly) preserving vector contents of the document. One of them is open source [Ghostscript by Artifex Software](https://www.ghostscript.com/). A few years ago, they [enhanced its PDFwrite output device](https://ghostscript.readthedocs.io/en/latest/Devices.html#vector-pdf-output-with-ocr-unicode-cmaps) so it behaves similarly to Type1toUnicode. Internally, it uses (also open source) [Tesseract OCR](https://tesseract-ocr.github.io/tessdoc/Installation.html) to recognize the text. Sadly, Artifex doesn't advertise this feature much and there were several bugs that prevented it from working properly. You need to install Ghostscript 10.06.0 or newer if you want to use it. It comes bundled with its own copy of Tesseract OCR, so you don't need to install it separately. On Windows, you'll need to [set up TESSDATA_PREFIX environment variable](https://ghostscript.readthedocs.io/en/latest/Devices.html#ocr-text-output) to a directory with [Tesseract language files of your choice](https://tesseract-ocr.github.io/tessdoc/Data-Files.html). Ghostscript has many options and usually requires several parameters to do what you want. For example, to repair our sample file, it's necessary to use
```
gswin64c -dNOPAUSE -sDEVICE=pdfwrite -sUseOCR=AsNeeded -sOCRLanguage="ces" -dAutoFilterColorImages=false -dColorImageFilter=/FlateEncode -dAutoFilterGrayImages=false -dGrayImageFilter=/FlateEncode  -sOutputFile=sample_repaired_Ghostscript.pdf sample.pdf -c quit
```
Notice the -sUseOCR=AsNeeded parameter. As explained in the [link above](https://ghostscript.readthedocs.io/en/latest/Devices.html#vector-pdf-output-with-ocr-unicode-cmaps), this directs Ghostscript to recognize only text which doesn't have valid Unicode representation. That's the best option in our case, because "our" magazines usually contain at least a few pages of properly encoded text. Therefore, it would be counter-productive to process it with (unreliable) OCR. You can download our [sample fixed with Ghostscript here](https://github.com/user-attachments/files/22067056/sample_repaired_Ghostscript.pdf). If you copy text from it and compare it with sample repaired by our script, you will notice that several characters were recognized wrong. This is mostly caused by how Ghostscript and Tesseract work together; if you're interested, one [Ghostscript developer explained it here](https://bugs.ghostscript.com/show_bug.cgi?id=708548#c1). It also has one nasty side effect: if a character is recognized wrong, the same error then appears in all other instances of that character. But of course that's still better than completely garbled text.

**Be warned that using Ghostscript has many other caveats.** Ghostscript actually completely rebuilds the input file, resulting in an entirely new PDF that only closely resembles the original one. In other words, even though it preserves vector content, it's still much more destructive than Type1toUnicode. This is apparent even in our simple sample, for example the main body font is slightly thicker. Even worse changes can occur with bitmap images, because by default, Ghostscript [recompresses them to optimize file size](https://ghostscript.com/blog/optimizing-pdfs.html). It's complicated to suppress this behavior, but in most cases it can be done via "Distiller Parameters". In our example, there are 4 "Filter" parameters that disable recompression for lossless grayscale and color images. Recompression is disabled by default for lossy (JPEG/DCT) images, but it can still kick in for large images (see PassThroughJPEGImages parameter). [Full list of these Distiller Parameters is here](https://ghostscript.readthedocs.io/en/latest/VectorDevices.html#distiller-parameters), but in our experience they can lead to counter-intuitive results. Be sure to examine your output files closely if you decide to give Ghostscript a try.

# How to run Type1toUnicode

There are actually two scripts in this repository, Type1toUnicode and opravAR. Both are available as Python sources and Windows executables (compiled with [PyInstaller 6.6.0](https://pyinstaller.org/en/stable/)). Python 3.12.3 was used during testing. You will probably need only Type1toUnicode, although what opravAR does is [explained in later chapter](#script-opravar-for-user-friendly-repair). The executables already contain all the necessary libraries, so they run right out the box. You will probably encounter [blue SmartScreen filter warnings](https://github.com/user-attachments/assets/a067c6a6-d85b-4f68-8123-b2fc8f61d345) when you run them for the first time. These warnings vary between Windows versions, either there is "Run anyway" button or you need to click on "More information" first. If you want to run the .py files, you will need following libraries:

* pypdf				4.2.0			https://pypdf.readthedocs.io/en/stable/
* jellyfish			1.0.3			https://github.com/jamesturk/jellyfish
* Levenshtein		0.25.1		https://rapidfuzz.github.io/Levenshtein/
* colorama			0.4.6			https://pypi.org/project/colorama/

All can be installed with pip
```
pip3 install xxxxxxxx
```
Note that the scripts were developed and tested only with these library versions and only on Windows 10. We have no idea if they'd work on other operating systems. 

# Analyzing your PDF files

Since Type1toUnicode has some limitations, first you need to determine whether your PDF file(s) can even be repaired. We specifically designed the script to help you with such an analysis. You don't need to understand the details, but the script can repair only [one PDF font type](https://www.prepressure.com/fonts/basics/type1), that's why we've chosen such a "weird" name. Let's start with its command-line syntax, which is also printed if you run the script without any arguments:

```
the following arguments are required: -p/--pdf_file, -f/--font_map
usage: type1tounicode [-h] -p PDF_FILE -f FONT_MAP [-v]

options:
  -h, --help            show this help message and exit
  -p PDF_FILE, --pdf_file PDF_FILE
                        Defines the path to the .pdf file
  -f FONT_MAP, --font_map FONT_MAP
                        Defines the path to the .json file
  -v, --verbose         Enable verbose output (prints more information)
```
Apart from your input PDF, you'll also need a JSON file with font mapping. Its purpose and structure will be [explained later](#font-map-json-file-and-how-to-create-it), but for starters you can use [multi_ascii.json](multi_ascii.json). It covers most popular fonts names, but contains only mapping for standard ASCII characters (codes 32 to 126). It should mostly work on documents authored with older Adobe products. Unfortunately, that also means it may not work on PDFs from other programs or it may assign wrong character codes. If that happens, repaired text won't be completely garbled anymore, but letters will be randomly swapped (or replaced with spaces). You will need to construct your own JSON file in such case.  

BTW, if [multi_ascii.json](multi_ascii.json) works well on your files, test them with [to_unicode.json](to_unicode.json) next. It has the same base, but covers many more characters for european languages and some dingbat fonts.

When the script finishes, it will print a short statistic like
```
File ABCD.PDF, 76 fonts found, 16 fonts skipped, 40 fonts repaired partially, 20 fonts repaired completely
```
If it manages to repair some font(s), it will create an output PDF file with suffix _repaired (so ABCD_repaired.pdf in our case). The script also generates a subdirectory with log files for every PDF file. The optional -v argument greatly expands these logs to allow for better analysis. You need to enable these verbose logs, so the actual command will look like
```
Type1toUnicode -p ABCD.PDF -f multi_ascii.json -v
```
Note that Type1toUnicode always tries to fix the files, even when you use the -v argument. Therefore, it also generates the output files every time you run it. It's designed to automatically overwrite them, so you don't need to delete them manually. After the initial run, you need to check the (verbose) logs. Real-world documents usually contain fonts of varying types and encodings, sometimes there are dozens of them. Type1toUnicode analyzes every font and logs ones it's unable to repair for whatever reason. Without going into unnecessary technical details, you may encounter several cases:

1. If the script didn't encounter any problems, only the final statistic will appear in the log and it will say "0 fonts skipped, 0 fonts repaired partially, XX font repaired completely". That's very rare, however.

2. Lines "no matching mapping name found in JSON file -> skipping" mean the script could repair such fonts, but it couldn't find proper mapping section in the JSON file. You need to create such a section, or rename existing one.

3. Lines with "no matching font section found in JSON file -> using alternative name" mean that the fonts were repaired using a "recycled" mapping in JSON file. This is actually a really nifty feature which will be [explained later](#font-names-matching-and-alternative-names).

4. Lines with "Glyph XYZ not found in mapping" mean the script repaired the font only partially, because the JSON mapping file is incomplete. Total number of such fonts will also appear in the final statistic. If this happens, you **must** add all missing characters to the JSON file, otherwise they will also miss in the repaired text. Again, this is [explained later](#how-to-find-the-correct-gn-unicode-pairs).

5.  If you encounter these lines, it either means the associated fonts already have proper encoding or the script can't repair them:
* "Font has other type than Type1 -> skipping"
* "ToUnicode already exists -> skipping"
* "FontDescriptor entries missing -> skipping"
* "table Differences does not exist -> skipping"
* "no ToUnicode but Differences incomplete  -> skipping"

Obviously, all such fonts increase the "XX fonts skipped" counter in the final statistic.

6. "no Font objects on the page" happens on pages that contain only images, most commonly in scanned documents. You'd need to use an OCR software to extract text from such pages.

**To sum it up:**
* In case 1, your PDF was already completely repaired.
* If you encounter cases 2 or 4, the fonts can be repaired, but additional work on the JSON file is needed.
* If the script finishes with "0 fonts repaired partially, 0 fonts repaired completely" and there are only cases 5 or 6 in the logs, then you're out of luck.

**Remember: you should always double-check the output PDFs, even when there are no warnings in the log!** Probably the easiest way is to select all text (Ctrl+A) and then paste it to an Unicode-capable plaintext editor (such as Notepad++).

## Analyzing multiple files

If you want to analyze or repair multiple files at once, you can put them in the same directory as the script and run
```
forfiles /m *.pdf /c "cmd /c Type1toUnicode.exe -p @file -f multi_ascii.json -v" 
```
If you want to also recurse into subdirectories, you can use
```
forfiles /s /m *.pdf /c "cmd /c [full path]Type1toUnicode.exe -p @file -f [full path]multi_ascii.json -v"
```
Or alternatively
```
FOR /r %a IN (*.pdf) DO Type1toUnicode -p "%a" -f multi_ascii.json -v
```
You can mass-analyze contents of the log files too, i.e. seach them for occurences of certain messages with [Grep](https://en.wikipedia.org/wiki/Grep) or other pattern-matching utilities.

# Font map JSON file and how to create it

So it seems your PDF file(s) could be repaired, but you need to edit or expand the font map file. It's not hard, but be warned it can get quite laborious. To do this effectively, you should take a while to understand what Type1toUnicode and JSON map file actually do.

## Font names matching and alternative names

In real-world documents, fonts rarely have "clean" names like "Arial". There may be dozens of variants of a given font, each for different size, color, emphasis etc. All fonts must have unique names and PDF authoring programs use different strategies how to make sure they stay unique. In the [sample files](https://github.com/user-attachments/files/16921939/T1tU_sample.zip), actual font names look like
```
GKCMAE+Arial068.313
GKCNLC+Arial.kurz.va058.313
GKCJJF+Arial.tu.n.083.313
GKCJHD+Times.New.Roman.tu.n.083.313
```
As you can see, each name starts with random 6-letter string and ends with numbers that represent font's size (height). Moreover, each font family (e.g. Arial vs. Times) can require different glyph mapping (we've encountered such files). Type1toUnicode is designed to deal with both these issues: The JSON file can have separate sections that can hold different maps; each section must have name of the base font family. That's because Type1toUnicode employs a name-matching algorithm to decide which JSON section will be used to fix the respective font. In other words, it recognizes it should use section "Arial", even when the actual font name is much longer.

Having said that, we've found that different font families usually use the same mapping if the PDF was authored by one program. Obviously, it would be impractical to have multiple copies of the same map in the JSON file, just with different section name. Therefore, every section can also have multiple alternative names, as in the example below. If a match is found with one of these alternative names, that section is used during repair. When this happens, lines "no matching font section found in JSON file -> using alternative name AAA from section BBB" appear in the logs.
```json
  "product": "ToUnicode map",
  "version": 0.8,
  "releaseDate": "2024-05-17T00:00:00.000Z",
  "fonts":[
    {
      "name": "Arial",
      "alternativeNames": ["Bookman", "Courier", "Forte", "Franklin", "Helvetica", "+IBMPCDOS0", "ItcEras", "ItcErasBlackTT",  "Microsoft", "Lucida", "Switzerland", "Tahoma", "Times", "Verdana"],
      "data": {
        "G32": "0020",
        "G33": "0021",
        "G34": "0022",
        "G35": "0023",
```
The name-matching algorithm may fail to find a match if the full font name is too long. In the example above, notice there are two similar alternative names "ItcEras" and "ItcErasBlackTT". They belong to the same font family, but we encountered font name GGCNKL+ItcErasBlackTT058.313 that was so long that "ItcEras" wasn't sufficient to produce a match. Thus we had to add "ItcErasBlackTT" to the list, too. Similar situation happened with IBMPCDOS font, we had to add it as "+IBMPCDOS0". Such situations are announced by "no matching mapping name found in JSON file -> skipping" messages in the (verbose) logs.

Theoretically, the name-matching algorithm may also find wrong matches, i.e. choose wrong map section. We haven't encountered such occurence yet, but there is a possible workaround: sections that are at the top of the JSON file are searched first. So you could fix this by reordering and/or renaming sections within the file.

## Character codes, glyph names and toUnicode tables

The fact that you can copy+paste text from PDFs is more complex than you probably think. What you see on the screen are just glyphs, graphical symbols that may or may not contain information about which alphabet letter they actually represent. Moreover, PDF supports several schemes to reduce overall file size, so it typically stores only glyphs that are needed to render the given document. This is called "embedded subset" fonts. Another file size reduction comes from character ordering. In Type1 fonts, characters are ordered by their appearance in the text. In other words, every font has different character order. Suppose you have a document that starts with word "OUROBOROS", then characters in its font will get these character codes (CC):

| Letter | O |U |R |O |B |O |R |O |S |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Character code (CC) | 1 |2 |3 |1 |4 |1 |3 |1 |5 |

Notice that CC for letter "O" gets repeated every time it's needed. These character codes are linked with glyphs, so the renderer knows what to display at each code position. Glyphs have their own Glyph Names (GN) which may be linked to CCs like this:

| Letter | O | U | R | B | S |
|:---:|:---:|:---:|:---:|:---:|:---:|
| Character code (CC) | 1 | 2 | 3 | 4 | 5 |
| Glyph name (GN) | G79 | G85 | G82 | G66 | G83 |

There is no official or preferred glyph naming scheme, [as you will see later](#glyph-naming-schemes-and-possible-problems). Indeed, we've seen files where GNs were just arbitrary numbers, similar to CCs. But in the magazines we've repaired, the most glyphs had fixed names in Gxxx format. Non-Adobe authoring programs may use different glyph names, but they usually stay fixed, too. **Type1toUnicode can work only because of this fact.**

However, GNs themselves still don't reliably convey information about which letter they represent. It may work only in legacy font encodings like [WinANSI](https://en.wikipedia.org/wiki/Windows-1252) and [MacRoman](https://apple.fandom.com/wiki/Mac-Roman_encoding), but these are limited to about 220 characters, which is insufficient for modern documents. So in 1996, Adobe introduced toUnicode tables into PDF version 1.2. These are separate tables that link character codess with their [Unicode](https://en.wikipedia.org/wiki/Unicode) equivalent. For OUROBOROS, the toUnicode table would look like this:

| Letter | O | U | R | B | S |
|:---:|:---:|:---:|:---:|:---:|:---:|
| Character code (CC) | 1 | 2 | 3 | 4 | 5 |
| toUnicode | 004F | 0055 | 0052 | 0042 | 0053 |

**If you copy+paste garbled text from your PDF, it usually means these toUnicode tables are missing or are generated incorrectly.**

## How Type1toUnicode works internally

Type1toUnicode repairs the documents by generating new toUnicode tables and inserting them into the PDF files. PDF viewers will then prefer it instead of the broken original encoding. As mentioned above, Type1toUnicode leverages the fact that glyphs usually have fixed names for given letters. This is where the JSON mapping file comes in - if it contains correct GN-Unicode pairs, it can be used to construct valid CC-toUnicode tables. Grossly simplified, it works like this:
 
 **Read a CC -> read GN linked to the CC -> look up the GN in JSON mapping file and get its Unicode value -> add new CC-Unicode pair into the toUnicode table.**
 
This process has to be repeated for all CCs in every font, because the PDF standard requires that CC-Unicode table must have exactly the same length as CC-GN table.

## How to find the correct GN-Unicode pairs

This is where the real work begins, because you may need to manually construct the correct mapping. We've found only one effective way how to do it and it requires trial version of Infix PDF Editor:

https://www.iceni.com/infix.htm

Trial is limited to 30 days, but you will be able to use it even after it expires. (The company offers cheap monthly subscription if you wish to support them).

**Rule #1: unless necessary, don't try to repair all the fonts.** Real-world documents may contain dozens of fonts, but usually only one or two of them hold bulk of the text. The rest are for headings, image labels, special symbols (greek, math, dingbat etc.) and other "auxilliary" content. It's pretty common that such fonts contain less than 10 characters and contribute little practical information. We've even encountered "ghost" fonts - they were in the font list, but they weren't used anywhere within the document. Symbol fonts in particular are very laborious to repair, as we can tell from our experience. Nevertheless, [to_unicode.json](to_unicode.json) does contain maps for some such fonts; we were repairing about 200 similar magazines, so the effort was more justified. Remember: if you want to skip some font family, simply don't put its name anywhere within the JSON file.
 
So as the first step, you need to decide which font families you wish to repair and which ones you'll ignore. This is where Infix PDF Editor comes in handy for the first time. If you activate "Edit text" mode and click anywhere into text, its font name will be displayed in the small field on the left.
 
![Infix-click-font](https://github.com/xgmitt00-220814/Type1toUnicode/assets/169207159/c1b98156-0d36-4200-bddf-594ccfe05874)

**Rule #2: if you decide some font family is important, try to make its JSON character map as complete as possible.** You should do this for two reasons:

1. As mentioned earlier, the new toUnicode table must always have the same length as CC-GN table. If a GN is not found within the JSON file, Type1toUnicode replaces it with space (U+0020). In the extreme case (no GNs are found), the script would replace **all** characters with spaces!!

2. If too many characters are undefined, the log files will become flooded with "Glyph Gxxx not found in mapping" messages, particularly if the document contains multiple fonts from the same family. That makes them hard to read and mass-analyze with tools like Grep.

Okay, so how do you find the GN-Unicode pairs in practice? This is the real reason you'll need Infix PDF Editor; it's probably the only user-friendly program that can display how each glyph looks like. Suppose you'll see "Font /GKCKNF+Arial062.5 -> Glyph G232 not found in mapping" in the log. You need to load the PDF into Infix and then select "Text -> Remap fonts" in the main menu. This window will open:

![Infix GN](https://github.com/xgmitt00-220814/Type1toUnicode/assets/169207159/1520b027-6c61-4c5e-a1da-36b199db6c45)

Then you must select the correct font in the drop-down menu in the upper left corner. Infix will display all glyphs that are present in this font. Now you have to find glyph G232, their names (GNs) are displayed in the Raw Glyph field. The problem is, you don't know how the glyph looks like (which letter it represents). And unfortunately, Infix can't sort or filter the glyphs by their name (GN), it always displays them by their CC order. That means you have to manually go through the glyphs until you find G232. In the image, it's small latin letter C with caron, or "č". Now you have to manually find Unicode equivalent for this letter. The simplest way is to google it, but there are also specialized sites with entire Unicode listings, such as [Compart.com](https://www.compart.com/en/unicode/block/U+0100) or [Xah Code](http://xahlee.info/comp/unicode_index.html). These have been really useful when looking up Greek, math and dingbat characters. Either way, letter "č" has Unicode code 010D, which is what you need to add to your JSON file.
```json
        "G232": "010D",
```
Infix has another very useful feature, which is a bit hidden. If you select some text with cursor, then the "Text -> Remap fonts" menu will be replaced with "Text -> Remap selected characters". This will limit the Remap window only to the selected characters, as you can see in the image below. What's even better, characters are now sorted by their order in the *selected* text, not by their (arbitrary) CC. Note that some PDF documents mix different fonts even within single words; if that happens, you need to switch between them in the drop-down menu.

![Infix remap selected](https://github.com/xgmitt00-220814/Type1toUnicode/assets/169207159/69716953-3edb-4833-a71f-8bd2194e57f2)

Notice the small green triangles in the Remap font window. Here Infix displays what it presumes is actual letter the glyph represents. **But don't get fooled by them**, Infix frequently presumes wrong (it wanted to export G232 as "è"). Sometimes the triangles turn red; that occurs when Infix is unable to guess the letter, but it doesn't work consistently. Here is how it looks like for font GKCMAE+Arial068.313 from the [sample file](https://github.com/user-attachments/files/16921939/T1tU_sample.zip):

![Infix-remap-red](https://github.com/xgmitt00-220814/Type1toUnicode/assets/169207159/0490fd32-934c-4104-8224-1a96d1c5b9d9)

Again, don't rely on it. Even when Infix displays some font in green, it usually copy+pastes the text garbled, either completely or for non-ASCII characters.

BTW, if you can't use Infix for whatever reason (you don't use Windows, for example), there is [open source font editor FontForge](https://fontforge.org/) which can display the glyphs, too. On the "Open Font" screen, you need to switch "Filter" to "Extract from PDF". Unfortunately, it doesn't work very well:

1. FontForge is **extremely** picky about correct PDF syntax, it refused to open about **half** of all files we tried.

2. There is no easy way to switch between fonts within one PDF file. You have to close FontForge, run it again, open the same PDF file and choose a different font.

Infix is much faster to use and far more reliable.

## Glyph naming schemes and possible problems

Like we previously mentioned, different fonts and/or PDF authoring programs use different glyph naming schemes. Obviously it's impossible to cover them all, but here is what we've encountered so far. You can see most of them in [to_unicode.json](to_unicode.json), although the czech magazines predominantly used Gxxx scheme.

* First of all, **beware of glyph names with numbers 0 to 31,** like G20 or g3. These frequently aren't fixed and their glyph (and thus Unicode equivalent) changes file from file. There are historical reasons why it happens. In short, ASCII codes 0 to 31 are reserved for unprintable control characters. So in order to optimize file size, some PDF authoring programs replace them with actual printable glyphs. However, this replacement may be arbitrary. If you want to repair such glyphs in multiple documents, you need to cross-compare them to make sure they're really fixed. Theoretically, you could also create a separate JSON file for every PDF document, but that would be very time-consuming.

* **Glyphs G232 and g232 may have different toUnicode mapping!** Type1toUnicode glyph name search subroutine is case-sensitive because of this, and you **must** put exact glyph names into the JSON file. Nevertheless, the mapping is usually identical for standard ASCII (codes 32 to 126), only then it starts to differ. We're not sure which fonts or programs use the gxxx scheme. 

* Glyph numbering may not be decadic, but hexadecimal. An example is section for font family MSTT31, near bottom of [to_unicode.json](to_unicode.json). Notice the hexadecimal numbers equal their Unicode code for standard ASCII characters, but start to differ from 80h (128 decadic) upwards.

* In some fonts, glyphs names are human-readable, such as "zero", "zcaron", "epsilon" and so on. These names are defined in [Adobe Glyph List](https://en.wikipedia.org/wiki/Adobe_Glyph_List) and theoretically, all PDF readers should recognize them. In other words, text from such fonts should always be copied correctly, or at least we haven't seen any document where they weren't.

* In some PDF documents, GNs are simply numbers. These are usually generated arbitrarily and change file by file. Again, technically it would be possible to repair them, but you'd have to prepare a separate JSON file for each document.

## Is there a more effective way to construct the JSON file?
In most cases, yes. While glyph naming may be arbitrary, usually it's derived from some pre-existing encoding. The trick is to find out which one, because there [are many of them](https://www.iana.org/assignments/character-sets/character-sets.xhtml). In olden pre-Unicode times, character encodings were specific to operating systems and languages, due to HW and SW limitations. There were even company-specific and/or file format specific encodings, like PDF's built-in [StandardEncoding](https://en.wikipedia.org/wiki/PostScript_Standard_Encoding), WinAnsiEncoding or MacRomanEncoding. 

So for staters, it's helpful to check your file's metadata for clues which OS and authoring program was used to create it. Then you need to manually identify at least a dozen glyphs (especially codes above 128 decimal), so you can compare it with pre-existing encoding tables. In our case, it was "PageMaker 6.5" and "Acrobat Distiller 4.05 for Windows". And indeed, we soon noticed that for most fonts, glyph order matches [Windows-1250 code page](https://cs.wikipedia.org/wiki/Windows-1250#Mapov%C3%A1n%C3%AD_do_Unik%C3%B3du) which was used by Czech language Windows. However, you shouldn't blindly copy-paste it to your JSON file, because there may be differences. In our case, G194 should be U+00C2 (letter Â) according to Windows-1250, but it's actually used for U+22C5 (dot operator) in the magazines. Another major difference is G234, which was changed from romanian letter "ę" to czech letter "ý". In other words, the editors created their own custom encoding, albeit only slightly different from Windows-1250. Fortunately for us, they sticked with it for almost 20 years. Unfortunately, you can't know that beforehand. In the end, we manually checked all characters in [to_unicode.json](to_unicode.json), one glyph at a time. But remember: it depends on how accurately you want to repair your PDFs. If some characters aren't normally used in your language, you can ignore them. However, we repaired 200 similar magazines, so the effort was more justified.

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

