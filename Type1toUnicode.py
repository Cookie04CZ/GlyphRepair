import logging, os, datetime
from json import load, JSONDecodeError
from sys import exit
from jellyfish import jaro_winkler_similarity # type: ignore
from Levenshtein import ratio as lev_ratio # type: ignore
from argparse import ArgumentParser
from pypdf import PdfReader, PdfWriter # type: ignore
from pypdf.generic import NameObject, StreamObject # type: ignore
from colorama import init  # type: ignore

NAME = 'Type1toUnicode' #Program name
VERSION = '0.4.0' #Actual version
SUB_TYPE = 'Type1' #Type of the searched fonts
#ToUnicode table template
TEMPLATE = \
"""
/CIDInit /ProcSet findresource begin 12 dict begin begincmap /CIDSystemInfo <<
/Registry ({name}+0) /Ordering (T1UV) /Supplement 0 >> def
/CMapName /{name}+0 def
/CMapType 2 def
1 begincodespacerange <{fchar_hex}> <{lchar_hex}> endcodespacerange
{lchar_dec} beginbfchar
{mapping}
endbfchar
endcmap CMapName currentdict /CMap defineresource pop end end
"""
# Defining the CustomArgumentParser class, which inherits from ArgumentParser
class CustomArgumentParser(ArgumentParser):
    # Overriding the error method from the ArgumentParser class
    def error(self, message):
        # Printing the program name and version along with the error message
        print(f"{NAME} \nversion: {VERSION}\n\n{message}")
        # Printing the help message for the program
        self.print_help()
        # Exiting the program with an error code 1
        self.exit(1)

class UnicodeMapper:
    # Defining a class method to get a unicode value
    @classmethod
    def get_unicode_value(cls, data, font_name, property_name):
        # Looping through each font in the 'fonts' list within the data dictionary
        for font in data.get('fonts', []):
            # Checking if the current font's name matches the provided font_name
            if font.get('name') == font_name:
                # Returning the unicode value for the given property_name if found
                return font.get('data', {}).get(property_name, None)
        # Returning None if the font or property_name is not found
        return None
    
    # Defining a class method to find a similar font based on the search_font name
    @classmethod
    def find_similar_font(cls, font_dict, search_font):
        # Initializing variables to track the highest similarity score and the corresponding font names
        similarity, return_font, f_name = 0, None, None 
        # Looping through each font name and its mapped name in the font_dict dictionary
        for font_name, mapped_name in font_dict.items():
        # Calculating the Jaro-Winkler similarity between the font_name and search_font
            jw_sim = jaro_winkler_similarity(font_name, search_font) * 100
        # Updating the variables if the similarity is higher than the current highest similarity and at least 70
            if (jw_sim > similarity) and jw_sim >= 70:
                similarity, f_name, return_font = jw_sim, font_name, mapped_name
        # Updating the variables if the Jaro-Winkler similarity is less than 45 but the Levenshtein ratio is greater than 40
            if (jw_sim < 45 and (lev_ratio(font_name, search_font) * 100) > 40):
                similarity, f_name, return_font = jw_sim, font_name, mapped_name
    # Returning the font name with the highest similarity and its mapped name
        return f_name, return_font  
    
class File:

    @classmethod
    # Defining a class method to load a JSON file
    def load_json(cls, filename, encoding='utf-8'):
        # Opening the specified JSON file with the given encoding in read mode
        with open(filename, 'r', encoding=encoding) as json_file:
            # Loading and returning the content of the JSON file
            return load(json_file)

    @classmethod
    # Defining a class method to validate a file's existence and extension
    def validate(cls, filename, extension):
        try:
            # Checking if the file exists
            if not os.path.exists(filename):
                # Printing an error message in red if the file does not exist and exiting with code 2
                print(f"\033[31mFile does not exist: {filename}\033[0m")
                exit(2)

            # Getting the file extension
            _, file_extension = os.path.splitext(filename)
            # Checking if the file extension matches the expected extension
            if file_extension.lower() != extension.lower():
                # Printing an error message in red if the file format is not as expected and exiting with code 3
                print(f"\033[31mFile is not in expected format. Expected {extension}\033[0m")
                exit(3)

            # If the file is a JSON file, attempting to open and load it
            if file_extension.lower() == '.json':
                with open(filename, 'r', encoding='utf-8') as json_file:
                    load(json_file)
        # Handling JSON decoding errors
        except JSONDecodeError as e:
            print(f"\033[31mError in parsing JSON file: {filename}\n{e}\033[0m")
            exit(4)
        # Handling any other exceptions
        except Exception as e:
            print(f"\033[31mAn exception occurred: {e}\033[0m")
            exit(5)
    
    @classmethod
    # Defining a class method to update metadata
    def update_metadata(cls, existing_metadata):
        # Getting the current date and time
        now = datetime.datetime.now()
        # Formatting the current date and time in a specific format
        formatted_time = 'D:' + now.strftime('%Y%m%d%H%M%S')

        # Creating a dictionary with new metadata values
        new_metadata = {
            '/ModDate': formatted_time, # Adding the formatted current date and time as modification date
            '/Producer': NAME + " " + VERSION # Adding the producer name and version of program
        }
        # Creating a copy of the existing metadata
        updated_metadata = existing_metadata.copy()
        # Updating the existing metadata with the new metadata
        updated_metadata.update(new_metadata)
        return updated_metadata

def main():
    #Colorama initialization
    init()
    # Initializing counters for skipped, partialy repaired, and fully repaired fonts
    cnt_skipped = cnt_rep_part = cnt_rep_comp = 0
    # Initializing a variable to store the unicode error flag (initially set to False)
    error_unicode = False
    # Creating an instance of CustomArgumentParser
    argparser = CustomArgumentParser()
    # Adding a required arguments for PDF file path, font map file path and to enable verbose output
    argparser.add_argument('-p', '--pdf_file', type=str, required=True, help='Defines the path to the .pdf file')
    argparser.add_argument('-f', '--font_map', type=str, required=True, help='Defines the path to the .json file')
    argparser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose output (prints more information)')
    # Parsing the command-line arguments
    args = argparser.parse_args()

    # Validating the PDF file and font map file
    File.validate(args.pdf_file, '.pdf')
    File.validate(args.font_map, '.json')

    # Creating a PdfReader object to read the specified PDF file
    reader = PdfReader(args.pdf_file)
    # Accessing the metadata of the PDF file
    metadata = reader.metadata
    
    # Creating a PdfWriter object to write to a repaired PDF file
    writer = PdfWriter()

    # Creating a logger object for the "PdfRepair" logger
    logger = logging.getLogger("PdfRepair")
    # Setting the logging level to DEBUG
    logger.setLevel(logging.DEBUG)
    # Defining the log directory path
    log_directory = os.path.join (os.getcwd(), 'Log')
    # Creating a file handler for logging, setting the log file name based on the PDF file name
    file_handler = logging.FileHandler(os.path.join(log_directory, f'{args.pdf_file[:-4]}_log.txt'), mode='w', delay=True)
    # Creating a formatter for the log messages
    formatter = logging.Formatter('%(message)s')
    # Creating a formatter for the log messages
    file_handler.setFormatter(formatter)
    # Adding the file handler to the logger
    logger.addHandler(file_handler)
    # Checking if the log directory exists and creating new if it does not exist
    if not os.path.exists(log_directory):
        os.makedirs(log_directory)

    # Loading the JSON data from the font map file
    json_data = File.load_json(args.font_map)
    # Initializing an empty dictionary to store font mappings
    font_dict = {}
    # Initializing an empty set to keep track of analyzed fonts
    analyzed_fonts = set()

    # Iterating over each font in the JSON data
    for font in json_data['fonts']:
        # Mapping the font's name to itself in the font dictionary
        font_dict[font['name']] = font['name']

        if 'alternativeNames' in font:
            # Iterating over each alternative name
            for alt_name in font['alternativeNames']:
                # Mapping each alternative name to the primary font name in the font dictionary
                font_dict[alt_name] = font['name']

    # Iterating over each page in the PDF
    for pagenum, page in enumerate(reader.pages):
        # Checking if the page contains any font objects
        if '/Font' not in page['/Resources']:
            # Logging and skipping the page if no font objects are found (if verbose is enabled)
            if args.verbose:
                    logger.debug('Page "%s" -> no Font objects on the page -> skipping', pagenum+1)
            continue

        # Iterating over each font object on the page
        for font, data in page['/Resources']['/Font'].items():
            _dataobj = data.get_object() # Getting the font object
            _subtype = _dataobj['/Subtype'] # Getting the subtype of the font
            # Checking if the font subtype is not Type1
            if SUB_TYPE not in _subtype:
                if args.verbose:
                    # Logging and skipping the font if it is not Type1 (if verbose is enabled)
                    if '/BaseFont' in _dataobj:
                        _fontname = _dataobj['/BaseFont']
                        logger.debug('Page "%s" -> Font "%s" has other type than Type1 -> "%s" -> skipping', pagenum+1, _fontname, _subtype)
                    else:
                        logger.debug('Page "%s" -> Font has other type than Type1 -> "%s" -> skipping', pagenum+1, _subtype)
                # Incrementing the counter for skipped fonts
                cnt_skipped += 1
                continue

            _fontname = _dataobj['/BaseFont'] # Getting the base font name
            # Checking if the font has already been analyzed
            if _fontname in analyzed_fonts:
                continue
            else:
                analyzed_fonts.add(_fontname) # Adding the font to the set of analyzed fonts

            # Checking if the font encoding or differences table does not exist
            if '/Encoding' not in _dataobj or '/Differences' not in _dataobj['/Encoding']:
                # Logging and skipping the font if the differences table does not exist (if verbose is enabled)
                if args.verbose:
                    logger.debug('Page "%s" -> Font "%s" -> table Differences does not exist -> skipping', pagenum+1, _fontname)
                # Incrementing the counter for skipped fonts
                cnt_skipped += 1
                continue
            
            # Checking if the font descriptor entries are missing
            if '/FirstChar' not in _dataobj or '/LastChar' not in _dataobj:
                # Logging and skipping the font if the font descriptor entries are missing (if verbose is enabled)
                if args.verbose:
                    logger.debug('Page "%s" -> Font "%s" -> FontDescriptor entries missing -> skipping', pagenum+1, _fontname)
                # Incrementing the counter for skipped fonts
                cnt_skipped += 1
                continue

            fchar = _dataobj['/FirstChar'] # Getting the value of first character object
            lchar = _dataobj['/LastChar'] # Getting the value of last character object

            # Checking if the differences table is incomplete
            if ((lchar-fchar)+2) != len(_dataobj['/Encoding']['/Differences']):
                # Logging and skipping the font if the differences table is incomplete (if verbose is enabled)
                if args.verbose:
                    logger.debug('Page "%s" -> Font "%s" -> no ToUnicode but Differences incomplete -> skipping', pagenum+1, _fontname)
                # Incrementing the counter for skipped fonts
                cnt_skipped += 1
                continue
            
            # Checking if the ToUnicode entry already exists
            if '/ToUnicode' in _dataobj:
                # Logging and skipping the font if the ToUnicode entry already exists (if verbose is enabled)
                if args.verbose:
                    logger.debug('Page "%s" -> Font "%s" -> ToUnicode already exists -> skipping', pagenum+1, _fontname)
                # Incrementing the counter for skipped fonts
                cnt_skipped += 1
                continue
            
            # Finding a similar font name using the UnicodeMapper function find_similar_font
            alt_fontname, mapped_fontname = UnicodeMapper.find_similar_font(font_dict, _fontname)
            if mapped_fontname is None:
                 # Logging and skipping the font if no matching mapping name is found in the JSON file (if verbose is enabled)
                if args.verbose:
                    logger.debug('Page "%s" -> Font "%s" -> no matching mapping name found in JSON file -> skipping', pagenum+1, _fontname)
                # Incrementing the counter for skipped fonts
                cnt_skipped += 1
                continue
            
            # Logging if an alternative name is used from the JSON file (if verbose is enabled)
            if mapped_fontname is not None and mapped_fontname not in _fontname:
                logger.debug('Page "%s" -> Font "%s" -> no matching font section found in JSON file -> using alternative name "%s" from section: "%s"', pagenum+1, _fontname, alt_fontname, mapped_fontname)

            fchar_hex = f"{fchar:X}".zfill(2) # Formatting the first character value as a hexadecimal string
            lchar_hex = f"{lchar:X}".zfill(2) # Formatting the last character value as a hexadecimal string
            name = f"PAGE{pagenum+1}+{font[1:]}" # Creating a name for the font map
            _mapping = []  #Initializing the mapping list

            # Iterating over the differences table to create the unicode mapping
            for idx, char in enumerate(_dataobj['/Encoding']['/Differences'][1:]):
                # Checking if the character is not a string
                if not isinstance(char,str):                
                    char = str(char) # Converting the character to a string
                else:
                    char = char[1:] # Removing the leading '/' from the character string

                # Formatting the index as a hexadecimal string
                idx = (f"{(idx+fchar):X}".zfill(2))
                # Getting the unicode value for the character using the UnicodeMapper function get_unicode_value
                unicode_value = UnicodeMapper.get_unicode_value(json_data, mapped_fontname, char)
                
                # Checking if the unicode value is not found
                if unicode_value is None:
                    # Logging that the glyph is not found in the mapping (if verbose is enabled)
                    logger.debug('Page "%s" -> Font "%s" -> Glyph "%s" not found in mapping', pagenum+1, _fontname, char)
                    # Adding a mapping for the character to a space (U+0020)
                    _mapping.append(f"<{idx}> <0020>")
                    # Setting the error_unicode flag to True
                    error_unicode = True
                    continue

                # Adding the unicode mapping for the character
                _mapping.append(f"<{idx}> <{unicode_value}>")

            # Incrementing the respective counters based on whether an error occurred in the unicode mapping
            if error_unicode is True:
                cnt_rep_part += 1
                error_unicode = False
            else:
                cnt_rep_comp +=1

            # Joining the mapping list into a string
            _mapping = '\n'.join(_mapping)

            # Creating a StreamObject for the ToUnicode mapping
            _stream_data = StreamObject()
            # Setting the data for the stream object
            _stream_data.set_data(
                bytes(
                    TEMPLATE.format(
                        name=name, # The name for the ToUnicode map
                        fchar_hex=fchar_hex, # The first character in hexadecimal format
                        lchar_hex=lchar_hex, # The last character in hexadecimal format
                        lchar_dec=(lchar-fchar)+1, # The number of characters
                        mapping=_mapping # The unicode mapping
                    ),
                    encoding='utf-8' # Encoding the data as UTF-8
                )
            )
            # Compressing the stream data using Flate encoding
            _stream_data.flate_encode()

            # Updating the font object with the new ToUnicode mapping
            _dataobj.get_object().update(
                {
                    NameObject('/ToUnicode'): _stream_data # Adding the ToUnicode stream object to the font object
                }
            )
    
    # Checking if any fonts were repaired completely or partially
    if cnt_rep_comp != 0 or cnt_rep_part !=0:
        # Adding each page from the reader to the writer
        for page in reader.pages:
                 writer.add_page(page)

        # Updating the metadata of the PDF file
        writer.add_metadata(File.update_metadata(metadata))
        # Creating the output file name for the repaired PDF
        outfile = args.pdf_file.split('.')[0] + '_repaired.pdf'
        # Writing the repaired PDF to the output file
        with open(outfile, 'wb') as output_pdf:
            writer.write(output_pdf)

        # Logging details about the repaired fonts (if verbose is enabled)
        if args.verbose:
            logger.debug('File "%s", "%s" fonts found, "%s" fonts skipped, "%s" fonts repaired partially, "%s" fonts repaired completely', args.pdf_file, (cnt_skipped+cnt_rep_part+cnt_rep_comp), cnt_skipped, cnt_rep_part, cnt_rep_comp)
        # Printing a warning if some fonts were repaired partially
        if cnt_rep_part > 0:
            print(f"\033[33mSome font(s) have undefined character(s) mapping, please see log file {args.pdf_file[:-4]}_log.txt in Log directory.\033[0m")
    else:
        # Logging that no output PDF file was created
        logger.debug('No output PDF file created!')
        # Printing a message that no output PDF file was created
        print("\033[33mNo output PDF file created!\033[0m'")
    
    # Printing a summary of the font repair process
    print(f"\033[32mFile {args.pdf_file}, {cnt_skipped+cnt_rep_part+cnt_rep_comp} fonts found, {cnt_skipped} fonts skipped, {cnt_rep_part} fonts repaired partially, {cnt_rep_comp} fonts repaired completely\033[0m")

# Checking if the script is being run as the main module
if __name__ == '__main__':
    # Calling the main function to execute the script
    main()