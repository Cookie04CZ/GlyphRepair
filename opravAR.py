from os import listdir, path
from subprocess import run, CalledProcessError
from json import load, JSONDecodeError
from hashlib import sha256
from argparse import ArgumentParser
from colorama import init  # type: ignore

class Hash:
    @classmethod
    def find_hash_in_json(cls, json_file_path, dir_path, verbose):
        try:
            # Load hashes from the JSON file into a dictionary
            with open(json_file_path, 'r') as file:
                data = load(file)

            # Initialize a set to store hashes from the JSON file
            json_hashes = set()

            # Define a recursive function to collect hashes from nested dictionaries
            def collect_hashes(year_dict):
                for month, value in year_dict.items():
                    if isinstance(value, dict):
                        collect_hashes(value)  # Recursive search for nested dictionaries
                    else:
                        json_hashes.add(value)

            # Iterate through each 'magazine' in the data to collect hashes
            for magazine in data["magazines"].values():
                collect_hashes(magazine)

            # Iterate over all .pdf files in the specified directory and calculate their hashes
            for filename in listdir(dir_path):
                if filename.endswith(".pdf"): # Check if the file extension is .pdf
                    file_path = path.join(dir_path, filename) # Create the full path to the file
                    with open(file_path, 'rb') as file: # Open the file in binary mode for hashing
                        file_content = file.read() # Read the entire file content into memory
                        file_hash = sha256(file_content).hexdigest() # Compute SHA-256 hash of the file content
                        # Compare the file hash with the hashes from the JSON file
                        if file_hash in json_hashes and verbose:
                            run(['Type1toUnicode.exe', '-p', filename, '-f', 'to_unicode.json', '-v'], check=True)
                        elif file_hash in json_hashes:
                            run(['Type1toUnicode.exe', '-p', filename, '-f', 'to_unicode.json'], check=True)
                        else:
                            print(f"\033[33mHash not found for file: {filename}\033[0m")

        except FileNotFoundError:
            print("\033[31mFile does not exist.\033[0m")
        except JSONDecodeError as e:
            print("\033[31mError in parsing JSON file: {json_file_path}\n{e}\033[0m")
        except CalledProcessError:
            print("\033[31mError when running Type1toUnicode.exe\033[0m")
        except Exception as e:
            print(f"\033[31mAn exception occurred: {e}\033[0m")

def main():
    #Colorama initialization
    init()

    # Create an ArgumentParser object for handling command line arguments
    argparser = ArgumentParser()

    # Add an argument for specifying the directory containing PDF files
    # The argument '-d' is a shorthand, '--pdf_dir' is the full name, and it is required
    argparser.add_argument('-d', '--pdf_dir', type=str, required=True, help='Defines the path to directory with .pdf files')

    # Add an argument for specifying the JSON file containing hashes
    # The argument '-f' is a shorthand, '--json_file' is the full name, and it is required
    argparser.add_argument('-j', '--json_file', type=str, required=True, help='Defines the path to the .json file')

    # Add a boolean argument for enabling verbose output
    # This will not require a value; presence in the command line will set it to True
    argparser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose output (prints more information)')
    args = argparser.parse_args()

    # Call the 'find_hash_in_json' method of the 'Hash' class
    # 'magazine_hash.json' specifies the JSON file containing hashes,
    # 'args.pdf_dir' is the directory to search for PDF files,
    # 'args.verbose' controls the verbosity of the output
    Hash.find_hash_in_json(args.json_file, args.pdf_dir, args.verbose)

if __name__ == '__main__':
    main()