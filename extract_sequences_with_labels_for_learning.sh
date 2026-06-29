#!/bin/bash

# Check if a FASTQ file was provided
if [ -z "$1" ] || [ ! -f "$1" ]; then
    echo "Usage: $0 <path_to_fastq_file> [True|False]"
    echo "Default label is False if not specified."
    exit 1
fi

FASTQ_FILE="$1"
LABEL="${2:-False}" # Defaults to False if you don't provide a second argument

# Determine output filename (e.g., all_misclassified_species.fastq -> all_misclassified_species_prepared.csv)
OUTPUT_FILE="${FASTQ_FILE%.fastq}_prepared.csv"

echo "Extracting sequences from: $FASTQ_FILE"
echo "Assigning label:           $LABEL"

# Write the CSV Header expected by fastai
echo "sequence,label" > "$OUTPUT_FILE"

# Use awk to pull every 4th line starting from line 2 (the sequence line)
# and format it nicely into a CSV row
awk -v lbl="$LABEL" 'NR%4==2 {print $0 "," lbl}' "$FASTQ_FILE" >> "$OUTPUT_FILE"

echo "[+] Successfully created dataset for fastai: $OUTPUT_FILE"

# How to use it:

#     Save it as fastq_to_fastai.sh and make it executable: chmod +x fastq_to_fastai.sh

#     Run it on your master misclassified file:
#     Bash

#     ./fastq_to_fastai.sh all_misclassified_Bacillus_anthracis.fastq False

# This will output a file named all_misclassified_Bacillus_anthracis_prepared.csv that looks exactly like this:

# final data pre is then e.g.
# cat correctly_assigned/Shigella_boydii_correctly_assigned_prepared.csv incorrectly_assigned/all_misclassified_Shigella_boydii_prepared.csv | shuf > Shigella_boydii_all_correctly_incorrectly_assigned.csv
