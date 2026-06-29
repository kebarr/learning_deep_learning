import sys
import json
import pandas as pd



def parse_and_write_to_file(infile, outfile, correct_species, mode= "incorrect"):
    with open(infile, "r") as f:
        infile_json= json.load(f)

    print("Finding incorrect classification")
    if mode == "incorrect":
        tax_assignments = [read for read in infile_json["Reads"] if read["Species assigned"] !=correct_species]
    else:
        print(f"Looking for correct assigments to {correct_species}") # can't from jsons for non pathogens
        tax_assignments = [read for read in infile_json["Reads"] if read["Species assigned"] ==correct_species]
        print(f"Found {len(tax_assignments)} correctly assigned reads")
    incorrectly_assigned_pathogen = [assignment["Species assigned"] for assignment in tax_assignments]
    
    incorrectly_assigned_read_id = [assignment["Read ID"] for assignment in tax_assignments]
    if len(incorrectly_assigned_pathogen) != len(incorrectly_assigned_read_id):
        raise ValueError("there should be a read ID for each pathogen")
    if len(incorrectly_assigned_pathogen) > 0:
        print(incorrectly_assigned_pathogen[0])
        print(incorrectly_assigned_read_id[0])
        print(f"{len(incorrectly_assigned_pathogen)} misclassified reads found")
        out_json = {"Species_names":[], "Read IDs":[]}
        out_json["Correct species"] = [correct_species for i in range(len(incorrectly_assigned_pathogen))]
        for readid, species in zip(incorrectly_assigned_read_id, incorrectly_assigned_pathogen):
            out_json["Species_names"].append(species)
            out_json["Read IDs"].append(readid)
        out_df = pd.DataFrame.from_dict(out_json)
        out_df.to_csv(outfile)




if __name__ == '__main__':
    correct_species = sys.argv[1]
    infile = sys.argv[2]
    outfile = sys.argv[3]
    mode = "incorrect"
    if len(sys.argv) > 4:
        mode = sys.argv[4]
    print(f"Running in mode {mode}")
    species_for_output_name = correct_species.replace(" ", "_")
    print(f"outfile: {outfile}")
    print("parsing files")
    parse_and_write_to_file(infile, outfile, correct_species, mode)


# python get_misclassified_correctly_classified_reads.py "Brucella abortus" /home/katie/Documents/Dragonfly/running_pipeline/running_dir_Brucella_abortus/output_Brucella_abortus /home/katie/Documents/deep_learning_tutorials/miclassified_reads_all
