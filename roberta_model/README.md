# Training and Testing RoBERTa Model

To be able to train and test the NLP model you must follow these steps:

* Firstly, the paths are currently hardcoded to use my shortcode so once you have set up your SLURM folders, change to your shortcode. ALl of these need to be changed to be dynamic to each user instead of hardcoded
* Then navigate to `roberta_model/slurm_out_files` as all SLURM outputs will be saved here
* Then run the line below wherein the model will be trained with the `RoBERTa_config` config for `10` epoch - change these if you want to try different configs or epoch lengths

`sbatch ../bash_scripts/train.sh /vol/bitbucket/es1519/NLPClassification_01/roberta_model/configs/RoBERTa_config.json 10` 

* Once this is complete, find the saved checkpoints which will be under the `/saved/` folder with the SLURM id as the version number.
* Then run the `convert_weights.py` script to convert the checkpoint into a useable format like this: 

`python convert_weights.py --checkpoint {checkpoint_name}`

* Or if you want to convert multiple checkpoints at once, you can run this script in SLURM:

`sbatch ../bash_scripts/convert_folder.sh {folder_location}`

* Then using this new saved checkpoint, you can run the evaluation script from the `/slurm_out/` folder:

`sbatch ../bash_scripts/evaluate_model.sh {saved_checkpoint} {path_to_test_csv} {path_to_config}`

* You can then check the saved JSON file or the SLURM output file to see the results
* You can also evaluate a whole folder of checkpoints at once too using:

`sbatch ../bash_scripts/evaluate_folder.sh {converted_checkpoint_folder} {path_to_test_csv} {path_to_config}`

* This will then store all the results in a JSON file
* You can then use the `plot_model_scores.py` script to view all the scores per epoch. This will create a graph, if the `--save_to` flag is present, of the 4 tracked scores per epoch and also print out the top 5 epochs based on the F1-score, e.g.:

`python model_eval/plot_model_scores.py --json {path_to_json} --save_to {file_to_save_plot}`