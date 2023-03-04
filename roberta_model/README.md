# Training and Testing RoBERTa Model

To be able to train and test the NLP model you must follow these steps:

* Firstly, the paths are currently hardcoded to use my shortcode so once you have set up your SLURM folders, change to your shortcode. ALl of these need to be changed to be dynamic to each user instead of hardcoded
* Then navigate to `roberta_model/slurm_out_files` as all SLURM outputs will be saved here
* Then run the line below wherein the model will be trained with the `RoBERTa_config` config for `1` epoch - change these if you want to try different configs or epoch lengths

`sbatch ../bash_scripts/train.sh /vol/bitbucket/es1519/NLPClassification_01/roberta_model/configs/RoBERTa_config.json 1` 

* Once this is complete, find the saved checkpoint which will be under the `/saved/` folder with the SLURM id as the version number.
* Then run the `convert_weights.py` script to convert the checkpoint into a useable format like this: 

`python convert_weights.py --checkpoint {checkpoint_name} --save_to {save_location}`

* Then using this new saved checkpoint, you can run the evaluation script from the `/slurm_out/` folder:

`sbatch ../bash_scripts/evaluate_model.sh {saved_checkpoint} {path_to_test_csv} {path_to_config}`

* You can then check the saved JSON file or the SLURM output file to see the results
* If you want to run inference testing then use the `run_prediction.py` script like so: TODO: Not working yet

`python run_prediction.py --from_ckpt_path {saved_checkpoint} --input "Hello, World!"`