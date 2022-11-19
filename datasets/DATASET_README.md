### Step 1
Download the following annotations and images from the Visual Genome website http://visualgenome.org/api/v0/api_home.html and put them under the datasets/vg directory.

	- datasets
	  - vg
	    - annottions
	      - image_data.json
	      - objects.json
	      - relationships.json
	    - images
	      - VG_100K
		- 2377357.jpg
		- 2377380.jpg
		- ...
	      - VG_100K_2
		- 2417985.jpg
		- 2417973.jpg
		- ...

### Step 2 
Execute ```./preprocess.py``` to pre-process the dataset annotations, and the following two files will be saved to the datasets/vg/annottions directory.
We implement in our own code the same dataset pre-processing techniques in https://github.com/danfeiX/scene-graph-TF-release/tree/master/data_tools.
No GPU is required at this step.
#### You can SKIP this step by directly using the processed annotations provided in step 3.

	- datasets
	  - vg
	    - annottions
	      - instances_vg_train.json
	      - instances_vg_test.json
	      - ...
            - images
	      ...

	We provide the pre-processed dataset annotations here
	instances_vg_train: https://drive.google.com/file/d/1xEUk6jqZtKE0Myrs1I-lPoQZ3-2sTCjF/view?usp=sharing
	instances_vg_test.json: https://drive.google.com/file/d/1qkiXDw9sMsyCOMB-avBYnF8VGIqds0_k/view?usp=sharing


### Step 3 - Execute ```./prepare_datasets.py``` to build the dataloader offline to speed up the later training process.
	Pre-processed annotations in step 2 will be loaded to form an annotation for each image data ready to be used in the scene graph training process.
	One GPU is required at this step.
	Each annotation for each image will be saved under the datasets/vg_scene_graph_annot directory as follows

	- datasets
	  - vg
			...
		- vg_scene_graph_annot
			- VG_100K
				- 2377357_annotations.pkl
			  - 2377380_annotations.pkl
			  - ...
			- VG_100K_2
				- 2417985_annotations.pkl
			  - 2417973_annotations.pkl
			  - ...
			...

	#### We provide all processed annotations here (it takes 2.09GB after unzipping)
	https://drive.google.com/file/d/1hPLP-6Ub7s7zCthrfO-C2b_f_agXCiGv/view?usp=sharing


### The final dataset directory should have the following structure
	- datasets
	  - vg
			- annottions
			  - image_data.json
			  - objects.json
			  - relationships.json
			  - instances_vg_train.json
			  - instances_vg_test.json
			- images
			  - VG_100K
			    - 2377357.jpg
			    - 2377380.jpg
			    - ...
			  - VG_100K_2
			  	- 2417985.jpg
			  	- 2417973.jpg
			  	- ...
		- vg_scene_graph_annot
			- VG_100K
				- 2377357_annotations.pkl
			  - 2377380_annotations.pkl
			  - ...
			- VG_100K_2
				- 2417985_annotations.pkl
			  - 2417973_annotations.pkl
			  - ...
			...




